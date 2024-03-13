import collections
import logging
from collections import deque
from typing import Union

import fancy_gym
import numpy as np
import torch as ch

from trust_region_projections.algorithms.abstract_algo import AbstractAlgorithm
from trust_region_projections.losses.loss_factory import get_value_loss_and_critic
from trust_region_projections.models.policy.abstract_gaussian_policy import AbstractGaussianPolicy
from trust_region_projections.models.policy.policy_factory import get_policy_network
from trust_region_projections.models.value.critic import BaseCritic
from trust_region_projections.projections.base_projection_layer import BaseProjectionLayer
from trust_region_projections.projections.projection_factory import get_projection_layer
from trust_region_projections.sampling.dataclass import TrajectoryOnPolicy
from trust_region_projections.sampling.trajectory_sampler import TrajectorySampler
from trust_region_projections.utils.custom_store import CustomStore
from trust_region_projections.utils.network_utils import get_lr_schedule, get_optimizer
from trust_region_projections.utils.torch_utils import flatten_batch, generate_minibatches, get_numpy, select_batch, \
    tensorize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('policy_gradient')


class PolicyGradient(AbstractAlgorithm):
    def __init__(self,
                 sampler: TrajectorySampler,
                 policy: AbstractGaussianPolicy,
                 critic: BaseCritic,

                 optimizer_policy: str = "adam",
                 optimizer_critic: str = None,
                 lr_policy: float = 3e-4,
                 lr_critic: float = 3e-4,

                 projection: BaseProjectionLayer = None,

                 train_steps: int = 1000,
                 epochs: int = 10,
                 epochs_critic: int = 10,
                 n_minibatches: int = 4,

                 lr_schedule: str = "",
                 clip_grad_norm: Union[float, None] = 0.5,

                 critic_coeff: float = 0.0,
                 max_entropy_coeff: float = 0.0,
                 entropy_penalty_coeff: float = 0.0,

                 n_training_samples: int = 2048,
                 discount_factor: float = 0.99,
                 use_gae: bool = True,
                 gae_scaling: float = 0.95,

                 norm_advantages: Union[bool, None] = True,
                 clip_advantages: Union[float, None] = None,

                 importance_ratio_clip: Union[float, None] = 0.2,
                 clip_critic: Union[float, None] = 0.2,

                 store: CustomStore = None,
                 verbose: int = 1,
                 evaluate_deterministic: bool = True,
                 evaluate_stochastic: bool = False,
                 log_interval: int = 1,
                 save_interval: int = -1,

                 seed: int = 1,
                 cpu: bool = True,
                 dtype: ch.dtype = ch.float32,
                 ):
        """
        Basic policy gradient that can be used for PPO and with Projections layers.
        Args:
            sampler: Takes care of generating trajectory samples.
            policy: An `AbstractPolicy` which maps observations to action distributions.
                    Normally ConditionalGaussianPolicy is used.
            critic: An `AbstractPolicy` which returns the value prediction for input states.
                    Normally ConditionalGaussianPolicy is used.
            optimizer_policy: Optimizer to use for the agent and vf.
            optimizer_critic: Different vf optimizer if training separately.
            lr_policy: Learning rate for actor or joint optimizer.
            lr_critic: Learning rate for (optional) vf optimizer.
            train_steps: Total number of training steps.
            epochs: Number of policy updates for each batch of sampled sampling.
            epochs_critic: Number of vf updates for each batch of sampled sampling.
            n_minibatches: Number of minibatches for each batch of sampled sampling.
            lr_schedule: Learning rate schedule type: 'linear' or ''
            clip_grad_norm: Gradient norm clipping.
            max_entropy_coeff: Coefficient when complementing the reward with a max entropy objective.
            critic_coeff: Multiplier for vf loss to balance with policy gradient loss.
                    Default to `0.0` trains vf and policy separately.
                    `0.5` , which was used by OpenAI trains jointly.
            entropy_penalty_coeff: Coefficient for entropy regularization loss term.

            n_training_samples: Number of rollouts per environment (Batch size is n_training_samples * n_envs)
            discount_factor: Discount factor for return computation.
            use_gae: Use generalized advantage estimation for computing per-timestep advantage.
            gae_scaling: Lambda parameter for TD-lambda return and GAE.
            norm_advantages: If `True`, standardizes advantages for each update.
            clip_advantages: Value above and below to clip normalized advantages.
            importance_ratio_clip: Epsilon in clipped, surrogate PPO objective.
            clip_critic: Difference between new and old value predictions are clipped to this threshold.
            store: Cox store
            verbose: Add more logging output.
            log_interval: How often to log.
            save_interval: How often to save model.
            seed: Seed for generating envs
            dtype: Data type to use, either of float32 or float64. The later might be necessary for higher
                    dimensions in order to learn the full covariance.
            cpu: Compute on CPU only.
        """

        super().__init__(policy, sampler, projection, train_steps, clip_grad_norm, store, verbose,
                         evaluate_deterministic, evaluate_stochastic, log_interval=log_interval,
                         save_interval=save_interval, seed=seed, cpu=cpu, dtype=dtype)

        # training
        self.epochs = epochs
        self.val_epochs = epochs_critic
        self.n_minibatches = n_minibatches

        # normalizing and clipping
        self.norm_advantages = norm_advantages
        self.clip_advantages = clip_advantages
        self.clip_critic = clip_critic
        self.importance_ratio_clip = importance_ratio_clip

        # GAE
        self.use_gae = use_gae
        self.gae_scaling = gae_scaling

        # loss parameters
        self.discount_factor = discount_factor
        self.max_entropy_coeff = max_entropy_coeff
        self.vf_coeff = critic_coeff
        self.entropy_coeff = entropy_penalty_coeff

        # environment
        self.n_training_samples = n_training_samples

        # vf model
        self.critic = critic

        # optimizer
        self.optimizer_policy = get_optimizer(optimizer_policy, self.policy.parameters(), lr_policy, eps=1e-5)
        self.lr_schedule_policy = get_lr_schedule(lr_schedule, self.optimizer_policy, self.train_steps)
        if critic:
            self.optimizer_critic = get_optimizer(optimizer_critic, self.critic.parameters(), lr_critic, eps=1e-5)
            self.lr_schedule_critic = get_lr_schedule(lr_schedule, self.optimizer_critic, self.train_steps)

        if self.store:
            self.setup_stores()

    def setup_stores(self):
        """
        Setup cox stores for saving.
        Returns:

        """
        # Logging setup
        super(PolicyGradient, self).setup_stores()

        loss_dict = {
            'total_loss': float,
            'critic_loss': float,
            'policy_loss': float,
            'entropy_loss': float,
            'trust_region_loss': float,
        }

        if self.projection.do_regression:
            loss_dict.update({'regression_loss': float})

        self.store.add_table('loss', loss_dict)

        if self.verbose >= 1:
            self.store.add_table('stats', {
                "values": float,
                "ratio": float,
                "returns": float,
                'old_values': float
            })

        if self.lr_schedule_policy:
            lr_dict = {}
            lr_dict.update({f"lr": float})

            if self.lr_schedule_critic:
                lr_dict.update({f"lr_vf": float})
            self.store.add_table('lr', lr_dict)

    def advantage_and_return(self, rewards: ch.Tensor, values: ch.Tensor, dones: ch.Tensor,
                             time_limit_dones: ch.Tensor):
        """
        Based on: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/storage.py#L66
        License: MIT

        Calculate advantage (with GAE) and discounted returns.
        Further, we provide specific treatment for terminal states which reached max horizon.

        GAE: delta_t^V = r_t + discount * V(s_{t+1}) - V(s_t)
        with
        V(s_t+1) = {0 if s_t is terminal
                   {v_s_{t+1} if s_t not terminal and t != T (last step)
                   {v_s if s_t not terminal and t == T

        Args:
            rewards: Rewards from environment
            values: Value estimates
            dones: Done flags for true termination
            time_limit_dones: Done flags for reaching max horizon

        Returns:
            advantages and returns

        """
        returns = ch.zeros_like(values)
        masks = ~dones
        time_limit_masks = ~time_limit_dones

        if self.use_gae:
            gae = 0
            for step in reversed(range(rewards.size(0))):
                pcont = self.discount_factor * masks[step]
                td = rewards[step] + pcont * values[step + 1] - values[step]
                gae = td + pcont * self.gae_scaling * gae
                gae = gae * time_limit_masks[step]
                returns[step] = gae + values[step]
        else:
            returns[-1] = values[-1]
            for step in reversed(range(rewards.size(0))):
                pcont = self.discount_factor * masks[step]
                returns[step] = time_limit_masks[step] * (rewards[step] + pcont * returns[step + 1]) + \
                                time_limit_dones[step] * values[step]

        returns = returns[:-1]
        advantages = returns - values[:-1]

        return advantages.clone().detach(), returns.clone().detach()

    def surrogate_loss(self, advantages: ch.Tensor, new_logpacs: ch.Tensor, old_logpacs: ch.Tensor):
        """
        Computes the surrogate reward for IS policy gradient R(\theta) = E[r_t * A_t]
        Optionally, we support clamping the ratio (for PPO) R(\theta) = E[clamp(r_t, 1-e, 1+e) * A_t]
        Args:
            advantages: unnormalized advantages
            new_logpacs: the log probabilities from current policy
            old_logpacs: the log probabilities
        Returns:
            The surrogate loss as described above
        """

        # Normalized Advantages
        if self.norm_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # topk = advantages.topk(50, 0, sorted=False)[0]
            # advantages = ((advantages - topk.mean()) / (topk.std() + 1e-8)).clamp(min=-3)

        if self.clip_advantages > 0:
            advantages = ch.clamp(advantages, -self.clip_advantages, self.clip_advantages)

        # Ratio of new probabilities to old ones
        ratio = (new_logpacs - old_logpacs).exp()

        surrogate_loss = ratio * advantages

        # PPO clipped ratio
        if self.importance_ratio_clip > 0:
            ratio_clipped = ratio.clamp(1 - self.importance_ratio_clip, 1 + self.importance_ratio_clip)
            surrogate_loss2 = ratio_clipped * advantages
            surrogate_loss = ch.min(surrogate_loss, surrogate_loss2)

        return -surrogate_loss.mean(), collections.OrderedDict(ratio=ratio.mean())

    def value_loss(self, values: ch.Tensor, returns: ch.Tensor, old_vs: ch.Tensor):
        """
        Computes the value function loss.

        When using GAE we have L_t = ((v_t + A_t).detach() - v_{t})
        Without GAE we get L_t = (r(s,a) + y*V(s_t+1) - v_{t}) accordingly.

        Optionally, we clip the value function around the original value of v_t

        Returns:
        Args:
            values: value estimates
            returns: computed returns with GAE or n-step
            old_vs: old value function estimates from behavior policy

        Returns:
            Value function loss
        """

        vf_loss = (returns - values).pow(2)

        if self.clip_critic > 0:
            # In OpenAI's PPO implementation, we clip the value function around the previous value estimate
            # and use the worse of the clipped and unclipped versions to train the value function
            vs_clipped = old_vs + (values - old_vs).clamp(-self.clip_critic, self.clip_critic)
            vf_loss_clipped = (vs_clipped - returns).pow(2)
            vf_loss = ch.max(vf_loss, vf_loss_clipped)

        return vf_loss.mean()

    def update_policy(self, dataset: TrajectoryOnPolicy):
        """
        Policy Optimization step
        Args:
            dataset: NameTuple with obs, actions, logpacs, returns, advantages, values, and q_values
        Returns:
            Loss dict with total loss, policy_loss, entropy_loss, and delta_loss
        """

        obs, actions, old_logpacs, returns, advantages, q = \
            dataset.obs, dataset.actions, dataset.logpacs, dataset.returns, dataset.advantages, dataset.q

        # losses, vf_losses, surrogates, entropy_losses, trust_region_losses = \
        #     [tensorize(0., self.cpu, self.dtype) for _ in range(5)]
        loss_dict = collections.defaultdict(lambda: tensorize(0., True, self.dtype))

        if self.verbose >= 1:
            stats_dict = collections.defaultdict(lambda: tensorize(0., True, self.dtype))

        # set initial entropy value in first step to calculate appropriate entropy decay
        if self.projection.initial_entropy is None:
            self.projection.initial_entropy = self.policy.entropy(q).mean()

        for _ in range(self.epochs):
            batch_indices = generate_minibatches(obs.shape[0], self.n_minibatches)

            # Minibatches SGD
            for indices in batch_indices:
                batch = select_batch(indices, obs, actions, old_logpacs, advantages, q[0], q[1])
                b_obs, b_actions, b_old_logpacs, b_advantages, b_old_mean, b_old_std = batch
                b_q = (b_old_mean, b_old_std)

                p = self.policy(b_obs)
                proj_p = self.projection(self.policy, p, b_q, self._global_steps)

                new_logpacs = self.policy.log_probability(proj_p, b_actions)

                # Calculate policy rewards
                surrogate_loss, policy_stats = self.surrogate_loss(b_advantages, new_logpacs, b_old_logpacs)

                # Calculate entropy bonus
                entropy_loss = -self.entropy_coeff * self.policy.entropy(proj_p).mean()

                # Trust region loss
                trust_region_loss = self.projection.get_trust_region_loss(self.policy, p, proj_p)

                # Total loss
                loss = surrogate_loss + entropy_loss + trust_region_loss

                # If we are sharing weights or train jointly, take the value step simultaneously
                if self.vf_coeff > 0 and not self.critic:
                    # if no vf model is present, the model is part of the policy, therefore has to be trained jointly
                    batch_vf = select_batch(indices, returns, dataset.values)
                    vs = self.policy.get_value(b_obs)
                    vf_loss = self.value_loss(vs, *batch_vf)  # b_returns, b_old_values)
                    loss += self.vf_coeff * vf_loss
                    loss_dict['critic_losses'] += loss.detach()

                self.optimizer_policy.zero_grad()
                loss.backward()
                if self.clip_grad_norm > 0:
                    ch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad_norm)
                self.optimizer_policy.step()

                loss_dict['policy_loss'] += surrogate_loss.detach()
                loss_dict['trust_region_loss'] += trust_region_loss.detach()
                loss_dict['entropy_loss'] += entropy_loss.detach()
                loss_dict['total_loss'] += loss.detach()

                if self.verbose >= 1:
                    stats_dict.update({k: stats_dict[k] + v.mean().item() for k, v in policy_stats.items()})

        steps = self.epochs * self.n_minibatches
        loss_dict = collections.OrderedDict(sorted({k: v / steps for k, v in loss_dict.items()}.items()))

        if self.verbose >= 1:
            stats_dict = collections.OrderedDict(sorted({k: v / steps for k, v in stats_dict.items()}.items()))

        if not self.policy.contextual_std and self.projection.proj_type not in ["ppo", "papi"]:
            # set policy with projection value without doing regression.
            # In non-contextual cases we have only one cov, so the projection is the same.
            self.policy.set_std(proj_p[1][0].detach())

        return loss_dict, stats_dict

    def update_critic(self, dataset: TrajectoryOnPolicy):
        """
        Take an optimizer step fitting the value function parameterized by a neural network
        Args:
            dataset: NameTuple with obs, returns, and values
        Returns:
            Loss of the value regression problem

        """

        obs, returns, old_values = dataset.obs, dataset.returns, dataset.values

        vf_losses = tensorize(0., self.cpu, self.dtype)
        if self.verbose >= 1:
            stats_dict = collections.defaultdict(lambda: tensorize(0., True, self.dtype))

        for _ in range(self.val_epochs):
            splits = generate_minibatches(obs.shape[0], self.n_minibatches)

            # Minibatch SGD
            for indices in splits:
                sel_returns, sel_old_values, sel_obs = select_batch(indices, returns, old_values, obs)

                # Set not needed values to None to avoid computational overhead
                # batch = TrajectoryOnPolicy(sel_obs, None, None, None, sel_returns, None, sel_old_values, None, None,
                #                            None)
                # critic_loss = self.value_loss(self.critic, batch)

                vs = self.critic(sel_obs)
                critic_loss = self.value_loss(vs, sel_returns, sel_old_values)

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.clip_grad_norm > 0:
                    ch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad_norm)
                self.optimizer_critic.step()

                vf_losses += critic_loss.detach()

                if self.verbose >= 1:
                    stats_dict['values'] += vs.detach().mean()

        steps = self.val_epochs * self.n_minibatches
        loss_dict = collections.OrderedDict(critic_loss=(vf_losses / steps))
        if self.verbose >= 1:
            loss_dict.update({k: v / steps for k, v in loss_dict.items()}.items())
            stats_dict = collections.OrderedDict(returns=returns.mean(), old_values=old_values.mean(), **stats_dict)

        return loss_dict, stats_dict

    def sample(self) -> TrajectoryOnPolicy:
        """
        Generate trajectory samples.
        Returns:
            NamedTuple with samples
        """
        with ch.no_grad():
            dataset = self.sampler.run(self.n_training_samples, self.policy, self.critic)
            (obs, actions, logpacs, rewards, values, dones, time_limit_dones, term_obs, old_means, old_stds) = dataset

            if self.max_entropy_coeff > 0:
                # add entropy to rewards in order to maximize trajectory of discounted reward + entropy
                # R = E[sum(y^t (r_t + a*H_t)]
                rewards += self.max_entropy_coeff * self.policy.entropy((old_means, old_stds)).detach()

            # Calculate advantages and returns
            advantages, returns = self.advantage_and_return(rewards, values, dones, time_limit_dones)

        # Unrolled sampling (T, n_envs, ...) -> (T*n_envs, ...) to train in one forward pass
        unrolled = map(flatten_batch,
                       (obs, actions, logpacs, rewards, returns, advantages, values[:-1], dones, time_limit_dones,
                        term_obs))
        q_unrolled = tuple(map(flatten_batch, (old_means, old_stds)))

        # trajectory = collections.namedtuple("Trajectory",
        #                                     "obs actions logpacs rewards returns "
        #                                     "advantages values dones time_limit_dones q_values")

        return TrajectoryOnPolicy(*unrolled, q_unrolled)

    def step(self):
        """
        Take a full training step, including sampling, policy and vf update.
        Returns:
            metrics, train/test reward

        """
        self._global_steps += 1

        loss_dict, stats_dict = {}, {}
        dataset = self.sample()

        if self.critic:
            # Train value network separately
            loss_dict_critic, stats_dict_critic = self.update_critic(dataset)
            loss_dict.update(loss_dict_critic)
            stats_dict.update(stats_dict_critic)

        # Policy optimization step or in case the network shares weights/is trained jointly also value update
        loss_dict_policy, stats_dict_policy = self.update_policy(dataset)
        loss_dict.update(loss_dict_policy)
        stats_dict.update(stats_dict_policy)

        # PAPI projection after the policy updates with PPO.
        if self.projection.proj_type == "papi":
            self.projection(self.policy, None, dataset.q, self._global_steps,
                            obs=dataset.obs, lr_schedule=self.lr_schedule_policy,
                            lr_schedule_vf=self.lr_schedule_critic)

        self.lr_schedule_step()

        logging_step = self._global_steps * self.n_training_samples * self.sampler.num_envs * self.sampler.sample_cost
        loss_dict.update(self.regression_step(dataset.obs, dataset.q, self.n_minibatches, logging_step))

        if self.log_interval != 0 and self._global_steps % self.log_interval == 0:
            self.store.log_table_and_tb('loss', loss_dict, step=logging_step)
            self.store['loss'].flush_row()

            if self.verbose >= 1:
                self.store.log_table_and_tb('stats', stats_dict, step=logging_step)
                self.store['stats'].flush_row()

            # TODO: DO not commit
            # dist = dict(zip(('mean', 'std'), dataset.q))
            # self.store.update_row('distribution', dist)
            # self.store['distribution'].flush_row()

        metrics_dict = self.log_metrics(dataset.obs, dataset.q, logging_step)
        loss_dict.update(metrics_dict)

        eval_out = self.evaluate_policy(logging_step, self.evaluate_deterministic, self.evaluate_stochastic)

        return loss_dict, eval_out

    def lr_schedule_step(self):
        if self.lr_schedule_policy:
            lr_dict = {}
            # Linear learning rate annealing
            # PAPI uses a different concept for lr decay that is implemented in its projection
            self.lr_schedule_policy.step() if not self.projection.proj_type == "papi" else None
            lr_dict.update({f"lr": self.lr_schedule_policy.get_last_lr()[0]})
            if self.lr_schedule_critic:
                self.lr_schedule_critic.step() if not self.projection.proj_type == "papi" else None
                lr_dict.update({f"lr_vf": self.lr_schedule_critic.get_last_lr()[0]})

            self.store.log_table_and_tb('lr', lr_dict, step=self._global_steps * self.n_training_samples)
            self.store['lr'].flush_row()

    def learn(self):
        """
        Train agent fully for train_steps
        Returns:
            exploration_dict and evaluation_dict

        """

        try:
            rewards = deque(maxlen=5)
            rewards_test = deque(maxlen=5)

            epoch = self._global_steps
            for epoch in range(self._global_steps, self.train_steps):
                metrics_dict, rewards_dict = self.step()

                if self.log_interval != 0 and self._global_steps % self.log_interval == 0 and self.verbose >= 2:
                    logger.info("-" * 80)
                    metrics = ", ".join((*map(lambda kv: f'{kv[0]}={get_numpy(kv[1]):.4f}', metrics_dict.items()),))
                    logger.info(f"iter {epoch:6d}: {metrics}")

                if self.save_interval > 0 and epoch % self.save_interval == 0:
                    self.save(epoch)

                rewards.append(rewards_dict['exploration'][0]['mean'])
                rewards_test.append(rewards_dict['evaluation'][0]['mean'])

            self.store["final_results"].append_row({
                'iteration': epoch,
                '5_rewards': np.array(rewards).mean(),
                '5_rewards_test': np.array(rewards).mean(),
            })

            # final evaluation and save of model
            if self.save_interval > 0:
                self.save(self.train_steps)

            logging_step = (
                    self._global_steps * self.n_training_samples * self.sampler.num_envs * self.sampler.sample_cost)
            eval_out = self.evaluate_policy(logging_step, self.evaluate_deterministic, self.evaluate_stochastic)

        except Exception as e:
            raise e
            self.save(self.train_steps)
            # Add one in case this step was logged already
            logging_step = self._global_steps * self.n_training_samples * self.sampler.num_envs + 1
            eval_out = self.evaluate_policy(logging_step, self.evaluate_deterministic, self.evaluate_stochastic)

        return eval_out

    def save(self, iteration):
        """
        Save current training config
        Args:
            iteration: current iteration

        Returns:

        """
        checkpoint_dict = {
            'iteration': iteration,
            'policy': self.policy.state_dict(),
            'optimizer_policy': self.optimizer_policy.state_dict(),
            'envs': self.sampler.envs,
            'envs_test': self.sampler.envs_test,
            # 'sampler': self.sampler
            # 'vec_normalize'
            # 'env_runner': None
        }

        if self.critic:
            checkpoint_dict.update({'critic': self.critic.state_dict(),
                                    'optimizer_critic': self.optimizer_critic.state_dict()
                                    })
        self.store['checkpoints'].append_row(checkpoint_dict)

    @staticmethod
    def agent_from_data(store, train_steps=None, checkpoint_iteration=-1):
        """
        Initializes an agent from serialized data (via cox)
        Args:
            store: the name of the store where everything is logged
            train_steps: new number of total training steps
            checkpoint_iteration: Which step to load from

        Returns:
            agent, agent_params
        """

        param_keys = list(store['metadata'].df.columns)

        def process_item(v):
            try:
                return v.item()
            except (ValueError, AttributeError):
                return v

        param_values = [process_item(store.load('metadata', v, "object", checkpoint_iteration)) for v in param_keys]
        agent_params = {k: v for k, v in zip(param_keys, param_values)}
        if train_steps is not None:
            agent_params['train_steps'] = train_steps
        agent = PolicyGradient.agent_from_params(agent_params, store)

        mapper = ch.device('cpu') if agent_params['cpu'] else ch.device('cuda')
        iteration = store.load('checkpoints', 'iteration', '', checkpoint_iteration)

        def load_state_dict(model, ckpt_name):
            state_dict = store.load('checkpoints', ckpt_name, "state_dict", checkpoint_iteration, map_location=mapper)
            model.load_state_dict(state_dict)

        load_state_dict(agent.policy, 'policy')
        load_state_dict(agent.optimizer_policy, 'optimizer_policy')
        if agent.lr_schedule_policy:
            agent.lr_schedule_policy.last_epoch = iteration

        if agent.critic:
            load_state_dict(agent.critic, 'critic')
            load_state_dict(agent.optimizer_critic, 'optimizer_critic')
            if agent.lr_schedule_policy:
                agent.lr_schedule_critic.last_epoch = iteration

        # agent.sampler = store.load('checkpoints', 'sampler', 'pickle', checkpoint_iteration)
        env = store.load('checkpoints', 'envs', 'pickle', checkpoint_iteration)
        env.set_venv(agent.sampler.envs.venv)
        agent.sampler.envs = env
        # agent.sampler.envs_test = store.load('checkpoints', 'envs_test', 'pickle', checkpoint_iteration)
        env_test = store.load('checkpoints', 'envs_test', 'pickle', checkpoint_iteration)
        env_test.set_venv(agent.sampler.envs_test.venv)
        agent.sampler.envs_test = env_test
        agent._global_steps = iteration + 1
        agent.store = store

        return agent, agent_params

    @staticmethod
    def agent_from_params(params, store=None):
        """
        Construct a run given a dict of HPs.
        Args:
            params: param dict
            store: Cox logging instance.

        Returns:
            agent

        """

        print(params)

        handle_timelimit = False

        use_cpu = params['cpu']
        device = ch.device("cpu" if use_cpu else "cuda")
        dtype = ch.float64 if params['dtype'] == "float64" else ch.float32
        seed = params['seed']

        # Only create env here for shapes, replanning needs to be added as this modifies the shapes
        kwargs = {}
        replanning_interval = params['environment'].get('replanning_interval', -1)
        if replanning_interval > 0:
            kwargs.update({'black_box_kwargs': {
                'replanning_schedule': lambda pos, vel, obs, action, t: t % replanning_interval == 0}})
        env = fancy_gym.make(params['environment']['env_id'], seed=seed, **kwargs)
        action_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]

        np.random.seed(seed)
        ch.manual_seed(seed)

        # vf network
        critic = None
        if not params['policy']['share_weights']:
            value_loss, critic = get_value_loss_and_critic(value_loss_type="value", dim=obs_dim, device=device,
                                                           dtype=dtype, **params["critic"])

        # policy network
        policy = get_policy_network(proj_type=params['projection']['proj_type'], squash=False, device=device,
                                    dtype=dtype, obs_dim=obs_dim, action_dim=action_dim, **params['policy'],
                                    vf_model=critic if params['algorithm']["critic_coeff"] != 0 else None)

        # environments
        sampler = TrajectorySampler(discount_factor=params['algorithm']['discount_factor'],
                                    handle_timelimit=handle_timelimit,
                                    scale_actions=False, cpu=use_cpu, dtype=dtype, seed=seed, **params['environment'])

        # projections
        projection = get_projection_layer(action_dim=action_dim, total_train_steps=params['training']['train_steps'],
                                          cpu=use_cpu, dtype=dtype, **params['projection'],
                                          **params['projection_regression'])

        # if params['cpu'] and not params['proj_type'].lower() == "kl":
        # Do not lower threads for kl, we need higher order multithreading for the numerical solver.
        ch.set_num_threads(1)

        p = PolicyGradient(
            sampler=sampler,
            policy=policy,
            # only pass the model if not trained jointly, otherwise, the vf is accessed through the policy.
            critic=critic if params['algorithm']["critic_coeff"] == 0. else None,
            projection=projection,

            **params['optimizer'],
            **params['training'],
            **params['algorithm'],
            **params['logging'],

            store=store,

            seed=seed,
            cpu=use_cpu,
            dtype=dtype
        )

        return p
