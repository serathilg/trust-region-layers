import collections
import copy
import logging
from typing import Union

import fancy_gym
import numpy as np
import torch as ch
from torch import nn as nn

from trust_region_projections.algorithms.abstract_algo import AbstractAlgorithm
from trust_region_projections.losses.loss_factory import get_value_loss_and_critic
from trust_region_projections.losses.value_loss import AbstractCriticLoss
from trust_region_projections.models.policy.abstract_gaussian_policy import AbstractGaussianPolicy
from trust_region_projections.models.policy.policy_factory import get_policy_network
from trust_region_projections.models.value.critic import BaseCritic
from trust_region_projections.projections.base_projection_layer import BaseProjectionLayer
from trust_region_projections.projections.projection_factory import get_projection_layer
from trust_region_projections.replay_buffer.abstract_replay_buffer import AbstractReplayBuffer
from trust_region_projections.replay_buffer.replay_buffer_factory import get_replay_buffer
from trust_region_projections.sampling.dataclass import TrajectoryOffPolicy, TrajectoryOffPolicyLogpacs, \
    TrajectoryOffPolicyLogpacsRaw, TrajectoryOffPolicyRaw
from trust_region_projections.sampling.trajectory_sampler import TrajectorySampler
from trust_region_projections.utils.custom_store import CustomStore
from trust_region_projections.utils.network_utils import get_lr_schedule, get_optimizer, polyak_update
from trust_region_projections.utils.torch_utils import get_numpy, tensorize, get_stats_dict


class VlearnQ(AbstractAlgorithm):

    def __init__(
            self,
            sampler: TrajectorySampler,
            policy: AbstractGaussianPolicy,
            critic: BaseCritic,
            replay_buffer: AbstractReplayBuffer,

            value_loss: AbstractCriticLoss,

            optimizer_policy: str = "adam",
            optimizer_critic: str = "adam",
            optimizer_alpha: str = "adam",
            lr_policy: float = 3e-4,
            lr_critic: float = 3e-4,
            lr_alpha: float = 3e-4,

            projection: BaseProjectionLayer = None,

            train_steps: int = 990,
            batch_size: int = 256,
            updates_per_epoch: int = 1000,
            n_training_samples: int = 1000,
            initial_samples: int = 10000,

            lr_schedule: str = "",
            clip_grad_norm: Union[float, None] = 40.0,

            target_update_interval: int = 1,
            polyak_weight: float = 5e-3,
            polyak_weight_policy_trl: float = 5e-3,
            polyak_weight_policy_log: float = 5e-3,
            log_ratio_clip: float = 1.0,
            trl_policy_update: str = "polyak",
            log_policy_update: str = "polyak",

            alpha: Union[str, float] = "auto",
            target_entropy: float = "auto",

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

        super().__init__(policy, sampler, projection, train_steps, clip_grad_norm, store, verbose,
                         evaluate_deterministic, evaluate_stochastic, log_interval=log_interval,
                         save_interval=save_interval, seed=seed, cpu=cpu, dtype=dtype)

        # training
        self.value_loss = value_loss
        self.batch_size = batch_size
        self.initial_samples = initial_samples
        self.updates_per_epoch = updates_per_epoch
        self.n_training_samples_per_epoch = n_training_samples
        self.sample_frequency = max(self.updates_per_epoch / self.n_training_samples_per_epoch, 1.)
        self.n_samples = max(self.n_training_samples_per_epoch / self.updates_per_epoch, 1.)
        assert self.sample_frequency.is_integer() and self.n_samples.is_integer(), \
            "updates_per_epoch and n_training_samples should allow for a division without remainders."

        # experience replay
        self.replay_buffer = replay_buffer

        ################################################################################################################

        # loss parameters
        self.polyak_weight = polyak_weight
        self.polyak_weight_policy_trl = polyak_weight_policy_trl
        self.polyak_weight_policy_log = polyak_weight_policy_log
        self.target_update_interval = target_update_interval
        self.trl_policy_update = trl_policy_update

        # networks
        self.critic = critic
        self.old_policy_trl = copy.deepcopy(self.policy)
        self.old_policy_log = copy.deepcopy(self.policy)

        self.qf_criterion = nn.MSELoss()

        # entropy tuning parameters
        self.auto_entropy_tuning = alpha == "auto"
        if self.auto_entropy_tuning:
            # heuristic value from Tuomas
            self.target_entropy = -np.prod(self.sampler.action_shape) if target_entropy == "auto" else target_entropy
            self.log_alpha = tensorize(0., cpu=cpu, dtype=dtype).requires_grad_(True)
            self.optimizer_alpha = get_optimizer(optimizer_alpha, [self.log_alpha], lr_alpha, eps=1e-7)
            self.lr_schedule_alpha = get_lr_schedule(lr_schedule, self.optimizer_alpha, self.train_steps)
        else:
            self.log_alpha = tensorize(alpha, cpu=cpu, dtype=dtype).log()

        # optimizers
        self.optimizer_policy = get_optimizer(optimizer_policy, self.policy.parameters(), lr_policy, eps=1e-7)
        self.optimizer_critic = get_optimizer(optimizer_critic, self.critic.parameters(), lr_critic, eps=1e-7)

        self.lr_schedule_policy = get_lr_schedule(lr_schedule, self.optimizer_policy, self.train_steps)
        self.lr_schedule_critic = get_lr_schedule(lr_schedule, self.optimizer_critic, self.train_steps)

        self._epoch_steps = 0
        self._total_samples = 0

        if self.store is not None:
            self.setup_stores()

        self._logger = logging.getLogger('sac')

    def setup_stores(self):
        # Logging setup
        super(VlearnQ, self).setup_stores()

        if self.lr_schedule_policy:
            self.store.add_table('lr', {
                f"lr_policy": float,
                f"lr_critic": float,
            })

        self.store.add_table('loss', {
            **self.value_loss.loss_schema,
            'total_loss': float,
            'policy_loss': float,
            'alpha_loss': float,
            'trust_region_loss': float,
        })

        if self.verbose >= 1:
            d = {
                **self.value_loss.stats_schema,
                'alpha': float,
                'logpacs': float,
            }
            self.store.add_table('stats', {f'{k}_{t}': v for k, v in d.items() for t in ['mean', 'std', 'max', 'min']})

    def sample(self, n_samples, reset_envs=False, random_actions=False) -> Union[
        TrajectoryOffPolicyRaw, TrajectoryOffPolicyLogpacsRaw]:
        """
        Generate trajectory samples.
        Args:
            n_samples: number of samples to generate
            reset_envs: reset the environment after reaching the max episode length.
                        Automatic TimeLimits are disabled for generating training data.
                        SAC explicitly requires that.

        Returns:
            NamedTuple with samples
        """

        rollout_steps = int(np.ceil(n_samples / self.sampler.num_envs))
        if reset_envs:
            self.replay_buffer.reset()

        with ch.no_grad():
            traj = self.sampler.run(rollout_steps, self.policy, reset_envs=reset_envs, is_on_policy=False,
                                    off_policy_logpacs=self.value_loss.requires_logpacs, random_actions=random_actions)

        return traj

    def get_samples_and_logpacs(self, obs: ch.Tensor, return_distributions: bool = False, n_samples: int = 1):
        """
        Generate samples with repa trick and compute log probs of Tanh Gaussian
        Args:
            obs: batched obs [batch_dim x obs_dim]
            return_distributions: return the current policy distribution as well as the projected one.

        Returns:
            squashed_actions, logpacs and optionally p, proj_p if `return_distributions` is True.

        """

        # Make sure policy accounts for squashing functions like tanh correctly!
        p = self.policy(obs)
        q = self.old_policy_trl(obs)
        proj_p = self.projection(self.policy, p, q, self._global_steps)
        # proj_p = p
        new_actions = self.policy.rsample(proj_p, n=n_samples)
        squashed_actions = self.policy.squash(new_actions)
        logpacs = self.policy.log_probability(proj_p, squashed_actions, pre_squash_x=new_actions)

        if return_distributions:
            return squashed_actions, logpacs, p, proj_p

        return squashed_actions, logpacs

    def update_critic(self, batch: Union[TrajectoryOffPolicy, TrajectoryOffPolicyLogpacs]):

        """
        Compute qf update based on replay buffer samples
        Args:
            batch: namedtuple with:
                    obs: batch observations
                    rewards: batch rewards
                    actions: batch actions
                    dones: batch terminals

        Returns: qf1_loss, qf2_loss, current_qf1, current_qf2, q_target (for logging)

        """

        next_actions, next_logpacs, _, proj_p = self.get_samples_and_logpacs(batch.next_obs, True)
        old_next_logpacs = None
        if self.value_loss.requires_logpacs:
            old_next_logpacs = self.policy.log_probability(proj_p, batch.actions[:, 1:])

        entropy = - self.log_alpha.exp() * next_logpacs
        qf_loss, loss_dict, info_vals = self.value_loss(self.critic, batch, next_actions, old_next_logpacs, entropy)

        self.optimizer_critic.zero_grad()
        qf_loss.backward()
        if self.clip_grad_norm > 0:
            ch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad_norm)
        self.optimizer_critic.step()

        if self.total_iters % self.target_update_interval == 0:
            self.critic.update_target_net(self.polyak_weight)

        return loss_dict, info_vals

    def _update_alpha(self, logpacs: ch.Tensor):
        """
        Update alpha parameter for automatic entropy tuning.
        Args:
            logpacs: log probabilities of current policy

        Returns:
            alpha_loss
        """
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (logpacs + self.target_entropy).detach()).mean()
            self.optimizer_alpha.zero_grad()
            alpha_loss.backward()
            self.optimizer_alpha.step()
        else:
            alpha_loss = logpacs.new_tensor(0.)

        return {"alpha_loss": alpha_loss.detach()}

    def update_policy_and_alpha(self, batch: Union[TrajectoryOffPolicy, TrajectoryOffPolicyLogpacs]):
        """
        Update Policy network and Alpha.
        Args:
            obs: batch observations

        Returns:
            loss_dict{total_loss, policy_loss, alpha_loss, trust_region_loss}, info_vals{alpha, next_logpacs}
        """
        ################################################################################################################

        with ch.no_grad():
            sampled_actions, sampled_logpacs = self.get_samples_and_logpacs(batch.obs, return_distributions=False,
                                                                            n_samples=20)

        p = self.policy(batch.obs)
        q = self.old_policy_trl(batch.obs)
        proj_p = self.projection(self.policy, p, q, self._global_steps)
        new_logpacs = self.policy.log_probability(proj_p, batch.actions)

        loss_dict = self._update_alpha(new_logpacs)

        with ch.no_grad():
            # ratio
            q_log = self.old_policy_log(batch.obs)
            old_logpacs = self.old_policy_log.log_probability(q_log, batch.actions).detach()

            # entropy
            alpha = self.log_alpha.exp()
            entropy = - alpha * sampled_logpacs

            obs = batch.obs
            if obs.dim() != sampled_actions.dim():
                # first dimension is sampling dimension of actions
                obs = obs[None].expand((sampled_actions.shape[0],) + obs.shape)

            q_values = self.critic((obs, sampled_actions))
            q_values = ch.atleast_2d(q_values)
            values = (q_values + entropy).mean(0)

            advantages = self.critic((batch.obs, batch.actions)) - values

        ratio = (new_logpacs - old_logpacs).exp()

        policy_loss = - (ratio * advantages.detach()).mean()

        # Calculate trust region loss
        trust_region_loss = self.projection.get_trust_region_loss(self.policy, p, proj_p)
        # trust_region_loss = ch.zeros(1)
        total_loss = policy_loss + trust_region_loss

        # Gradient step
        self.optimizer_policy.zero_grad()
        total_loss.backward()
        self.optimizer_policy.step()

        info_vals = collections.OrderedDict(
            alpha=alpha.detach(),
            logpacs=new_logpacs.detach()
        )

        loss_dict.update(collections.OrderedDict(
            total_loss=total_loss.detach(),
            policy_loss=policy_loss.detach(),
            # policy_loss_unscaled=policy_loss_unscaled.detach(),
            trust_region_loss=trust_region_loss.detach()
        ))

        # update old policy for projection
        if self.total_iters % self.target_update_interval:
            polyak_update(self.policy, self.old_policy_log, self.polyak_weight_policy_log)
            if self.trl_policy_update == "polyak":
                polyak_update(self.policy, self.old_policy_trl, self.polyak_weight_policy_trl)

        return loss_dict, info_vals

    def epoch_step(self):
        """
        Policy and qf optimization step.
        Returns:
            Loss dict and stats dict
        """
        self._epoch_steps = 0

        loss_dict = collections.defaultdict(lambda: tensorize(0., True, self.dtype))

        stats_vals = {}
        if self.verbose >= 1:
            stats_vals = collections.defaultdict(lambda: tensorize(0., True, self.dtype))

        for _ in range(self.updates_per_epoch):
            #######################################################################################################
            # generate one new sample per iteration based on new policy
            if self._epoch_steps % self.sample_frequency == 0:
                self._total_samples += self.n_samples
                self.replay_buffer.add_samples(self.sample(self.n_samples))

            # increment after sampling block to sample in first iteration
            self._epoch_steps += 1

            # ############################################################################################################

            batch = self.replay_buffer.random_batch(self.batch_size)

            loss_dict_critic, stats_dict_critic = self.update_critic(batch)
            loss_dict_policy, stats_dict_policy = self.update_policy_and_alpha(batch)

            assert loss_dict_policy.keys().isdisjoint(loss_dict_critic.keys())
            loss_dict.update({k: loss_dict[k] + v.mean().item() for k, v in
                              loss_dict_policy.items() | loss_dict_critic.items()})

            if self.verbose >= 1:
                assert stats_dict_policy.keys().isdisjoint(stats_dict_critic.keys())
                stats_dict_critic = get_stats_dict(stats_dict_critic)
                stats_dict_policy = get_stats_dict(stats_dict_policy)
                stats_vals.update({k: stats_vals[k] + v.mean().item() for k, v in
                                   stats_dict_policy.items() | stats_dict_critic.items()})

        ################################################################################################################

        # Logging after each epoch
        loss_dict.update({k: v / self.updates_per_epoch for k, v in loss_dict.items()})

        if self.verbose >= 1:
            stats_vals.update({k: v / self.updates_per_epoch for k, v in stats_vals.items()})

        return loss_dict, stats_vals

    def step(self):

        self._global_steps += 1

        if self.projection.initial_entropy is None:
            q = self.old_policy(self.replay_buffer.random_batch(self.replay_buffer.size).obs)
            self.projection.initial_entropy = self.policy.entropy(q).mean()

        # Add all samples in the beginning
        # rlkit uses it like this, original code and stable-baselines add them one by one for each updates_per_epoch
        # samples = self.sample(self.n_training_samples_per_epoch)
        # self.replay_buffer.add_samples(samples)  # , reset_envs=True))
        # print("Actions Buffer, std {} - mean  {}".format(*ch.std_mean(samples.actions)))

        loss_dict, stats_dict = self.epoch_step()

        if self.log_interval != 0 and self._global_steps % self.log_interval == 0:
            self.store.log_table_and_tb('loss', loss_dict, step=self._total_samples)
            self.store['loss'].flush_row()
            if self.verbose >= 1:
                self.store.log_table_and_tb('stats', stats_dict, step=self._total_samples)
                self.store['stats'].flush_row()

        self.lr_schedule_step()

        ################################################################################################################
        # TODO Maybe use more/less samples here in the off-policy setting?
        batch = self.replay_buffer.random_batch(1 * self.batch_size)
        obs = batch.obs
        with ch.no_grad():
            q = self.old_policy(obs)

        metrics_dict = self.log_metrics(obs, q, self._total_samples)

        loss_dict.update(metrics_dict)
        loss_dict.update(stats_dict)

        eval_out = self.evaluate_policy(self._total_samples, self.evaluate_deterministic, self.evaluate_stochastic)

        # # update old policy for projection
        self.old_policy.load_state_dict(self.policy.state_dict())

        return loss_dict, eval_out

    def lr_schedule_step(self):
        if self.lr_schedule_policy:
            self.lr_schedule_policy.step()
            self.lr_schedule_critic.step()
            self.lr_schedule_alpha.step()
            lr_dict = {f"lr_policy": self.lr_schedule_policy.get_last_lr()[0],
                       f"lr_critic": self.lr_schedule_critic.get_last_lr()[0],
                       f"lr_alpha": self.lr_schedule_alpha.get_last_lr()[0]
                       }

            self.store.log_table_and_tb('lr', lr_dict, step=self._global_steps * self.n_training_samples_per_epoch)
            self.store['lr'].flush_row()

    def learn(self):

        rewards = collections.deque(maxlen=5)
        rewards_test = collections.deque(maxlen=5)

        if self.initial_samples > 0:
            self._total_samples += self.initial_samples
            self.replay_buffer.add_samples(self.sample(self.initial_samples, random_actions=True))

        for epoch in range(self.train_steps):
            metrics_dict, rewards_dict = self.step()

            if self.verbose >= 2 and self.log_interval != 0 and self._global_steps % self.log_interval == 0:
                self._logger.info("-" * 80)
                metrics = ", ".join((*map(lambda kv: f'{kv[0]}={get_numpy(kv[1]):.4f}', metrics_dict.items()),))
                self._logger.info(f"iter {epoch:6d}: {metrics}")

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

        eval_out = self.evaluate_policy(self._total_samples, self.evaluate_deterministic, self.evaluate_stochastic)

        return eval_out

    def save(self, iteration):
        save_dict = {
            'iteration': iteration,
            'critic': self.critic.state_dict(),
            'policy': self.policy.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
            'optimizer_policy': self.optimizer_policy.state_dict(),
            'envs': self.sampler.envs,
            'envs_test': self.sampler.envs_test,
            # 'sampler': self.sampler
        }
        if self.auto_entropy_tuning:
            save_dict.update({
                'log_alpha': self.log_alpha,
                'optimizer_alpha': self.optimizer_alpha.state_dict(),
            })

        self.store['checkpoints'].append_row(save_dict)

    @property
    def total_iters(self):
        return (self._global_steps - 1) * self.updates_per_epoch + self._epoch_steps

    @staticmethod
    def agent_from_data(store, train_steps=None, checkpoint_iteration=-1):
        """
        Initializes an agent from serialized data (via cox)
        Args:
            store: the name of the store where everything is logged
            train_steps: Which step to load from
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
        agent = VlearnQ.agent_from_params(agent_params, store)

        mapper = ch.device('cpu') if agent_params['cpu'] else ch.device('cuda')
        iteration = store.load('checkpoints', 'iteration', '', checkpoint_iteration)

        def load_state_dict(model, ckpt_name):
            state_dict = store.load('checkpoints', ckpt_name, "state_dict", checkpoint_iteration, map_location=mapper)
            model.load_state_dict(state_dict)

        load_state_dict(agent.policy, 'policy')
        load_state_dict(agent.critic, 'critic')

        load_state_dict(agent.optimizer_policy, 'optimizer_policy')
        load_state_dict(agent.optimizer_critic, 'optimizer_critic')

        if agent.auto_entropy_tuning:
            agent.log_alpha = store.load('checkpoints', 'log_alpha', "state_dict", checkpoint_iteration,
                                         map_location=mapper)
            load_state_dict(agent.optimizer_alpha, 'optimizer_alpha')

        if agent.lr_schedule_policy:
            agent.lr_schedule_policy.last_epoch = iteration
            agent.lr_schedule_critic.last_epoch = iteration

        # agent.sampler = store.load('checkpoints', 'sampler', 'pickle', checkpoint_iteration)
        env = store.load('checkpoints', 'envs', 'pickle', checkpoint_iteration)
        env.set_venv(agent.sampler.envs.venv)
        agent.sampler.envs = env
        # agent.sampler.envs_test = store.load('checkpoints', 'envs_test', 'pickle', checkpoint_iteration)
        env = store.load('checkpoints', 'envs_test', 'pickle', checkpoint_iteration)
        env.set_venv(agent.sampler.envs_test.venv)
        agent.sampler.envs_test = env
        agent._global_steps = iteration + 1
        agent.store = store

        return agent, agent_params

    @staticmethod
    def agent_from_params(params, store=None):
        """
        Construct a trainer object given a dictionary of hyperparameters.
        Trainer is in charge of sampling sampling, updating policy network,
        updating value network, and logging.
        Inputs:
        - params, dictionary of required hyperparameters
        - store, a cox.Store object if logging is enabled
        Outputs:
        - A Trainer object for training a PPO/TRPO agent
        """

        print(params)

        do_squash = True
        handle_timelimit = True
        disable_timelimit = False
        # use_time_feature_wrapper = False

        use_cpu = params['cpu']
        device = ch.device("cpu" if use_cpu else "cuda")
        dtype = ch.float64 if params['dtype'] == "float64" else ch.float32
        seed = params['seed']

        env = fancy_gym.make(params['environment']['env_id'], seed=seed)
        action_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]

        np.random.seed(seed)
        ch.manual_seed(seed)

        # critic network and value loss
        value_loss, critic = get_value_loss_and_critic(dim=obs_dim + action_dim, device=device, dtype=dtype,
                                                       **params['value_loss'], **params["critic"])

        # policy network
        policy = get_policy_network(proj_type=params['projection']['proj_type'], squash=do_squash, device=device,
                                    dtype=dtype, obs_dim=obs_dim, action_dim=action_dim, **params['policy'])

        # environments
        sampler = TrajectorySampler(discount_factor=params['replay_buffer']['discount_factor'],
                                    scale_actions=do_squash, handle_timelimit=handle_timelimit,
                                    disable_timelimit=disable_timelimit,
                                    cpu=use_cpu, dtype=dtype, seed=seed, **params['environment'])

        # projections
        projection = get_projection_layer(action_dim=action_dim, total_train_steps=params['training']['train_steps'],
                                          cpu=use_cpu, dtype=dtype, **params['projection'])

        # replay buffer
        replay_buffer = get_replay_buffer(observation_dim=obs_dim, action_dim=action_dim, dtype=dtype, device=device,
                                          handle_timelimit=handle_timelimit, **params['replay_buffer'])

        # if params['cpu'] and not params['proj_type'].lower() == "kl":
        # Do not lower threads for kl, we need higher order multithreading for the numerical solver.
        ch.set_num_threads(1)

        agent = VlearnQ(
            sampler=sampler,
            policy=policy,
            critic=critic,
            replay_buffer=replay_buffer,
            value_loss=value_loss,
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

        return agent
