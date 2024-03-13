import collections
import copy
import logging
from typing import Union

import fancy_gym
import numpy as np
import torch as ch
import torch.nn as nn

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
from trust_region_projections.sampling.dataclass import TrajectoryOffPolicy, TrajectoryOffPolicyRaw, \
    TrajectoryOffPolicyLogpacs, TrajectoryOffPolicyLogpacsRaw
from trust_region_projections.sampling.trajectory_sampler import TrajectorySampler
from trust_region_projections.utils.custom_store import CustomStore
from trust_region_projections.utils.network_utils import get_lr_schedule, get_optimizer, polyak_update
from trust_region_projections.utils.torch_utils import get_numpy, tensorize


class TD3(AbstractAlgorithm):

    def __init__(
            self,
            sampler: TrajectorySampler,
            policy: AbstractGaussianPolicy,
            target_policy: AbstractGaussianPolicy,
            critic: BaseCritic,
            replay_buffer: AbstractReplayBuffer,

            value_loss: AbstractCriticLoss,

            optimizer_policy: str = "adam",
            optimizer_critic: str = "adam",
            lr_policy: float = 3e-4,
            lr_critic: float = 3e-4,

            projection: BaseProjectionLayer = None,

            train_steps: int = 990,
            batch_size: int = 256,
            updates_per_epoch: int = 1000,
            n_training_samples: int = 1000,
            initial_samples: int = 10000,

            lr_schedule: str = "",
            clip_grad_norm: Union[float, None] = 0.5,

            polyak_weight: float = 5e-3,
            policy_update_interval: int = 2,
            target_critic_update_interval: int = 1,

            target_policy_noise: float = 0.2,
            clip_target_noise: float = 0.5,

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

        # instead of of leveraging a exploration noise parameter directly here, we utilize the covariance already
        # present in the policy. Given we neither use it to compute the loss nor for sampling, it maintains fixed
        # as long as it is not contextual.
        assert not policy.contextual_std

        super().__init__(policy, sampler, projection, train_steps, clip_grad_norm, store, verbose,
                         evaluate_deterministic, evaluate_stochastic, log_interval=log_interval,
                         save_interval=save_interval, seed=seed, cpu=cpu, dtype=dtype)

        # training
        self.value_loss = value_loss
        self.batch_size = batch_size
        self.initial_samples = initial_samples
        self.updates_per_epoch = updates_per_epoch
        self.n_training_samples = n_training_samples

        # experience replay
        self.replay_buffer = replay_buffer

        ################################################################################################################

        # loss parameters
        self.policy_noise = target_policy_noise
        self.clip_noise = clip_target_noise
        self.polyak_weight = polyak_weight
        self.policy_update_interval = policy_update_interval
        self.target_critic_update_interval = target_critic_update_interval

        # networks
        self.critic = critic
        self.target_policy = target_policy
        self.old_policy = copy.deepcopy(self.policy)

        self.qf_criterion = nn.MSELoss()

        # optimizers
        self.optimizer_policy = get_optimizer(optimizer_policy, self.policy.parameters(), lr_policy)
        self.optimizer_critic = get_optimizer(optimizer_critic, self.critic.parameters(), lr_critic)

        self.lr_schedule_policy = get_lr_schedule(lr_schedule, self.optimizer_policy, self.train_steps)
        self.lr_schedule_critic = get_lr_schedule(lr_schedule, self.optimizer_critic, self.train_steps)

        self._epoch_steps = 0

        if self.store is not None:
            self.setup_stores()

        self._logger = logging.getLogger('td3')

    def setup_stores(self):
        # Logging setup
        super(TD3, self).setup_stores()

        if self.lr_schedule_policy:
            self.store.add_table('lr', {
                f"lr_policy": float,
                f"lr_critic": float,
            })

        self.store.add_table('loss', {
            **self.value_loss.loss_schema,
            # 'total_loss': float,
            'policy_loss': float,
            # 'trust_region_loss': float,
        })

        if self.verbose >= 1:
            self.store.add_table('stats', {
                **self.value_loss.stats_schema,
                # 'logpacs': float,
            })

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
        if reset_envs: self.replay_buffer.reset()

        with ch.no_grad():
            traj = self.sampler.run(rollout_steps, self.policy, reset_envs=reset_envs, is_on_policy=False,
                                    random_actions=random_actions, off_policy_logpacs=self.value_loss.requires_logpacs)

        return traj

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

        with ch.no_grad():
            # For retrace, we have one action for each observations in the full sequence (T+1),
            # however we only have T next observations
            next_action_shape = batch.next_obs.shape[:-1] + batch.actions.shape[-1:]
            # Select action according to policy and add clipped noise
            noise = (ch.randn(next_action_shape) * self.policy_noise).clamp(-self.clip_noise, self.clip_noise)
            p = self.target_policy(batch.next_obs)
            # This assumes our policy is squashing with tanh as SAC
            next_actions = (self.target_policy.squash(p[0]) + noise).clamp(-1, 1)

            old_next_logpacs = None
            if self.value_loss.requires_logpacs:
                old_next_logpacs = self.policy.log_probability(p, batch.actions[:, 1:])

        entropy = 0
        qf_loss, loss_dict, info_vals = self.value_loss(self.critic, batch, next_actions, old_next_logpacs, entropy)

        self.optimizer_critic.zero_grad()
        qf_loss.backward()
        if self.clip_grad_norm > 0:
            ch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad_norm)
        self.optimizer_critic.step()

        if self.total_iters % self.target_critic_update_interval == 0:
            self.critic.update_target_net(self.polyak_weight)

        return loss_dict, info_vals

    def update_policy(self, obs):
        """
        Compute policy update based on replay buffer samples
        Args:
            obs: batch observations

        Returns:
            dict with policy_loss
        """
        policy_loss = obs.new_tensor(0.)
        if self.total_iters % self.policy_update_interval == 0:
            # Update policy network
            a, _ = self.policy(obs)
            action = self.policy.squash(a)
            policy_loss = -self.critic.q1((obs, action)).mean()

            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

            # Polyak averaging for target policy
            polyak_update(self.policy, self.target_policy, self.polyak_weight)

        # multiply loss by update interval to get correct mean for logging,
        # otherwise, that value will be x interval too small
        return collections.OrderedDict(policy_loss=policy_loss.detach() * self.policy_update_interval), {}

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
            self._epoch_steps += 1

            #######################################################################################################
            # generate one new sample per iteration based on new policy
            self.replay_buffer.add_samples(self.sample(1))

            # Sample replay buffer
            batch = self.replay_buffer.random_batch(self.batch_size)

            loss_dict_critic, stats_dict_critic = self.update_critic(batch)
            loss_dict_policy, stats_dict_policy = self.update_policy(batch.obs)

            assert loss_dict_policy.keys().isdisjoint(loss_dict_critic.keys())
            loss_dict.update({k: loss_dict[k] + v.mean().item() for k, v in
                              loss_dict_policy.items() | loss_dict_critic.items()})

            if self.verbose >= 1:
                assert stats_dict_policy.keys().isdisjoint(stats_dict_critic.keys())
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

        # if self.projection.initial_entropy is None:
        #     q_values = self.old_policy(self.replay_buffer.random_batch(self.initial_samples).obs)
        #     self.projection.initial_entropy = self.policy.entropy(q_values).mean()

        # Add all samples in the beginning
        # rlkit uses it like this, original code and stable-baselines add them one by one for each updates_per_epoch
        # self.replay_buffer.add_samples(self.sample(self.n_training_samples ,reset_envs=True))

        loss_dict, stats_dict = self.epoch_step()

        logging_step = self.total_iters + self.initial_samples
        if self.log_interval != 0 and self._global_steps % self.log_interval == 0:
            self.store.log_table_and_tb('loss', loss_dict, step=logging_step)
            self.store['loss'].flush_row()

            if self.verbose >= 1:
                self.store.log_table_and_tb('stats', stats_dict, step=logging_step)
                self.store['stats'].flush_row()
                loss_dict.update(stats_dict)

        if self.lr_schedule_policy:
            self.lr_schedule_policy.step()
            self.lr_schedule_critic.step()

        ################################################################################################################
        # TODO Maybe use more/less samples here in the off-policy setting?
        batch = self.replay_buffer.random_batch(32 * self.batch_size)
        obs = batch.obs
        with ch.no_grad():
            q = self.old_policy(obs)

        metrics_dict = self.log_metrics(obs, q, logging_step)
        loss_dict.update(metrics_dict)

        eval_out = self.evaluate_policy(logging_step, self.evaluate_deterministic, self.evaluate_stochastic)

        # update old policy for projection
        self.old_policy.load_state_dict(self.policy.state_dict())

        return loss_dict, eval_out

    def learn(self):

        rewards = collections.deque(maxlen=5)
        rewards_test = collections.deque(maxlen=5)

        if self.initial_samples > 0:
            self.replay_buffer.add_samples(self.sample(self.initial_samples, random_actions=True, reset_envs=True))

        for epoch in range(self._global_steps, self.train_steps):
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

        logging_step = self.total_iters + self.initial_samples
        eval_out = self.evaluate_policy(logging_step, self.evaluate_deterministic, self.evaluate_stochastic)

        return eval_out

    @property
    def total_iters(self):
        return (self._global_steps - 1) * self.updates_per_epoch + self._epoch_steps

    @property
    def total_samples(self):
        return self.total_iters + self.initial_samples

    def save(self, iteration):
        pass

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
        agent = TD3.agent_from_params(agent_params, store)

        mapper = ch.device('cpu') if agent_params['cpu'] else ch.device('cuda')
        iteration = store.load('checkpoints', 'iteration', '', checkpoint_iteration)

        def load_state_dict(model, ckpt_name):
            state_dict = store.load('checkpoints', ckpt_name, "state_dict", checkpoint_iteration, map_location=mapper)
            model.load_state_dict(state_dict)

        load_state_dict(agent.policy, 'policy')
        load_state_dict(agent.critic, 'critic')

        load_state_dict(agent.optimizer_policy, 'optimizer_policy')
        load_state_dict(agent.optimizer_critic, 'optimizer_critic')

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
        # Although TD3 uses a deterministic policy, we set a covariance here.
        # This allows to encode the gaussian exploration noise, which is used during sampling, directly in the policy.
        # Therefore, we always make the covariance diagonal and non contextual, initialized with 0 scale.
        p_params = params['policy']
        policy = get_policy_network("diag", params['projection']['proj_type'], squash=do_squash, device=device,
                                    dtype=dtype, obs_dim=obs_dim, action_dim=action_dim, init=p_params['init'],
                                    hidden_sizes=p_params['hidden_sizes'], activation=p_params['activation'],
                                    contextual_std=False, init_std=params['algorithm'].pop('exploration_noise'),
                                    trainable_std=False, share_weights=False, minimal_std=0.0, scale=0.0)

        target_policy = copy.deepcopy(policy)

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

        agent = TD3(
            sampler=sampler,
            policy=policy,
            target_policy=target_policy,
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
