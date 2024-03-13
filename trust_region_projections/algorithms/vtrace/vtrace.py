import collections
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
from trust_region_projections.sampling.dataclass import TrajectoryOffPolicyLogpacs, \
    TrajectoryOffPolicyLogpacsRaw
from trust_region_projections.sampling.trajectory_sampler import TrajectorySampler
from trust_region_projections.utils.custom_store import CustomStore
from trust_region_projections.utils.network_utils import get_lr_schedule, get_optimizer
from trust_region_projections.utils.torch_utils import get_numpy, tensorize


class Vtrace(AbstractAlgorithm):

    def __init__(
            self,
            sampler: TrajectorySampler,
            policy: AbstractGaussianPolicy,
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
            sample_frequency: int = 1000,
            n_training_samples: int = 1000,
            initial_samples: int = 10000,

            lr_schedule: str = "",
            clip_grad_norm: Union[float, None] = 40.0,

            target_update_interval: int = 1,
            polyak_weight: float = 5e-3,
            lambda_weight: float = 1.0,
            clip_rho_threshold: float = 1.0,
            clip_rho_pg_threshold: float = 1.0,
            clip_c_threshold: float = 1.0,
            entropy_coeff: float = 0.00025,
            reward_clipping: str = "soft_asymmetric",

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
        self.sample_frequency = sample_frequency
        self.n_training_samples = n_training_samples
        # self.n_training_samples_per_epoch = n_training_samples
        # self.sample_frequency = max(self.updates_per_epoch / self.n_training_samples_per_epoch, 1.)
        # self.n_samples = max(self.n_training_samples_per_epoch / self.updates_per_epoch, 1.)
        # assert self.sample_frequency.is_integer() and self.n_samples.is_integer(), \
        #     "updates_per_epoch and n_training_samples should allow for a division without remainders."

        # experience replay
        self.replay_buffer = replay_buffer

        ################################################################################################################

        # loss parameters
        self.polyak_weight = polyak_weight
        self.target_update_interval = target_update_interval
        self.lambda_weight = tensorize(lambda_weight, self.cpu, self.dtype)
        self.clip_rho_threshold = tensorize(clip_rho_threshold, self.cpu, self.dtype)
        self.clip_rho_pg_threshold = tensorize(clip_rho_pg_threshold, self.cpu, self.dtype)
        self.clip_c_threshold = tensorize(clip_c_threshold, self.cpu, self.dtype)
        self.entropy_coeff = entropy_coeff
        self.reward_clipping = reward_clipping

        # networks
        self.critic = critic

        self.critic_criterion = nn.MSELoss()

        # optimizers
        self.optimizer_policy = get_optimizer(optimizer_policy, self.policy.parameters(), lr_policy)
        self.optimizer_critic = get_optimizer(optimizer_critic, self.critic.parameters(), lr_critic)

        self.lr_schedule_policy = get_lr_schedule(lr_schedule, self.optimizer_policy, self.train_steps)
        self.lr_schedule_critic = get_lr_schedule(lr_schedule, self.optimizer_critic, self.train_steps)

        self._epoch_steps = 0
        self._total_samples = 0

        if self.store is not None:
            self.setup_stores()

        self._logger = logging.getLogger('vtrace')

    def setup_stores(self):
        # Logging setup
        super(Vtrace, self).setup_stores()

        if self.lr_schedule_policy:
            self.store.add_table('lr', {
                f"lr_policy": float,
                f"lr_critic": float,
            })

        self.store.add_table('loss', {
            # **self.value_loss.loss_schema,
            **{"critic_loss": float},
            'total_loss': float,
            'policy_loss': float,
            'entropy_loss': float,
        })

        if self.verbose >= 1:
            self.store.add_table('stats', {
                "current_v": float,
                "bootstrap_v": float,
                "target_v_values": float,
                "rho": float,
                "c": float,
                "ratio": float,
                # **self.value_loss.stats_schema,
            })

    def sample(self, n_samples, reset_envs=False, random_actions=False) -> Union[TrajectoryOffPolicyLogpacsRaw]:
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
                                    random_actions=random_actions, off_policy_logpacs=True)
        traj = TrajectoryOffPolicyLogpacsRaw(*traj)
        return traj

    def compute_vtrace_targets(self, batch: TrajectoryOffPolicyLogpacs):

        obs, actions, rewards, next_obs, terminals, old_logpacs = batch
        # actions also contain the last action for the next state (only required for Q-estimates not V)
        actions = actions[:, :-1]  # [batch, T, action_dim]

        # Taken from https://github.com/deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/experiment.py#L365-L370
        if self.reward_clipping == 'abs_one':
            rewards = rewards.clamp(-1, 1)
        elif self.reward_clipping == 'soft_asymmetric':
            squeezed = ch.tanh(rewards / 5.0)
            # Negative rewards are given less weight than positive rewards.
            rewards = ch.where(rewards < 0, .3 * squeezed, squeezed) * 5.

        with ch.no_grad():
            p = self.policy(obs, train=False)
            logpacs = self.policy.log_probability(p, actions)
            ratio = (logpacs - old_logpacs).exp()
            rhos = ch.min(self.clip_rho_threshold, ratio)
            cs = self.lambda_weight * ch.min(self.clip_c_threshold, ratio)

            if isinstance(self.critic, BaseCritic):
                values, next_values = self.critic(obs), self.critic(next_obs)
            else:
                values, next_values = self.critic.target(obs), self.critic.target(next_obs)
            td_target = rewards + terminals * next_values
            delta = rhos * (td_target - values)

            vs_minus_v_xs = ch.zeros(*obs.shape[:-2], obs.shape[-2] + 1, dtype=self.dtype)

            for i in range(obs.shape[-2] - 1, -1, -1):
                vs_minus_v_xs[..., i] = terminals[:, i] * cs[:, i] * vs_minus_v_xs[:, i + 1] + delta[:, i]

            vs = vs_minus_v_xs[:, :-1] + values
            next_vs = vs_minus_v_xs[:, 1:] + next_values
            advantage = rewards + terminals * next_vs - values
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            pg_ratio = ch.min(self.clip_rho_pg_threshold, ratio)
            pg_advantages = pg_ratio * advantage

            # print(vs.mean(), advantage.mean())

        stats_dict = collections.OrderedDict(
            bootstrap_v=vs.mean(),
            target_v_values=next_vs.mean(),
            rho=rhos.mean(),
            c=cs.mean(),
            ratio=pg_ratio.mean(),
        )

        return vs, pg_advantages, stats_dict

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
            # #######################################################################################################
            # # generate one new sample per iteration based on new policy
            # if self._epoch_steps % self.sample_frequency == 0:
            #     self._total_samples += self.n_training_samples
            #     self.replay_buffer.add_samples(self.sample(self.n_training_samples))
            #
            # # increment after sampling block to sample in first iteration
            self._epoch_steps += 1

            batch = self.replay_buffer.random_batch(self.batch_size)

            vs, advantage, stats_dict = self.compute_vtrace_targets(batch)
            obs = batch.obs
            actions = batch.actions[:, :-1]  # [batch, T, action_dim]

            current_v = self.critic(obs)
            critic_loss = self.critic_criterion(current_v, vs)
            p = self.policy(obs)
            logpacs = self.policy.log_probability(p, actions)
            policy_loss = - logpacs * advantage
            entropy_loss = - self.entropy_coeff * logpacs

            total_loss = policy_loss + entropy_loss

            self.optimizer_policy.zero_grad()
            total_loss.mean().backward()
            if self.clip_grad_norm > 0:
                ch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad_norm)
            self.optimizer_policy.step()

            self.optimizer_critic.zero_grad()
            critic_loss.mean().backward()
            if self.clip_grad_norm > 0:
                ch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad_norm)
            self.optimizer_critic.step()

            if self.total_iters % self.target_update_interval == 0:
                self.critic.update_target_net(self.polyak_weight)

            stats_dict["current_v"] = current_v

            loss_dict['total_loss'] += total_loss.mean().detach()
            loss_dict['policy_loss'] += policy_loss.mean().detach()
            loss_dict['entropy_loss'] += entropy_loss.mean().detach()
            loss_dict['critic_loss'] += critic_loss.mean().detach()

            #######################################################################################################

            if self.verbose >= 1:
                stats_vals.update({k: stats_vals[k] + v.mean().item() for k, v in stats_dict.items()})

        ################################################################################################################

        # Logging after each epoch
        loss_dict.update({k: v / self.updates_per_epoch for k, v in loss_dict.items()})

        if self.verbose >= 1:
            stats_vals.update({k: v / self.updates_per_epoch for k, v in stats_vals.items()})

        return loss_dict, stats_vals

    def step(self):

        self._global_steps += 1

        # Add all samples in the beginning
        # rlkit uses it like this, original code and stable-baselines add them one by one for each updates_per_epoch
        self._total_samples += self.n_training_samples
        self.replay_buffer.add_samples(self.sample(self.n_training_samples, reset_envs=True))

        if self.projection.initial_entropy is None:
            # obs = self.replay_buffer.random_batch(self.replay_buffer.size).obs
            # q = self.old_policy(obs)
            self.projection.initial_entropy = tensorize(float("inf"), self.cpu, self.dtype)[None]

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
        batch = self.replay_buffer.random_batch(8 * self.batch_size)
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
            lr_dict = {f"lr_policy": self.lr_schedule_policy.get_last_lr()[0],
                       f"lr_critic": self.lr_schedule_critic.get_last_lr()[0],
                       }

            self.store.log_table_and_tb('lr', lr_dict, step=self._global_steps * self.n_training_samples_per_epoch)
            self.store['lr'].flush_row()

    def learn(self):

        rewards = collections.deque(maxlen=5)
        rewards_test = collections.deque(maxlen=5)

        if self.initial_samples > 0:
            self._total_samples += self.initial_samples
            self.replay_buffer.add_samples(self.sample(self.initial_samples, random_actions=False))

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
        }

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
        agent = Vtrace.agent_from_params(agent_params, store)

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

        do_squash = False
        handle_timelimit = True
        # use_time_feature_wrapper = False

        use_cpu = params['cpu']
        device = ch.device("cpu" if use_cpu else "cuda")
        dtype = ch.float64 if params['dtype'] == "float64" else ch.float32
        seed = params['seed']

        env = fancy_gym.make(params['environment']['env_id'], seed=seed)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        np.random.seed(seed)
        ch.manual_seed(seed)

        # critic network and value loss
        value_loss, critic = get_value_loss_and_critic(dim=obs_dim, device=device, dtype=dtype,
                                                       **params['value_loss'], **params["critic"])

        # policy network
        policy = get_policy_network(proj_type=params['projection']['proj_type'], squash=do_squash, device=device,
                                    dtype=dtype, obs_dim=obs_dim, action_dim=action_dim, **params['policy'])

        # environments
        sampler = TrajectorySampler(discount_factor=params['replay_buffer']['discount_factor'],
                                    scale_actions=do_squash, handle_timelimit=handle_timelimit,
                                    cpu=use_cpu, dtype=dtype, seed=seed, **params['environment'])

        # projections
        projection = get_projection_layer(action_dim=action_dim, total_train_steps=params['training']['train_steps'],
                                          cpu=use_cpu, dtype=dtype, **params['projection'],
                                          **params['projection_regression'])

        # replay buffer
        replay_buffer = get_replay_buffer(observation_dim=obs_dim, action_dim=action_dim, dtype=dtype, device=device,
                                          handle_timelimit=handle_timelimit, **params['replay_buffer'])

        # if params['cpu'] and not params['proj_type'].lower() == "kl":
        # Do not lower threads for kl, we need higher order multithreading for the numerical solver.
        ch.set_num_threads(1)

        p = Vtrace(
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

        return p
