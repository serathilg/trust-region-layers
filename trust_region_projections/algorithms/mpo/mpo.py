import collections
import copy

import gym
import logging
import numpy as np
import torch as ch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

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
from trust_region_projections.utils.projection_utils import gaussian_kl
from trust_region_projections.utils.torch_utils import flatten_batch, get_numpy, tensorize


class MPO(AbstractAlgorithm):
    """
    This implementation is derived from the Deepmind's MPO implementation
    https://github.com/deepmind/acme/
    Copyright 2018 DeepMind Technologies Limited. All rights reserved, licensed as Apache-2.0
    """

    def __init__(self,
                 sampler: TrajectorySampler,
                 policy: AbstractGaussianPolicy,
                 target_policy: AbstractGaussianPolicy,
                 critic: BaseCritic,
                 replay_buffer: AbstractReplayBuffer,

                 value_loss: AbstractCriticLoss,

                 optimizer_policy: str = "adam",
                 optimizer_critic: str = "adam",
                 optimizer_dual: str = "adam",
                 lr_policy: float = 1e-4,
                 lr_critic: float = 1e-4,
                 lr_dual: float = 1e-2,

                 projection: BaseProjectionLayer = None,

                 train_steps: int = 990,
                 batch_size: int = 256,
                 updates_per_epoch: int = 1000,
                 n_training_samples: int = 1000,
                 initial_samples: int = 10000,

                 lr_schedule: str = "",
                 clip_grad_norm: Union[float, None] = 40.0,

                 target_policy_update_interval: int = 100,
                 target_critic_update_interval: int = 100,
                 polyak_weight: float = 5e-3,

                 n_action_samples: int = 64,
                 dual_constraint: float = 0.1,
                 mean_constraint: float = 0.001,
                 var_constraint: float = 1e-6,
                 log_eta: float = 1.0,
                 log_alpha_mu: float = 1.0,
                 log_alpha_sigma: float = 10.0,

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
        self.n_training_samples = n_training_samples

        # experience replay
        self.replay_buffer = replay_buffer

        ################################################################################################################

        # initialize trainable Lagrange parameters
        self.log_eta = tensorize(log_eta, self.cpu, self.dtype).requires_grad_(True)
        self.log_alpha_mu = tensorize(log_alpha_mu, self.cpu, self.dtype).requires_grad_(True)
        self.log_alpha_sigma = tensorize(log_alpha_sigma, self.cpu, self.dtype).requires_grad_(True)

        self.eps = dual_constraint
        self.eps_mu = mean_constraint
        self.eps_sigma = var_constraint

        ################################################################################################################

        self.polyak_weight = polyak_weight
        self.target_policy_update_interval = target_policy_update_interval
        self.target_qf_update_interval = target_critic_update_interval

        self.n_action_samples = n_action_samples

        # networks
        self.critic = critic
        self.target_policy = target_policy
        self.old_policy = copy.deepcopy(self.policy)

        self.qf_criterion = nn.MSELoss()

        # optimizers
        self.optimizer_policy = get_optimizer(optimizer_policy, self.policy.parameters(), lr_policy)
        self.optimizer_critic = get_optimizer(optimizer_critic, self.critic.parameters(), lr_critic)
        dual_vars = [self.log_eta, self.log_alpha_mu, self.log_alpha_sigma]
        self.optimizer_dual = get_optimizer(optimizer_dual, dual_vars, lr_dual)

        self.lr_schedule_policy = get_lr_schedule(lr_schedule, self.optimizer_policy, self.train_steps)
        self.lr_schedule_critic = get_lr_schedule(lr_schedule, self.optimizer_critic, self.train_steps)
        self.lr_schedule_dual = get_lr_schedule(lr_schedule, self.optimizer_dual, self.train_steps)

        self._epoch_steps = 0

        if self.store is not None:
            self.setup_stores()

        self._logger = logging.getLogger("mpo")

    def setup_stores(self):
        # Logging setup
        super(MPO, self).setup_stores()

        if self.lr_schedule_policy:
            self.store.add_table('lr', {
                f"lr_policy": float,
                f"lr_critic": float,
                f"lr_dual": float,
            })

        self.store.add_table('loss', {
            **self.value_loss.loss_schema,
            'total_loss': float,
            'policy_loss': float,
            'kl_penalty_loss': float,
            'dual_loss': float,
        })

        if self.verbose >= 1:
            self.store.add_table('stats', {
                **self.value_loss.stats_schema,
                'kl_q_rel': float,
                'alpha_mean': float,
                'alpha_sigma': float,
                "eta": float,
                # "actions": float,
                "logpacs": float,
            })

    def sample(self, n_samples, reset_envs=False) -> Union[TrajectoryOffPolicyRaw, TrajectoryOffPolicyLogpacsRaw]:
        """
        Generate trajectory samples.
        Args:
            n_samples: number of samples to generate
            reset_envs: reset the environment after reaching the max episode length.
                        Automatic TimeLimits are disabled for generating training data.
        Returns:
            NamedTuple with samples
        """

        rollout_steps = int(np.ceil(n_samples / self.sampler.num_envs))
        if reset_envs: self.replay_buffer.reset()

        with ch.no_grad():
            traj = self.sampler.run(rollout_steps, self.policy, reset_envs=reset_envs, is_on_policy=False,
                                    off_policy_logpacs=self.value_loss.requires_logpacs)

        return traj

    def get_action_samples(self, next_obs: ch.Tensor, return_distributions: bool = False):

        """
        Generate samples with repa trick and compute log probs of Tanh Gaussian
        Args:
            next_obs: batched obs [batch_dim x obs_dim]
            return_distributions: return the current policy distribution as well as the target policy one.

        Returns:
            actions, squashed_actions and optionally p, p_target if `return_distributions` is True.

        """
        # sample M additional action for each state
        p = self.policy(next_obs)
        with ch.no_grad():
            p_target = self.target_policy(next_obs)

        sampled_actions = self.target_policy.sample(p_target, self.n_action_samples)  # [N, B, ...]
        sampled_actions_squashed = self.target_policy.squash(sampled_actions)

        if return_distributions:
            return sampled_actions, sampled_actions_squashed, p, p_target

        return sampled_actions, sampled_actions_squashed

    def update_critic(self, batch: Union[TrajectoryOffPolicy, TrajectoryOffPolicyLogpacs], actions_tp1: ch.Tensor,
                      p: Tuple[ch.Tensor, ch.Tensor]):

        """
        Compute qf update based on replay buffer samples
        Args:
            batch: namedtuple with:
                    obs: batch observations
                    rewards: batch rewards
                    actions: batch actions
                    dones: batch terminals
            actions_tp1: batch actions for next state s_t+1
            p: current policy Gaussian distribution for next observations

        Returns: qf1_loss, qf2_loss, current_qf1, current_qf2, q_target (for logging)

        """

        old_next_logpacs = None
        if self.value_loss.requires_logpacs:
            old_next_logpacs = self.policy.log_probability(p, batch.actions[:, 1:])

        qf_loss, loss_dict, info_vals = self.value_loss(self.critic, batch, actions_tp1, old_next_logpacs, 0)

        self.optimizer_critic.zero_grad()
        qf_loss.backward()
        if self.clip_grad_norm > 0:
            ch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad_norm)
        self.optimizer_critic.step()

        if self.total_iters % self.target_qf_update_interval:
            self.critic.update_target_net(self.polyak_weight)

        # return target_q_values separately as they are need for the policy loss computation.
        return loss_dict, info_vals, info_vals.get('target_q')

    @staticmethod
    def weights_and_eta_loss(eta: ch.Tensor, q_values: ch.Tensor, epsilon: float):
        """
        Computes normalized importance weights for the policy optimization.

        Args:
          eta: Scalar used to temper the Q-values before computing normalized
            importance weights from them. This is really the Lagrange dual variable
            in the constrained optimization problem, the solution of which is the
            non-parametric policy targeted by the policy loss.
          q_values: Q-values associated with the actions sampled from the target
            policy; expected shape [N, B].
          epsilon: Desired constraint on the KL between the target and non-parametric
            policies.

        Returns:
          Normalized importance weights, used for policy optimization.
          Temperature loss, used to adapt the temperature.
        """
        # Temper the given Q-values using the current temperature.
        tempered_q_values = q_values.detach() / eta

        # Compute the normalized importance weights used to compute expectations with
        # respect to the non-parametric policy.
        normalized_weights = F.softmax(tempered_q_values, dim=0).detach()

        # Compute the eta loss (dual of the E-step optimization problem).
        # max_q = ch.max(tempered_q_values, 0)
        # max is substracted by the torch method automatically
        q_logsumexp = ch.logsumexp(tempered_q_values, dim=0)
        log_num_actions = q_values.new_tensor(q_values.shape[0]).log()
        loss_eta = eta * (epsilon + q_logsumexp.mean() - log_num_actions)

        return normalized_weights, loss_eta

    @staticmethod
    def kl_penalty_and_dual_loss(kl: ch.Tensor, alpha: ch.Tensor, epsilon: float) -> Tuple[
        ch.Tensor, ch.Tensor]:
        """
        Computes the KL cost to be added to the Lagrangian and its dual loss.
        The KL cost is simply the alpha-weighted KL divergence and it is added as a
        regularizer to the policy loss. The dual variable alpha itself has a loss that
        can be minimized to adapt the strength of the regularizer to keep the KL
        between consecutive updates at the desired target value of epsilon.

        Args:
          kl: KL divergence between the target and online policies.
          alpha: Lagrange multipliers (dual variables) for the KL constraints.
          epsilon: KL bound
        Returns:
          loss_kl: alpha-weighted KL regularization to be added to the policy loss.
          loss_alpha: The Lagrange dual loss minimized to adapt alpha.
        """

        mean_kl = kl.mean(0)

        # Compute the regularization.
        loss_kl = (alpha.detach() * mean_kl).sum()

        # Compute the dual loss.
        loss_alpha = (alpha * (epsilon - mean_kl.detach())).sum()

        return loss_kl, loss_alpha

    def cross_entropy_loss(self, p: Tuple[ch.Tensor, ch.Tensor], next_actions_squashed: ch.Tensor,
                           next_actions: ch.Tensor, normalized_weights: ch.Tensor) -> ch.Tensor:
        """Compute cross-entropy online and the reweighted target policy.

        Args:
          p: Gaussian distribution from current policy
          next_actions_squashed: samples used in the Monte Carlo integration in the policy
            loss. Expected shape is [N, B, ...], where N is the number of sampled
            actions and B is the number of sampled states a.k.a. batchsize.
          next_actions: actions but without squashing applied.
          normalized_weights: target policy multiplied by the exponentiated Q values
            and normalized; expected shape is [N, B].

        Returns:
          loss_policy_gradient: the cross-entropy loss that, when differentiated,
            produces the policy gradient.
        """

        # Compute the M-step loss.
        log_prob = self.target_policy.log_probability(p, next_actions_squashed, pre_squash_x=next_actions)

        # Compute the weighted average log-prob using the normalized weights.
        loss_policy_gradient = - (log_prob * normalized_weights).sum(0)

        # Return the mean loss over the batch of states.
        return loss_policy_gradient.mean(0)

    def update_policy(self, p_target, p, next_actions_squashed, next_actions, target_q_values):
        """
        Compute policy update based on replay buffer samples with the end-to-end method from deepmind.
        Args:
          p_target: Gaussian distribution from target policy
          p: Gaussian distribution from current policy
          next_actions_squashed: samples used in the Monte Carlo integration in the policy
            loss. Expected shape is [N, B, ...], where N is the number of sampled
            actions and B is the number of sampled states a.k.a. batchsize.
          next_actions: actions but without squashing applied.
          target_q_values: Q values from target; expected shape is [N, B].

        Returns:
            dict with policy_loss
        """

        if self.value_loss.requires_logpacs:
            # unroll second and third dimension [B, T] for policy updates, first dim is sampling dim
            flat = map(lambda x: x.flatten(1, 2), (next_actions, next_actions_squashed, target_q_values))
            next_actions, next_actions_squashed, target_q_values = flat
            # unroll first and second dimension [B, T] for policy updates
            p, p_target = [tuple(map(flatten_batch, e)) for e in (p, p_target)]

        # Project dual variables to ensure they stay positive.
        min_log_eta = next_actions.new_tensor(-18.0)
        min_log_alpha = next_actions.new_tensor(-18.0)
        self.log_eta.data = ch.max(min_log_eta, self.log_eta)
        self.log_alpha_mu.data = ch.max(min_log_alpha, self.log_alpha_mu)
        self.log_alpha_sigma.data = ch.max(min_log_alpha, self.log_alpha_sigma)

        eta = F.softplus(self.log_eta) + 1e-8
        alpha_mean = F.softplus(self.log_alpha_mu) + 1e-8
        alpha_sigma = F.softplus(self.log_alpha_sigma) + 1e-8

        normalized_weights, loss_eta = self.weights_and_eta_loss(eta, target_q_values, self.eps)

        # Compute integrand.
        integrand = (self.n_action_samples * normalized_weights + 1e-8).log()
        # Expectation with respect to the non-parametric policy.
        kl_nonparametric = (normalized_weights * integrand).sum(0)

        p_fixed_std = (p[0], p_target[1])
        p_fixed_mean = (p_target[0], p[1])

        loss_policy_mean = self.cross_entropy_loss(p_fixed_std, next_actions_squashed, next_actions,
                                                   normalized_weights)
        loss_policy_stddev = self.cross_entropy_loss(p_fixed_mean, next_actions_squashed, next_actions,
                                                     normalized_weights)

        mean_part, _ = gaussian_kl(self.policy, p_target, p_fixed_std)
        _, cov_part = gaussian_kl(self.policy, p_target, p_fixed_mean)

        loss_kl_mean, loss_alpha_mean = self.kl_penalty_and_dual_loss(mean_part, alpha_mean, self.eps_mu)
        loss_kl_std, loss_alpha_std = self.kl_penalty_and_dual_loss(cov_part, alpha_sigma, self.eps_sigma)

        # Combine losses.
        policy_loss = loss_policy_mean + loss_policy_stddev
        kl_penalty_loss = loss_kl_mean + loss_kl_std
        dual_loss = loss_alpha_mean + loss_alpha_std + loss_eta
        total_loss = policy_loss + kl_penalty_loss

        self.optimizer_policy.zero_grad()
        total_loss.backward()
        self.optimizer_policy.step()

        self.optimizer_dual.zero_grad()
        dual_loss.backward()
        self.optimizer_dual.step()

        if self.total_iters % self.target_policy_update_interval == 0:
            polyak_update(self.policy, self.target_policy, self.polyak_weight)

        loss_dict = collections.OrderedDict(total_loss=total_loss.detach(),
                                            policy_loss=policy_loss.detach(),
                                            kl_penalty_loss=kl_penalty_loss.detach(),
                                            dual_loss=dual_loss.detach())

        stats_dict = collections.OrderedDict(kl_q_rel=kl_nonparametric.detach() / self.eps,
                                             eta=self.log_eta.detach(),
                                             alpha_mean=self.log_alpha_mu.detach(),
                                             alpha_sigma=self.log_alpha_sigma.detach(),
                                             logpacs=self.policy.log_probability(p, next_actions,
                                                                                 pre_squash_x=next_actions_squashed)
                                             )

        return loss_dict, stats_dict

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

            # Find better policy by gradient descent
        for _ in range(self.updates_per_epoch):
            # Sample replay buffer
            self._epoch_steps += 1
            batch = self.replay_buffer.random_batch(self.batch_size)

            next_actions, next_actions_squashed, p, p_target = self.get_action_samples(batch.next_obs, True)

            loss_dict_critic, stats_dict_critic, target_q = self.update_critic(batch, next_actions_squashed, p)
            loss_dict_policy, stats_dict_policy = self.update_policy(p_target, p, next_actions_squashed,
                                                                     next_actions, target_q)

            #######################################################################################################
            # generate one new sample per iteration based on new policy
            # self.replay_buffer.add_samples(self.sample(1, reset_envs=self._epoch_steps == 0))

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
        self.replay_buffer.add_samples(self.sample(self.n_training_samples, reset_envs=True))

        loss_dict, stats_dict = self.epoch_step()

        logging_step = self.total_iters + self.initial_samples
        if self.log_interval != 0 and self._global_steps % self.log_interval == 0:
            self.store.log_table_and_tb('loss', loss_dict, step=logging_step)
            self.store['loss'].flush_row()

            if self.verbose >= 1:
                self.store.log_table_and_tb('stats', stats_dict, step=logging_step)
                self.store['stats'].flush_row()

        loss_dict.update(stats_dict)

        self.lr_schedule_step()

        ################################################################################################################
        # TODO Maybe use more/less samples here in the off-policy setting?
        batch = self.replay_buffer.random_batch(32 * self.batch_size)
        obs = batch.obs
        with ch.no_grad():
            q = self.old_policy(obs)
            # q = self.target_policy(obs)

        metrics_dict = self.log_metrics(obs, q, logging_step)
        loss_dict.update(metrics_dict)

        eval_out = self.evaluate_policy(logging_step, self.evaluate_deterministic, self.evaluate_stochastic)

        # update old policy for projection
        self.old_policy.load_state_dict(self.policy.state_dict())

        return loss_dict, eval_out

    def lr_schedule_step(self):
        if self.lr_schedule_policy:
            self.lr_schedule_policy.step()
            self.lr_schedule_critic.step()
            self.lr_schedule_dual.step()
            lr_dict = {f"lr_policy": self.lr_schedule_policy.get_last_lr()[0],
                       f"lr_critic": self.lr_schedule_critic.get_last_lr()[0],
                       f"lr_dual": self.lr_schedule_dual.get_last_lr()[0]
                       }

            self.store.log_table_and_tb('lr', lr_dict, step=self._global_steps * self.n_training_samples)
            self.store['lr'].flush_row()

    def learn(self):
        """
        Trains a model based on MPO
        """

        rewards = collections.deque(maxlen=5)
        rewards_test = collections.deque(maxlen=5)

        if self.initial_samples > 0:
            self.replay_buffer.add_samples(self.sample(self.initial_samples))

        # start training
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

        target_policy = copy.deepcopy(policy)

        # environments
        sampler = TrajectorySampler(discount_factor=params['replay_buffer']['discount_factor'],
                                    scale_actions=do_squash,
                                    handle_timelimit=handle_timelimit,
                                    cpu=use_cpu, dtype=dtype, seed=seed,
                                    **params['environment'])

        # projections
        projection = get_projection_layer(action_dim=action_dim, total_train_steps=params['training']['train_steps'],
                                          cpu=use_cpu, dtype=dtype, **params['projection'])

        # replay buffer
        replay_buffer = get_replay_buffer(observation_dim=obs_dim, action_dim=action_dim, dtype=dtype, device=device,
                                          handle_timelimit=handle_timelimit, **params['replay_buffer'])

        ch.set_num_threads(1)

        p = MPO(
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

        return p

    def save(self, iteration):
        save_dict = {
            'iteration': iteration,
            'critic': self.critic.state_dict(),
            'policy': self.policy.state_dict(),
            'policy_target': self.target_policy.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
            'optimizer_policy': self.optimizer_policy.state_dict(),
            'optimizer_dual': self.optimizer_dual.state_dict(),
            'log_eta': self.log_eta,
            'log_alpha_mu': self.log_alpha_mu,
            'log_alpha_sigma': self.log_alpha_sigma,
            'envs': self.sampler.envs,
            'envs_test': self.sampler.envs_test,
            # 'sampler': self.sampler
        }
        self.store['checkpoints'].append_row(save_dict)

    @staticmethod
    def agent_from_data(store, train_steps=None):
        pass
