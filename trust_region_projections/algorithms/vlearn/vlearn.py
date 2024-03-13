import collections
import copy
import logging
from typing import Union, Tuple

import fancy_gym
import numpy as np
import torch as ch
from torch import nn as nn

from trust_region_projections.algorithms.abstract_algo import AbstractAlgorithm
from trust_region_projections.losses.loss_factory import get_value_loss_and_critic
from trust_region_projections.losses.value_loss import AbstractCriticLoss
from trust_region_projections.models.policy.abstract_gaussian_policy import AbstractGaussianPolicy
from trust_region_projections.models.policy.policy_factory import get_policy_network
from trust_region_projections.models.value.critic import BaseCritic, TargetCritic, DoubleCritic
from trust_region_projections.projections.base_projection_layer import BaseProjectionLayer
from trust_region_projections.projections.projection_factory import get_projection_layer
from trust_region_projections.replay_buffer.abstract_replay_buffer import AbstractReplayBuffer
from trust_region_projections.replay_buffer.logpac_buffer import LogpacReplayBuffer
from trust_region_projections.replay_buffer.replay_buffer_factory import get_replay_buffer
from trust_region_projections.sampling.dataclass import TrajectoryOffPolicyLogpacs, TrajectoryOffPolicyMixtureRaw, \
    TrajectoryOffPolicyRaw, TrajectoryOffPolicy
from trust_region_projections.sampling.trajectory_sampler import TrajectorySampler
from trust_region_projections.utils.custom_store import CustomStore
from trust_region_projections.utils.network_utils import get_lr_schedule, get_optimizer, polyak_update
from trust_region_projections.utils.projection_utils import gaussian_kl
from trust_region_projections.utils.torch_utils import get_numpy, tensorize, generate_minibatches, select_batch


class VLearning(AbstractAlgorithm):

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
            sample_frequency: int = 1,
            n_training_samples: int = 1,
            initial_samples: int = 10000,

            lr_schedule: str = "",
            clip_grad_norm: Union[float, None] = 40.0,

            target_update_interval: int = 1,
            policy_target_update_interval: int = 1,
            polyak_weight_critic: float = 5e-3,
            polyak_weight_policy_trl: float = 5e-3,
            polyak_weight_policy_log: float = 5e-3,
            log_ratio_clip: float = 1.0,
            trl_policy_update: str = "polyak",
            log_policy_update: str = "avg",
            entropy_coeff: float = 0.0,
            advantage_norm: bool = False,

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
        self.sample_frequency = sample_frequency
        self.n_training_samples = n_training_samples

        # experience replay
        self.replay_buffer = replay_buffer

        ################################################################################################################

        # loss parameters
        self.polyak_weight_critic = polyak_weight_critic
        self.polyak_weight_policy_trl = polyak_weight_policy_trl
        self.polyak_weight_policy_log = polyak_weight_policy_log
        self.log_ratio_clip = log_ratio_clip
        self.target_update_interval = target_update_interval
        self.policy_target_update_interval = policy_target_update_interval
        self.trl_policy_update = trl_policy_update
        self.log_policy_update = log_policy_update
        self.entropy_coeff = entropy_coeff
        self.advantage_norm = advantage_norm

        # networks
        self.critic = critic
        if self.trl_policy_update == "polyak":
            self.old_policy_polyak_trl = copy.deepcopy(self.policy)
        else:
            self.old_policy_copy_trl = copy.deepcopy(self.policy)

        if self.log_policy_update == "polyak":
            self.old_policy_polyak_log = copy.deepcopy(self.policy)

        self.qf_criterion = nn.MSELoss()

        # entropy tuning parameters
        self.auto_entropy_tuning = alpha == "auto"
        if self.auto_entropy_tuning:
            # heuristic value from Haarnoja
            self.target_entropy = -np.prod(self.sampler.action_shape) if target_entropy == "auto" else target_entropy
            self.log_alpha = tensorize(0., cpu=cpu, dtype=dtype).requires_grad_(True)
            self.optimizer_alpha = get_optimizer(optimizer_alpha, [self.log_alpha], lr_alpha)
            self.lr_schedule_alpha = get_lr_schedule(lr_schedule, self.optimizer_alpha, self.train_steps)
        else:
            self.log_alpha = tensorize(alpha, cpu=cpu, dtype=dtype).log()

        # optimizers
        self.optimizer_policy = get_optimizer(optimizer_policy, self.policy.parameters(), lr_policy)
        self.optimizer_critic = get_optimizer(optimizer_critic, self.critic.parameters(), lr_critic)

        self.lr_schedule_policy = get_lr_schedule(lr_schedule, self.optimizer_policy, self.train_steps)
        self.lr_schedule_critic = get_lr_schedule(lr_schedule, self.optimizer_critic, self.train_steps)

        self._epoch_steps = 0
        self._total_samples = 0

        if self.store is not None:
            self.setup_stores()

        self._logger = logging.getLogger('v_learning')

    def setup_stores(self):
        # Logging setup
        super(VLearning, self).setup_stores()

        if self.lr_schedule_policy:
            self.store.add_table('lr', {
                f"lr_policy": float,
                f"lr_critic": float,
            })

        loss_dict = {
            # **self.value_loss.loss_schema,
            'total_loss': float,
            'policy_loss': float,
            'entropy_loss': float,
            'alpha_loss': float,
            'trust_region_loss': float,
        }

        vf_loss = {'vf_loss': float}
        if type(self.critic) is DoubleCritic:
            vf_loss.update({'vf1_loss': float, 'vf2_loss': float})
        loss_dict.update(vf_loss)

        if self.projection.do_regression:
            loss_dict.update({'regression_loss': float})

        self.store.add_table('loss', loss_dict)

        if self.verbose >= 1:

            stats_dict = {
                "current_v": float,
                "advantages": float,
                "q_values": float,
                "bootstrap_v": float,
                "target_v_values": float,
                "ratio": float,
                "ratio_critic": float,
                # **self.value_loss.stats_schema,
                'alpha': float,
                # 'logpacs': float,
                # 'new_logpacs': float,
                # 'old_logpacs': float,
            }

            if type(self.critic) is DoubleCritic:
                stats_dict.update({'current_vf1': float, 'current_vf2': float})

            self.store.add_table('stats', stats_dict)

    def sample(self, n_samples, reset_envs=False, random_actions=False) -> Union[
        TrajectoryOffPolicyMixtureRaw, TrajectoryOffPolicyRaw]:
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
                                    off_policy_logpacs=False, random_actions=random_actions)
        if isinstance(self.replay_buffer, LogpacReplayBuffer):
            traj = TrajectoryOffPolicyMixtureRaw(*traj, policy=copy.deepcopy(self.policy))
        else:
            traj = TrajectoryOffPolicyRaw(*traj)
        return traj

    def update_critic(self, batch: Union[TrajectoryOffPolicyLogpacs, TrajectoryOffPolicy]):

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

        # entropy
        with ch.no_grad():

            p = self.policy(batch.obs)
            if self.trl_policy_update == "polyak":
                q = self.old_policy_polyak_trl(batch.obs)
            elif self.trl_policy_update == "copy":
                q = self.old_policy_copy_trl(batch.obs)

            proj_p = self.projection(self.policy, p, q, self._global_steps)

            # ratio
            new_logpacs = self.policy.log_probability(proj_p, batch.actions)
            if self.log_policy_update == "polyak":
                old_logpacs = self.old_policy_polyak_log.log_probability(q, batch.actions).detach()
            elif self.log_policy_update == "avg":
                old_logpacs = batch.logpacs
            ratio = (new_logpacs - old_logpacs).exp()
            if self.log_ratio_clip > 0:
                ratio = ratio.clamp(max=self.log_ratio_clip)
            # ratio = ratio / ratio.sum()

            # entropy = - self.log_alpha.exp() * new_logpacs
            alpha = self.log_alpha.exp()
            entropy = alpha * self.policy.entropy(proj_p)
            # entropy = 0

        # qf_loss, loss_dict, info_vals = self.value_loss(self.critic, batch, next_actions, old_next_logpacs, entropy)

        # value function now contains the entropy for every time step in the target.
        # In SAC the Q-function only contains the entropy of all steps BUT the first,
        # so we would not need to add it here
        target_v_values = self.critic.target(batch.next_obs)
        v_target = (batch.rewards + entropy + batch.terminals * target_v_values).detach()

        # We do not want inheritance here, hence no isinstance check
        if type(self.critic) is TargetCritic:
            current_vf = self.critic(batch.obs)
            critic_loss = ratio * (current_vf - v_target) ** 2
        elif type(self.critic) is DoubleCritic:
            current_vf1 = self.critic.q1(batch.obs)
            current_vf2 = self.critic.q2(batch.obs)
            critic1_loss = ratio * (current_vf1 - v_target) ** 2
            critic2_loss = ratio * (current_vf2 - v_target) ** 2
            critic_loss = critic1_loss + critic2_loss
            # TODO hack for logging
            current_vf = ch.min(current_vf1, current_vf2)
        else:
            raise ValueError("Invalid critic.")

        critic_loss = critic_loss.mean()
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        if self.clip_grad_norm > 0:
            ch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad_norm)
        self.optimizer_critic.step()

        if self.total_iters % self.target_update_interval == 0:
            self.critic.update_target_net(self.polyak_weight_critic)

        info_vals = collections.OrderedDict(current_v=current_vf.detach(),
                                            bootstrap_v=v_target.detach(),
                                            target_v_values=target_v_values.detach(),
                                            ratio_critic=ratio.detach(),
                                            alpha=alpha,
                                            # new_logpacs=new_logpacs.detach(),
                                            # old_logpacs=batch.logpacs.detach()
                                            )
        loss_dict = collections.OrderedDict(vf_loss=critic_loss.detach())

        if isinstance(self.critic, DoubleCritic):
            info_vals.update(
                collections.OrderedDict(current_vf1=current_vf1.detach(), current_vf2=current_vf2.detach()))
            loss_dict.update(collections.OrderedDict(vf1_loss=critic1_loss.detach(), vf2_loss=critic2_loss.detach()))

        return loss_dict, info_vals

    def _update_alpha(self, logpacs: ch.Tensor):
        """
        Update alpha parameter for automatic entropy tuning.
        Args:
            logpacs: log probabilies of current policy

        Returns:
            alpha_loss
        """
        if self.auto_entropy_tuning:
            alpha_loss = (-self.log_alpha * (logpacs + self.target_entropy).detach()).mean()
            self.optimizer_alpha.zero_grad()
            alpha_loss.backward()
            self.optimizer_alpha.step()
        else:
            alpha_loss = logpacs.new_tensor(0.)

        return {"alpha_loss": alpha_loss.detach()}

    def update_policy(self, batch: Union[TrajectoryOffPolicyLogpacs, TrajectoryOffPolicy]):
        """
        Update Policy network and Alpha.
        Args:
            batch: batch observations

        Returns:
            loss_dict{total_loss, policy_loss, alpha_loss, trust_region_loss}, info_vals{alpha, logpacs}
        """

        p = self.policy(batch.obs)
        if self.trl_policy_update == "polyak":
            q = self.old_policy_polyak_trl(batch.obs)
        elif self.trl_policy_update == "copy":
            q = self.old_policy_copy_trl(batch.obs)
        proj_p = self.projection(self.policy, p, q, self._global_steps)

        new_logpacs = self.policy.log_probability(proj_p, batch.actions)
        if self.log_policy_update == "polyak":
            q_log = self.old_policy_polyak_log(batch.obs)
            old_logpacs = self.old_policy_polyak_log.log_probability(q_log, batch.actions).detach()
        elif self.log_policy_update == "avg":
            old_logpacs = batch.logpacs.detach()
        ratio = (new_logpacs - old_logpacs).exp()
        if self.log_ratio_clip > 0:
            ratio = ratio.clamp(max=self.log_ratio_clip)
        # ratio = ratio / ratio.sum().detach()

        # loss_dict = {}
        loss_dict = self._update_alpha(new_logpacs)

        with ch.no_grad():
            # entropy
            next_v_values = self.critic(batch.next_obs)
            q_values = (batch.rewards + batch.terminals * next_v_values)
            advantages = q_values - self.critic(batch.obs)
            advs = advantages

        if self.advantage_norm == "awr":
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantage_weights = ch.exp(advantages / 0.05)
            advantage_weights = advantage_weights.clamp(max=20)

            if self.log_ratio_clip > 0:
                new_logpacs = new_logpacs.clamp(max=self.log_ratio_clip)

            policy_loss = (-new_logpacs * advantage_weights.detach()).mean()
        else:
            if self.advantage_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            policy_loss = (- ratio * advantages.detach()).mean()

        # Calculate trust region loss
        trust_region_loss = self.projection.get_trust_region_loss(self.policy, p, proj_p)
        entropy_loss = self.entropy_coeff * self.policy.entropy(proj_p)
        total_loss = policy_loss + trust_region_loss + entropy_loss

        # Gradient step
        self.optimizer_policy.zero_grad()
        total_loss.mean().backward()
        if self.clip_grad_norm > 0:
            ch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad_norm)
        self.optimizer_policy.step()

        info_vals = collections.OrderedDict(
            ratio=ratio.detach().mean(),
            # alpha=alpha.detach(),
            advantages=advs.detach(),
            q_values=q_values.detach(),
            # logpacs=logpacs.detach(),
            # logpacs=new_logpacs.detach()
        )

        loss_dict.update(collections.OrderedDict(
            total_loss=total_loss.detach(),
            policy_loss=policy_loss.detach(),
            trust_region_loss=trust_region_loss.detach(),
            entropy_loss=entropy_loss.detach(),
        ))

        # update old polyak policy
        if self.total_iters % self.policy_target_update_interval == 0:
            if self.trl_policy_update == "polyak":
                polyak_update(self.policy, self.old_policy_polyak_trl, self.polyak_weight_policy_trl)
            if self.log_policy_update == "polyak":
                polyak_update(self.policy, self.old_policy_polyak_log, self.polyak_weight_policy_log)

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
            # generate new samples per iteration based on new policy
            if self._epoch_steps % self.sample_frequency == 0:
                self._total_samples += self.n_training_samples
                self.replay_buffer.add_samples(self.sample(self.n_training_samples))

            # increment after sampling block to sample in first iteration
            self._epoch_steps += 1

            batch = self.replay_buffer.random_batch(self.batch_size)

            loss_dict_critic, stats_dict_critic = self.update_critic(batch)
            loss_dict_policy, stats_dict_policy = self.update_policy(batch)

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

        # Add all samples in the beginning
        # rlkit uses it like this, original code and stable-baselines add them one by one for each updates_per_epoch
        # self.replay_buffer.add_samples(self.sample(self.n_training_samples, reset_envs=True))

        if self.projection.initial_entropy is None:
            obs = self.replay_buffer.random_batch(self.replay_buffer.size).obs
            if self.trl_policy_update == "polyak":
                q = self.old_policy_polyak_trl(obs)
            elif self.trl_policy_update == "copy":
                q = self.old_policy_copy_trl(obs)
            self.projection.initial_entropy = self.policy.entropy(q).mean().detach()

        loss_dict, stats_dict = self.epoch_step()
        loss_dict.update(self.regression_step(None, None, self.batch_size, self._total_samples))

        if self.log_interval != 0 and self._global_steps % self.log_interval == 0:
            self.store.log_table_and_tb('loss', loss_dict, step=self._total_samples)
            self.store['loss'].flush_row()
            if self.verbose >= 1:
                self.store.log_table_and_tb('stats', stats_dict, step=self._total_samples)
                self.store['stats'].flush_row()

        ################################################################################################################
        # TODO Maybe use more/less samples here in the off-policy setting?
        batch = self.replay_buffer.random_batch(min(8 * self.batch_size, self.replay_buffer.size))
        obs = batch.obs
        with ch.no_grad():
            if self.trl_policy_update == "polyak":
                q = self.old_policy_polyak_trl(obs)
            elif self.trl_policy_update == "copy":
                q = self.old_policy_copy_trl(obs)

        metrics_dict = self.log_metrics(obs, q, self._total_samples)

        self.lr_schedule_step(obs, q)

        loss_dict.update(metrics_dict)
        loss_dict.update(stats_dict)

        eval_out = self.evaluate_policy(self._total_samples, self.evaluate_deterministic, self.evaluate_stochastic)

        if self.trl_policy_update == "copy":
            # update old copied policy
            self.old_policy_copy_trl.load_state_dict(self.policy.state_dict())

        return loss_dict, eval_out

    def evaluate_policy(self, logging_step, evaluate_deterministic: bool = True, evaluate_stochastic: bool = False,
                        render: bool = False, test_mode: bool = False, render_mode: str = "human"):
        """
        Evaluates the current policy on the test environments.
        Args:
            logging_step: Current logging step
            render: Render policy (if applicable)
            evaluate_deterministic: Make policy actions deterministic for testing (Can be used jointly with stochastic)
            evaluate_stochastic: Make policy actions stochastic for testing (Can be used jointly with deterministic)
            test_mode: disables any logging and purely executes environments

        Returns:
            exploration_dict, evaluation_dict, expectation_dict
        """
        exploration_dict, exploration_success_rate = {}, {}
        if not test_mode:
            exploration_dict, exploration_success_rate = self.sampler.get_exploration_performance()

        evaluation_dict = evaluation_success_rate = {}
        expectation_success_rate = expectation_dict = {}

        policy = (self.policy, self.old_policy_polyak_trl, self.projection) \
            if self.trl_policy_update == "polyak" else self.policy

        if evaluate_deterministic:
            evaluation_dict, evaluation_success_rate = \
                self.sampler.evaluate_policy(policy, render=render, deterministic=True, render_mode=render_mode)
        if evaluate_stochastic:
            expectation_dict, expectation_success_rate = \
                self.sampler.evaluate_policy(policy, render=render, deterministic=False, render_mode=render_mode)

        if self.log_interval != 0 and self._global_steps % self.log_interval == 0 and not test_mode:
            if self.verbose >= 2:

                self._logger.info(self.generate_performance_string(exploration_dict, "reward", "exploration"))
                self._logger.info(self.generate_performance_string(exploration_success_rate, "success_rate",
                                                                   "exploration"))

                if self.evaluate_deterministic:
                    self._logger.info(self.generate_performance_string(evaluation_dict, "reward"))
                    self._logger.info(self.generate_performance_string(evaluation_success_rate, "success_rate"))

                if evaluate_stochastic:
                    self._logger.info(self.generate_performance_string(expectation_dict, "reward", "expectation"))
                    self._logger.info(self.generate_performance_string(expectation_success_rate, "success_rate",
                                                                       "expectation"))

            self.store.log_table_and_tb('exploration_reward', exploration_dict, step=logging_step)
            self.store.log_table_and_tb('exploration_success_rate', exploration_success_rate, step=logging_step)
            self.store['exploration_reward'].flush_row()
            self.store['exploration_success_rate'].flush_row()

            if evaluate_deterministic:
                self.store.log_table_and_tb('evaluation_reward', evaluation_dict, step=logging_step)
                self.store.log_table_and_tb('evaluation_success_rate', evaluation_success_rate, step=logging_step)
                self.store['evaluation_reward'].flush_row()
                self.store['evaluation_success_rate'].flush_row()

            if evaluate_stochastic:
                self.store.log_table_and_tb('expectation_reward', expectation_dict, step=logging_step)
                self.store.log_table_and_tb('expectation_success_rate', expectation_success_rate, step=logging_step)
                self.store['expectation_reward'].flush_row()
                self.store['expectation_success_rate'].flush_row()

        return collections.OrderedDict(exploration=(exploration_dict, exploration_success_rate),
                                       evaluation=(evaluation_dict, evaluation_success_rate),
                                       expectation=(expectation_dict, expectation_success_rate))

    def add_initial_samples(self):
        if self._global_steps == 0 and self.initial_samples > 0:
            self._total_samples += self.initial_samples
            self.replay_buffer.add_samples(self.sample(self.initial_samples, random_actions=True))

    def regression_step(self, obs: ch.Tensor, q: Tuple[ch.Tensor, ch.Tensor], n_minibatches: int, logging_step: int):
        """
        Execute additional regression steps to match policy output and projection
        The policy parameters are updated in-place.
        Args:
            obs: observations from sampling
            q: old distribution
            n_minibatches: batch size for regression
            logging_step: step index for logging

        Returns:
            dict of mean regression loss
        """

        out = {}
        if self.projection.do_regression:
            # get prediction before the regression to compare to regressed policy
            # with ch.no_grad():
            #     p = self.policy(obs)
            #     p_proj = self.projection(self.policy, p, q, self._global_steps)

            # if self.verbose >= 2:
            #     self.store.log_table_and_tb('constraints_initial',
            #                                 self.projection.compute_metrics(self.policy, p, q, self._global_steps),
            #                                 step=logging_step)
            #     self.store.log_table_and_tb('constraints_projection',
            #                                 self.projection.compute_metrics(self.policy, p_proj, q, self._global_steps),
            #                                 step=logging_step)
            #     self.store['constraints_initial'].flush_row()
            #     self.store['constraints_projection'].flush_row()

            policy_unprojected = copy.deepcopy(self.policy)
            optim_reg = get_optimizer(self.projection.optimizer_type_reg, policy_unprojected.parameters(),
                                      learning_rate=self.projection.lr_reg)

            reg_losses = ch.zeros(1).item()

            for i in range(self.projection.regression_iters):
                batch = self.replay_buffer.random_batch(self.batch_size)
                obs = batch.obs

                with ch.no_grad():
                    # get current projected values --> targets for regression
                    flat_p = self.policy(obs)
                    if self.trl_policy_update == "polyak":
                        q = self.old_policy_polyak_trl(obs)
                    elif self.trl_policy_update == "copy":
                        q = self.old_policy_copy_trl(obs)
                    proj_p = self.projection(self.policy, flat_p, q, self._total_samples)

                p = policy_unprojected(obs)

                # invert scaling with coeff here as we do not have to balance with other losses
                # loss = self.projection.get_trust_region_loss(self.policy, p,
                #                                              proj_p) / self.projection.trust_region_coeff
                m, c = gaussian_kl(self.policy, p, proj_p)
                loss = (m + c).mean()

                optim_reg.zero_grad()
                loss.backward()
                optim_reg.step()
                reg_losses += loss.detach()

                if loss <= 1e-3:
                    break

            self.policy.load_state_dict(policy_unprojected.state_dict())

            # TODO
            # if not self.policy.contextual_std:
            #     # set policy with projection value.
            #     # In non-contextual cases we have only one cov, so the projection is the same.
            #     self.policy.set_std(p_target[1][0])

            out = {"regression_loss": (reg_losses / (i + 1))}

        # return self.projection.trust_region_regression(self.policy, obs, q, n_minibatches, self._global_steps)
        return out

    def lr_schedule_step(self, obs=None, q=None):
        is_tuple = isinstance(self.lr_schedule_policy, tuple)
        if self.lr_schedule_policy and not is_tuple:
            self.lr_schedule_policy.step()
            self.lr_schedule_critic.step()
            # self.lr_schedule_alpha.step()
            lr_dict = {f"lr_policy": self.lr_schedule_policy.get_last_lr()[0],
                       f"lr_critic": self.lr_schedule_critic.get_last_lr()[0],
                       # f"lr_alpha": self.lr_schedule_alpha.get_last_lr()[0]
                       }

            self.store.log_table_and_tb('lr', lr_dict, step=self._total_samples)
            self.store['lr'].flush_row()
        elif is_tuple:
            with ch.no_grad():
                p = self.policy(obs)
                constraint = ch.stack(self.projection.trust_region_value(self.policy, p, q), dim=-1).sum(-1).flatten()
                violation = constraint > self.projection.mean_bound + self.projection.cov_bound
                rel = violation.count_nonzero() / len(violation)
            if rel > 0.25:
                # Too many samples violate bounds
                # First schedule reduces lr by factor 0.8
                self.lr_schedule_policy[0].step()
            else:
                # Second schedule increases lr by factor 0.8
                self.lr_schedule_policy[1].step()

            lr_dict = {f"lr_policy": self.optimizer_policy.param_groups[0]['lr'],
                       f"lr_critic": self.optimizer_critic.param_groups[0]['lr']
                       # f"lr_alpha": self.lr_schedule_alpha.get_last_lr()[0]
                       }

            self.store.log_table_and_tb('lr', lr_dict, step=self._total_samples)
            self.store['lr'].flush_row()

    def learn(self):

        rewards = collections.deque(maxlen=5)
        rewards_test = collections.deque(maxlen=5)

        self.add_initial_samples()

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
        agent = VLearning.agent_from_params(agent_params, store)

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
        policy = get_policy_network(proj_type=params['projection']['proj_type'], device=device,
                                    dtype=dtype, obs_dim=obs_dim, action_dim=action_dim, **params['policy'])

        # environments
        sampler = TrajectorySampler(discount_factor=params['replay_buffer']['discount_factor'],
                                    scale_actions=params['policy']["squash"],
                                    handle_timelimit=handle_timelimit,
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

        p = VLearning(
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
