import abc
import copy
import logging
from collections import OrderedDict
from typing import Tuple, Union

import torch as ch

from trust_region_projections.models.policy.abstract_gaussian_policy import AbstractGaussianPolicy
from trust_region_projections.projections.base_projection_layer import BaseProjectionLayer
from trust_region_projections.sampling.trajectory_sampler import TrajectorySampler
from trust_region_projections.utils.custom_store import CustomStore


class AbstractAlgorithm(abc.ABC):
    def __init__(self, policy: AbstractGaussianPolicy, sampler: TrajectorySampler, projection: BaseProjectionLayer,
                 train_steps: int = 1000, clip_grad_norm: Union[float, None] = 0.5, store: CustomStore = None,
                 verbose: int = 1, evaluate_deterministic=True, evaluate_stochastic=False,
                 log_interval: int = 5, save_interval: int = -1, seed: int = 1, cpu: bool = True,
                 dtype: ch.dtype = ch.float32):

        """
        Abstract algorithm interface
        Args:
            sampler: Takes care of generating trajectory samples.
            policy: An `AbstractPolicy` which maps observations to action distributions.
            train_steps: Total number of training steps.
            clip_grad_norm: Gradient norm clipping.
            store: Cox store
            verbose: logging level of [0,1,2], higher values also include previous levels.
            0: basic values, 1: extended stats, 2: console logging
            evaluate_deterministic: Evaluate policy after each epoch deterministically
            evaluate_stochastic: Evaluate policy after each epoch stochastically
            log_interval: How often to log.
            save_interval: How often to save model.
            seed: Seed for generating envs
            cpu: Compute on CPU only.
            dtype: Data type to use, either of float32 or float64. The later might be necessary for higher
                    dimensions in order to learn the full covariance.
        """

        # Policy
        self.policy = policy
        self.old_policy = copy.deepcopy(self.policy)

        # training steps
        self.train_steps = train_steps
        self.clip_grad_norm = clip_grad_norm
        self._global_steps = 0

        # Environments
        self.sampler = sampler

        # projection
        self.projection = projection

        # Config
        self.seed = seed
        self.cpu = cpu
        self.device = "cpu" if cpu else "cuda"
        self.dtype = dtype

        # logging
        self.evaluate_deterministic = evaluate_deterministic
        self.evaluate_stochastic = evaluate_stochastic
        self.save_interval = save_interval
        self.verbose = verbose
        self.log_interval = log_interval

        self.store = store

        self._logger = logging.getLogger('abstract_algorithm')

    def setup_stores(self):
        base_schema = {
            'mean': float,
            'median': float,
            'std': float,
            'min': float,
            'max': float,
        }
        reward_schema = {'step_reward': float,
                         'length': float,
                         'length_std': float,
                         **base_schema}
        self.store.add_table('exploration_reward', reward_schema)
        self.store.add_table('exploration_success_rate', base_schema)
        if self.evaluate_deterministic:
            self.store.add_table('evaluation_reward', reward_schema)
            self.store.add_table('evaluation_success_rate', base_schema)
        if self.evaluate_stochastic:
            self.store.add_table('expectation_reward', reward_schema)
            self.store.add_table('expectation_success_rate', base_schema)

        # Table for final results
        self.store.add_table('final_results', {
            'iteration': int,
            '5_rewards': float,
            '5_rewards_test': float
        })

        constraint_schema = {
            'kl': float,
            'constraint': float,
            'mean_constraint': float,
            'cov_constraint': float,
            'entropy': float,
            'entropy_diff': float,
            'kl_max': float,
            'constraint_max': float,
            'mean_constraint_max': float,
            'cov_constraint_max': float,
            'entropy_max': float,
            'entropy_diff_max': float,
        }

        if self.projection.has_entropy_control:
            constraint_schema.update({'entropy_constraint': float})

        self.store.add_table('constraints', constraint_schema)

        if self.verbose >= 2:
            constraint_dist_schema = dict(kl=float,
                                          constraint=float,
                                          mean_constraint=float,
                                          cov_constraint=float,
                                          entropy=float,
                                          entropy_diff=float, )
            self.store.add_table('constraint_distribution_pre', constraint_dist_schema)
            self.store.add_table('constraint_distribution_post', constraint_dist_schema)

            self.store.add_table('distribution', {
                'mean': self.store.PICKLE,
                'std': self.store.PICKLE,
            })

            if self.projection and self.projection.do_regression:
                self.store.add_table('constraints_initial', constraint_schema)
                self.store.add_table('constraints_projection', constraint_schema)

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

        if evaluate_deterministic:
            evaluation_dict, evaluation_success_rate = \
                self.sampler.evaluate_policy(self.policy, render=render, deterministic=True, render_mode=render_mode)
        if evaluate_stochastic:
            expectation_dict, expectation_success_rate = \
                self.sampler.evaluate_policy(self.policy, render=render, deterministic=False, render_mode=render_mode)

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

        return OrderedDict(exploration=(exploration_dict, exploration_success_rate),
                           evaluation=(evaluation_dict, evaluation_success_rate),
                           expectation=(expectation_dict, expectation_success_rate))

    @staticmethod
    def generate_performance_string(performance_dict: dict, metric_type="reward", type="evaluation"):
        if not performance_dict: return ""
        perf_str = f"Avg. {type} {metric_type}: {performance_dict['mean']:.4f} +/- {performance_dict['std']:.4f}| " \
                   f"Median {type} {metric_type}: {performance_dict['median']:.4f}| " \
                   f"Min/Max {type} {metric_type}: {performance_dict['min']:.4f}/{performance_dict['max']:.4f} | "
        if performance_dict.get('length') is not None:
            perf_str += f"Avg. step {type} {metric_type}: {performance_dict['step_reward']:.4f} | " \
                        f"Avg. {type} episode length: {performance_dict['length']:.4f} +/- " \
                        f"{performance_dict['length_std'] :.2f}"

        return perf_str

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
        if self.projection.do_regression:
            # get prediction before the regression to compare to regressed policy
            with ch.no_grad():
                p = self.policy(obs)
                p_proj = self.projection(self.policy, p, q, self._global_steps)
                # TODO: self.policy.squash()

            if self.verbose >= 2:
                self.store.log_table_and_tb('constraints_initial',
                                            self.projection.compute_metrics(self.policy, p, q, self._global_steps),
                                            step=logging_step)
                self.store.log_table_and_tb('constraints_projection',
                                            self.projection.compute_metrics(self.policy, p_proj, q, self._global_steps),
                                            step=logging_step)
                self.store['constraints_initial'].flush_row()
                self.store['constraints_projection'].flush_row()

        return self.projection.trust_region_regression(self.policy, obs, q, n_minibatches, self._global_steps)

    def log_metrics(self, obs, q, logging_step):
        """
        Computes and logs the trust region metrics.
        Args:
            obs: observations used for evaluation
            q: old distributions
            logging_step: current logging step

        Returns:
            dict of trust region metrics
        """
        with ch.no_grad():
            p = self.policy(obs)
        metrics_dict = self.projection.compute_metrics(self.policy, p, q, self._global_steps)

        if self.log_interval != 0 and self._global_steps % self.log_interval == 0:
            self.store.log_table_and_tb('constraints', metrics_dict, step=logging_step)
            self.store['constraints'].flush_row()

            if self.verbose >= 3 and self._global_steps % (self.log_interval * 10) == 0:
                histo = self.projection.compute_metrics(self.policy, p, q, self._global_steps, aggregate=False)
                histo = {k: v.clip(0, self.projection.mean_bound * 3) for k, v in histo.items()}
                self.store.log_tb('constraint_distribution_pre', histo, step=logging_step, summary_type='histogram')
                proj_p = self.projection(self.policy, p, q, self._global_steps)
                histo = self.projection.compute_metrics(self.policy, proj_p, q, self._global_steps, aggregate=False)
                self.store.log_tb('constraint_distribution_post', histo, step=logging_step, summary_type='histogram')
        # self.store.log_table_and_tb('distribution', {
        #         'mean': get_numpy(p[0]),
        #         'std': get_numpy(p[1]),
        #     }, step=logging_step, summary_type="histogram")
        #     self.store['distribution'].flush_row()

        return metrics_dict

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def learn(self):
        pass

    @abc.abstractmethod
    def save(self, iteration):
        pass

    @staticmethod
    @abc.abstractmethod
    def agent_from_data(store, train_steps=None):
        pass

    @staticmethod
    @abc.abstractmethod
    def agent_from_params(params, store=None):
        pass

# import abc
# import copy
# import logging
# from collections import OrderedDict
# from typing import Tuple, Union
#
# import torch as ch
#
# from trust_region_projections.models.policy.abstract_gaussian_policy import AbstractGaussianPolicy
# from trust_region_projections.projections.base_projection_layer import BaseProjectionLayer
# from trust_region_projections.sampling.trajectory_sampler import TrajectorySampler
# from trust_region_projections.utils.custom_store import CustomStore
#
#
# class AbstractAlgorithm(abc.ABC):
#     def __init__(self, policy: AbstractGaussianPolicy, sampler: TrajectorySampler, projection: BaseProjectionLayer,
#                  train_steps: int = 1000, clip_grad_norm: Union[float, None] = 0.5, store: CustomStore = None,
#                  verbose: int = 1, evaluate_deterministic=True, evaluate_stochastic=False,
#                  log_interval: int = 5, save_interval: int = -1, seed: int = 1, cpu: bool = True,
#                  dtype: ch.dtype = ch.float32):
#
#         """
#         Abstract algorithm interface
#         Args:
#             sampler: Takes care of generating trajectory samples.
#             policy: An `AbstractPolicy` which maps observations to action distributions.
#             train_steps: Total number of training steps.
#             clip_grad_norm: Gradient norm clipping.
#             store: Cox store
#             verbose: logging level of [0,1,2], higher values also include previous levels.
#             0: basic values, 1: extended stats, 2: console logging
#             evaluate_deterministic: Evaluate policy after each epoch deterministically
#             evaluate_stochastic: Evaluate policy after each epoch stochastically
#             log_interval: How often to log.
#             save_interval: How often to save model.
#             seed: Seed for generating envs
#             cpu: Compute on CPU only.
#             dtype: Data type to use, either of float32 or float64. The later might be necessary for higher
#                     dimensions in order to learn the full covariance.
#         """
#
#         # Policy
#         self.policy = policy
#         self.old_policy = copy.deepcopy(self.policy)
#
#         # training steps
#         self.train_steps = train_steps
#         self.clip_grad_norm = clip_grad_norm
#         self._global_steps = 0
#
#         # Environments
#         self.sampler = sampler
#
#         # projection
#         self.projection = projection
#
#         # Config
#         self.seed = seed
#         self.cpu = cpu
#         self.device = "cpu" if cpu else "cuda"
#         self.dtype = dtype
#
#         # logging
#         self.evaluate_deterministic = evaluate_deterministic
#         self.evaluate_stochastic = evaluate_stochastic
#         self.save_interval = save_interval
#         self.verbose = verbose
#         self.log_interval = log_interval
#
#         self.store = store
#
#         self._logger = logging.getLogger('abstract_algorithm')
#
#     def setup_stores(self):
#         base_schema = {
#             'mean': float,
#             'median': float,
#             'std': float,
#             'min': float,
#             'max': float,
#         }
#         reward_schema = {'step_reward': float,
#                          'length': float,
#                          'length_std': float,
#                          **base_schema}
#         self.store.add_table('exploration_reward', reward_schema)
#         self.store.add_table('exploration_success_rate', base_schema)
#         if self.evaluate_deterministic:
#             self.store.add_table('evaluation_reward', reward_schema)
#             self.store.add_table('evaluation_success_rate', base_schema)
#             self.store.add_table('evaluation_max_heights', base_schema)
#             self.store.add_table('evaluation_goal_dists', base_schema)
#             self.store.add_table('evaluation_is_healthy', base_schema)
#             self.store.add_table('evaluation_contact_dists', base_schema)
#         if self.evaluate_stochastic:
#             self.store.add_table('expectation_reward', reward_schema)
#             self.store.add_table('expectation_success_rate', base_schema)
#
#         # Table for final results
#         self.store.add_table('final_results', {
#             'iteration': int,
#             '5_rewards': float,
#             '5_rewards_test': float
#         })
#
#         constraint_schema = {
#             'kl': float,
#             'constraint': float,
#             'mean_constraint': float,
#             'cov_constraint': float,
#             'entropy': float,
#             'entropy_diff': float,
#             'kl_max': float,
#             'constraint_max': float,
#             'mean_constraint_max': float,
#             'cov_constraint_max': float,
#             'entropy_max': float,
#             'entropy_diff_max': float,
#         }
#
#         if self.projection.has_entropy_control:
#             constraint_schema.update({'entropy_constraint': float})
#
#         self.store.add_table('constraints', constraint_schema)
#
#         if self.verbose >= 2:
#             constraint_dist_schema = dict(kl=float,
#                                           constraint=float,
#                                           mean_constraint=float,
#                                           cov_constraint=float,
#                                           entropy=float,
#                                           entropy_diff=float, )
#             self.store.add_table('constraint_distribution_pre', constraint_dist_schema)
#             self.store.add_table('constraint_distribution_post', constraint_dist_schema)
#
#             self.store.add_table('distribution', {
#                 'mean': self.store.PICKLE,
#                 'std': self.store.PICKLE,
#             })
#
#             if self.projection and self.projection.do_regression:
#                 self.store.add_table('constraints_initial', constraint_schema)
#                 self.store.add_table('constraints_projection', constraint_schema)
#
#     def evaluate_policy(self, logging_step, evaluate_deterministic: bool = True, evaluate_stochastic: bool = False,
#                         render: bool = False, test_mode: bool = False, render_mode: str = "human"):
#         """
#         Evaluates the current policy on the test environments.
#         Args:
#             logging_step: Current logging step
#             render: Render policy (if applicable)
#             evaluate_deterministic: Make policy actions deterministic for testing (Can be used jointly with stochastic)
#             evaluate_stochastic: Make policy actions stochastic for testing (Can be used jointly with deterministic)
#             test_mode: disables any logging and purely executes environments
#
#         Returns:
#             exploration_dict, evaluation_dict, expectation_dict
#         """
#         exploration_dict, exploration_success_rate = {}, {}
#         if not test_mode:
#             exploration_dict, exploration_success_rate = self.sampler.get_exploration_performance()
#
#         evaluation_dict = evaluation_success_rate = evaluation_max_heights = evaluation_goal_dists = \
#             evaluation_is_healthy = evaluation_contact_dists = {}, {}, {}, {}, {}, {}
#         expectation_success_rate = expectation_dict = {}
#
#         if evaluate_deterministic:
#             evaluation_dict, evaluation_success_rate, evaluation_max_heights, evaluation_goal_dists, \
#             evaluation_is_healthy, evaluation_contact_dists = \
#                 self.sampler.evaluate_policy(self.policy, render=render, deterministic=True, render_mode=render_mode)
#         if evaluate_stochastic:
#             expectation_dict, expectation_success_rate, _, _, _, _ = \
#                 self.sampler.evaluate_policy(self.policy, render=render, deterministic=False, render_mode=render_mode)
#
#         if self.log_interval != 0 and self._global_steps % self.log_interval == 0 and not test_mode:
#             if self.verbose >= 2:
#
#                 self._logger.info(self.generate_performance_string(exploration_dict, "reward", "exploration"))
#                 self._logger.info(self.generate_performance_string(exploration_success_rate, "success_rate",
#                                                                    "exploration"))
#
#                 if self.evaluate_deterministic:
#                     self._logger.info(self.generate_performance_string(evaluation_dict, "reward"))
#                     self._logger.info(self.generate_performance_string(evaluation_success_rate, "success_rate"))
#
#                 if evaluate_stochastic:
#                     self._logger.info(self.generate_performance_string(expectation_dict, "reward", "expectation"))
#                     self._logger.info(self.generate_performance_string(expectation_success_rate, "success_rate",
#                                                                        "expectation"))
#
#             self.store.log_table_and_tb('exploration_reward', exploration_dict, step=logging_step)
#             self.store.log_table_and_tb('exploration_success_rate', exploration_success_rate, step=logging_step)
#             self.store['exploration_reward'].flush_row()
#             self.store['exploration_success_rate'].flush_row()
#
#             if evaluate_deterministic:
#                 self.store.log_table_and_tb('evaluation_reward', evaluation_dict, step=logging_step)
#                 self.store.log_table_and_tb('evaluation_success_rate', evaluation_success_rate, step=logging_step)
#                 self.store.log_table_and_tb('evaluation_max_heights', evaluation_max_heights, step=logging_step)
#                 self.store.log_table_and_tb('evaluation_goal_dists', evaluation_goal_dists, step=logging_step)
#                 self.store.log_table_and_tb('evaluation_is_healthy', evaluation_is_healthy, step=logging_step)
#                 self.store.log_table_and_tb('evaluation_contact_dists', evaluation_contact_dists, step=logging_step)
#                 self.store['evaluation_reward'].flush_row()
#                 self.store['evaluation_success_rate'].flush_row()
#                 self.store['evaluation_max_heights'].flush_row()
#                 self.store['evaluation_goal_dists'].flush_row()
#                 self.store['evaluation_is_healthy'].flush_row()
#                 self.store['evaluation_contact_dists'].flush_row()
#
#             if evaluate_stochastic:
#                 self.store.log_table_and_tb('expectation_reward', expectation_dict, step=logging_step)
#                 self.store.log_table_and_tb('expectation_success_rate', expectation_success_rate, step=logging_step)
#
#                 self.store['expectation_reward'].flush_row()
#                 self.store['expectation_success_rate'].flush_row()
#
#         return OrderedDict(exploration=(exploration_dict, exploration_success_rate),
#                            evaluation=(evaluation_dict, evaluation_success_rate, evaluation_max_heights,
#                                        evaluation_goal_dists, evaluation_is_healthy, evaluation_contact_dists),
#                            expectation=(expectation_dict, expectation_success_rate))
#
#     @staticmethod
#     def generate_performance_string(performance_dict: dict, metric_type="reward", type="evaluation"):
#         if not performance_dict: return ""
#         perf_str = f"Avg. {type} {metric_type}: {performance_dict['mean']:.4f} +/- {performance_dict['std']:.4f}| " \
#                    f"Median {type} {metric_type}: {performance_dict['median']:.4f}| " \
#                    f"Min/Max {type} {metric_type}: {performance_dict['min']:.4f}/{performance_dict['max']:.4f} | "
#         if performance_dict.get('length') is not None:
#             perf_str += f"Avg. step {type} {metric_type}: {performance_dict['step_reward']:.4f} | " \
#                         f"Avg. {type} episode length: {performance_dict['length']:.4f} +/- " \
#                         f"{performance_dict['length_std'] :.2f}"
#
#         return perf_str
#
#     def regression_step(self, obs: ch.Tensor, q: Tuple[ch.Tensor, ch.Tensor], n_minibatches: int, logging_step: int):
#         """
#         Execute additional regression steps to match policy output and projection
#         The policy parameters are updated in-place.
#         Args:
#             obs: observations from sampling
#             q: old distribution
#             n_minibatches: batch size for regression
#             logging_step: step index for logging
#
#         Returns:
#             dict of mean regression loss
#         """
#         if self.projection.do_regression:
#             # get prediction before the regression to compare to regressed policy
#             with ch.no_grad():
#                 p = self.policy(obs)
#                 p_proj = self.projection(self.policy, p, q, self._global_steps)
#                 # TODO: self.policy.squash()
#
#             if self.verbose >= 2:
#                 self.store.log_table_and_tb('constraints_initial',
#                                             self.projection.compute_metrics(self.policy, p, q, self._global_steps),
#                                             step=logging_step)
#                 self.store.log_table_and_tb('constraints_projection',
#                                             self.projection.compute_metrics(self.policy, p_proj, q, self._global_steps),
#                                             step=logging_step)
#                 self.store['constraints_initial'].flush_row()
#                 self.store['constraints_projection'].flush_row()
#
#         return self.projection.trust_region_regression(self.policy, obs, q, n_minibatches, self._global_steps)
#
#     def log_metrics(self, obs, q, logging_step):
#         """
#         Computes and logs the trust region metrics.
#         Args:
#             obs: observations used for evaluation
#             q: old distributions
#             logging_step: current logging step
#
#         Returns:
#             dict of trust region metrics
#         """
#         with ch.no_grad():
#             p = self.policy(obs)
#         metrics_dict = self.projection.compute_metrics(self.policy, p, q, self._global_steps)
#
#         if self.log_interval != 0 and self._global_steps % self.log_interval == 0:
#             self.store.log_table_and_tb('constraints', metrics_dict, step=logging_step)
#             self.store['constraints'].flush_row()
#
#             if self.verbose >= 3 and self._global_steps % (self.log_interval * 10) == 0:
#                 histo = self.projection.compute_metrics(self.policy, p, q, self._global_steps, aggregate=False)
#                 histo = {k: v.clip(0, self.projection.mean_bound * 3) for k, v in histo.items()}
#                 self.store.log_tb('constraint_distribution_pre', histo, step=logging_step, summary_type='histogram')
#                 proj_p = self.projection(self.policy, p, q, self._global_steps)
#                 histo = self.projection.compute_metrics(self.policy, proj_p, q, self._global_steps, aggregate=False)
#                 self.store.log_tb('constraint_distribution_post', histo, step=logging_step, summary_type='histogram')
#         # self.store.log_table_and_tb('distribution', {
#         #         'mean': get_numpy(p[0]),
#         #         'std': get_numpy(p[1]),
#         #     }, step=logging_step, summary_type="histogram")
#         #     self.store['distribution'].flush_row()
#
#         return metrics_dict
#
#     @abc.abstractmethod
#     def step(self):
#         pass
#
#     @abc.abstractmethod
#     def learn(self):
#         pass
#
#     @abc.abstractmethod
#     def save(self, iteration):
#         pass
#
#     @staticmethod
#     @abc.abstractmethod
#     def agent_from_data(store, train_steps=None):
#         pass
#
#     @staticmethod
#     @abc.abstractmethod
#     def agent_from_params(params, store=None):
#         pass
