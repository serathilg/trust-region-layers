import collections
import logging
from copy import deepcopy
from typing import Union

import fancy_gym
import gym
import numpy as np
import torch as ch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from trust_region_projections.models.policy.abstract_gaussian_policy import AbstractGaussianPolicy
from trust_region_projections.models.value.critic import BaseCritic
from trust_region_projections.models.value.vf_net import VFNet
from trust_region_projections.sampling.dataclass import TrajectoryOffPolicyLogpacsRaw, TrajectoryOffPolicyRaw, \
    TrajectoryOnPolicyRaw
from trust_region_projections.utils.torch_utils import get_numpy, tensorize, to_gpu


def make_env(env_id: str, seed: int, rank: int, disable_timelimit: bool = False, use_time_feature_wrapper: bool = False,
             wrap_monitor: bool = False, **kwargs) -> callable:
    """
    returns callable to create gym environment or monitor

    Args:
        env_id: gym env ID
        seed: seed for env
        rank: rank if multiple env are used
        wrap_monitor: Whether to use a Monitor for episode stats or not
        disable_timelimit: use gym env without artificial termination signal

    Returns: callable for env constructor

    """

    assert not (disable_timelimit and use_time_feature_wrapper), \
        "Cannot disable TimeLimit and use TimeFeatureWrapper at the same time. "

    def _get_env():
        replanning_interval = kwargs.pop('replanning_interval', -1)
        if replanning_interval > 0:
            kwargs.update({'black_box_kwargs': {
                'replanning_schedule': lambda pos, vel, obs, action, t: t % replanning_interval == 0}})
        env = fancy_gym.make_rank(env_id, seed=seed, rank=rank, return_callable=False, **kwargs)

        # Remove env from gym TimeLimitWrapper
        if disable_timelimit:
            env = env.env if isinstance(env, gym.wrappers.TimeLimit) else env
        elif use_time_feature_wrapper:
            # env = sb3_contrib.common.wrappers.TimeFeatureWrapper(env)
            raise NotImplementedError()

        # if log_dir is not None:
        #     import os
        #     env = gym.wrappers.Monitor(env=env, directory=os.path.join(log_dir, str(rank)))

        return Monitor(env) if wrap_monitor else env

    return _get_env


class TrajectorySampler(object):
    def __init__(self,
                 env_id: str,
                 n_envs: int = 1,
                 n_test_envs: int = 1,
                 eval_runs: int = 5,
                 discount_factor: float = 0.99,
                 norm_observations: Union[bool, None] = False,
                 clip_observations: Union[float, None] = 0.0,
                 norm_rewards: Union[bool, None] = False,
                 clip_rewards: Union[float, None] = 0.0,
                 scale_actions: bool = False,
                 disable_timelimit: bool = False,
                 # use_time_feature_wrapper: bool = False,
                 handle_timelimit: bool = True,
                 cpu: bool = True,
                 dtype=ch.float32,
                 seed: int = 1,
                 **kwargs
                 ):

        """
        Instance that takes care of generating Trajectory samples.
        Args:
           env_id: ID of training env
           n_envs: Number of parallel envs to run for more efficient sampling.
           n_test_envs: Number of environments to use during occasional testing of the current policy.
           discount_factor: Discount factor for return computation.
           norm_observations: If `True`, keeps moving mean and variance of observations and normalizes
                   incoming observations. Additional optimization proposed in (Ilyas et al., 2018).
           clip_observations: Value above and below to clip normalized observation.
                   Additional optimization proposed in (Ilyas et al., 2018) set to `5` or `10`.
           norm_rewards: If true, keeps moving variance of rewards and normalizes incoming rewards.
                   Reward normalization was implemented in OpenAI baselines.
           clip_rewards: Value above and below to clip normalized reward.
                       Additional optimization proposed in (Ilyas et al., 2018) set to `5` or `10`.
           dtype: Data type to use, either of float32 or float64. The later might be necessary for higher
                   dimensions in order to learn the full covariance.
           cpu: Compute on CPU only.
           seed: Seed for generating envs
        """

        self.dtype = dtype
        self.cpu = cpu

        self.total_rewards = collections.deque(maxlen=100)
        self.total_steps = collections.deque(maxlen=100)
        self.total_success_rate = collections.deque(maxlen=100)

        self.eval_runs = eval_runs
        self.scale_actions = scale_actions

        self.handle_time_limit = handle_timelimit

        clip_observations = clip_observations if clip_observations else np.inf
        clip_rewards = clip_rewards if clip_rewards else np.inf

        # When replanning, we generate more than on sample per step call, this allows to scale the logs accordingly
        self.sample_cost = kwargs.get('replanning_interval', 1)


        vec_env_fun = SubprocVecEnv if n_envs > 1 else DummyVecEnv
        self.env_fns = [make_env(env_id, seed, i, disable_timelimit, wrap_monitor=True, **kwargs) for i in range(n_envs)]
        envs = vec_env_fun(self.env_fns)
        # envs = VecMonitor(envs)
        self.envs = VecNormalize(envs, training=True,
                                 norm_obs=norm_observations, norm_reward=norm_rewards, clip_obs=clip_observations,
                                 clip_reward=clip_rewards, gamma=discount_factor)
        # reset once in the beginning to have non empty old observation
        self.envs.reset()

        vec_env_fun = SubprocVecEnv if n_test_envs > 1 else DummyVecEnv
        self.env_fns_test = [make_env(env_id, seed + n_envs, i, wrap_monitor=True, **kwargs) for i in range(n_envs)]
        envs_test = vec_env_fun(self.env_fns_test)
        # envs_test = VecMonitor(envs_test)
        self.envs_test = VecNormalize(envs_test, training=False, norm_obs=norm_observations, norm_reward=False,
                                      clip_obs=clip_observations, clip_reward=np.inf, gamma=discount_factor)

        self._logger = logging.getLogger("trajectory_sampler")

    def __call__(self, rollout_steps, policy: AbstractGaussianPolicy, critic: Union[None, VFNet, BaseCritic] = None,
                 reset_envs: bool = False, is_on_policy: bool = True, off_policy_logpacs: bool = False):
        self.run(rollout_steps, policy, critic, reset_envs, is_on_policy, off_policy_logpacs)

    def run(self, rollout_steps, policy: AbstractGaussianPolicy, critic: Union[None, VFNet, BaseCritic] = None,
            reset_envs: bool = False, is_on_policy: bool = True, off_policy_logpacs: bool = False,
            random_actions: bool = False) -> Union[
        TrajectoryOnPolicyRaw, TrajectoryOffPolicyRaw, TrajectoryOffPolicyLogpacsRaw]:
        """
        Args:
            rollout_steps: Number of rollouts to generate, total number of samples returned is n_envs x rollout_steps
            policy: Policy model to generate samples for
            critic: Value model instance for on_policy
            reset_envs: Whether to reset all envs in the beginning.
            is_on_policy: if True returns relevant trajectory information for on policy learning,
                          if False for off policy learning
            off_policy_logpacs: returns logpacs for the off policy setting, e.g. when using retrace

        Returns:
            NamedTuple Trajectory with the respective data as torch tensors.
        """

        needs_proj = False
        if isinstance(policy, tuple):
            needs_proj = True
            policy, old_policy, projection = policy

        # Here, we init the containers that will contain the experiences

        base_shape = (rollout_steps, self.num_envs)
        base_shape_p1 = (rollout_steps + 1, self.num_envs)
        base_action_shape = base_shape + self.envs.action_space.shape

        mb_obs = ch.zeros(base_shape_p1 + self.envs.observation_space.shape, dtype=self.dtype)
        mb_actions = ch.zeros(base_action_shape, dtype=self.dtype)
        mb_rewards = ch.zeros(base_shape, dtype=self.dtype)
        mb_dones = ch.zeros(base_shape, dtype=ch.bool)

        # proper timelimit handling
        mb_time_limit_dones = ch.zeros(base_shape, dtype=ch.bool)
        mb_terminal_obs = ch.zeros(base_shape_p1 + self.envs.observation_space.shape, dtype=self.dtype)

        ep_reward = []
        ep_length = []
        ep_success_rate = []

        if is_on_policy or off_policy_logpacs:
            mb_means = ch.zeros(base_action_shape, dtype=self.dtype)
            mb_stds = ch.zeros(base_action_shape + self.envs.action_space.shape, dtype=self.dtype)

        # continue from last state
        # Before first step we already have self.obs because env calls self.obs = env.reset() on init
        obs = self.envs.reset() if reset_envs else self.envs.normalize_obs(self.envs.get_original_obs())
        obs = tensorize(obs, self.cpu, self.dtype)

        # For n in range number of steps
        for i in range(rollout_steps):
            # Given observations, get action value and logpacs
            if random_actions:
                actions = tensorize([self.action_space.sample()], self.cpu, self.dtype)
            else:
                pds = policy(obs, train=False)
                if needs_proj:
                    # TODO fix 0 for entropy bound
                    pds = projection(policy, pds, old_policy(pds, train=False), 0)
                actions = policy.sample(pds)
                actions = policy.squash(actions)

            actions = self._maybe_scale_actions(actions)
            mb_obs[i] = obs
            mb_actions[i] = actions

            # maybe unscale squashed action back to original action space
            unscaled_action = self._maybe_unscale_actions(get_numpy(actions))
            obs, rewards, dones, infos = self.envs.step(unscaled_action)
            obs = tensorize(obs, self.cpu, self.dtype)

            if (is_on_policy or off_policy_logpacs) and not random_actions:
                mb_means[i] = pds[0]
                mb_stds[i] = pds[1]

            for j, info in enumerate(infos):
                # store correct terminal observation when terminating due to time limit
                if self.handle_time_limit:
                    mb_time_limit_dones[i, j] = info.get("TimeLimit.truncated", False)
                    if mb_time_limit_dones[i, j]:
                        mb_terminal_obs[i, j] = tensorize(info["terminal_observation"], self.cpu, self.dtype)
                ep_i = info.get("episode")
                if ep_i is not None:
                    ep_reward.append(ep_i['r'])
                    ep_length.append(ep_i['l'])
                    ep_success_rate.append([info.get('success', info.get('is_success', -1))])

            mb_rewards[i] = tensorize(rewards, self.cpu, self.dtype)
            mb_dones[i] = tensorize(dones, self.cpu, ch.bool)

        # save last value for vf/qf prediction of next state
        mb_obs[-1] = obs

        if is_on_policy:
            # compute all logpacs and value estimates at once --> less computation
            mb_logpacs = policy.log_probability((mb_means, mb_stds), mb_actions)
            mb_values = (critic if critic else policy.get_value)(mb_obs, train=False)

            out = (mb_obs[:-1], mb_actions, mb_logpacs, mb_rewards, mb_values,
                   mb_dones, mb_time_limit_dones, mb_terminal_obs, mb_means, mb_stds)
        else:
            out = (mb_obs, mb_actions, mb_rewards, mb_dones, mb_time_limit_dones, mb_terminal_obs)
            if off_policy_logpacs:
                mb_logpacs = policy.log_probability((mb_means, mb_stds), mb_actions)
                out += (mb_logpacs,)

        if not self.cpu:
            out = tuple(map(to_gpu, out))

        self.total_rewards.extend(ep_reward)
        self.total_steps.extend(ep_length)
        self.total_success_rate.extend(ep_success_rate)

        if is_on_policy:
            return TrajectoryOnPolicyRaw(*out)
        elif off_policy_logpacs:
            return TrajectoryOffPolicyLogpacsRaw(*out)
        else:
            return TrajectoryOffPolicyRaw(*out)

    def evaluate_policy(self, policy: AbstractGaussianPolicy, render: bool = False, deterministic: bool = True,
                        render_mode: str = "human"):

        needs_proj = False
        if isinstance(policy, tuple):
            needs_proj = True
            policy, old_policy, projection = policy

        if self.num_envs_test == 0:
            return {}

        ep_rewards = np.zeros((self.eval_runs, self.num_envs_test,))
        ep_lengths = np.zeros((self.eval_runs, self.num_envs_test,))
        ep_success_rates = np.ones((self.eval_runs, self.num_envs_test)) * -1

        # copy normalization stats from training env to test env
        try:
            self.envs_test.obs_rms = deepcopy(self.envs.obs_rms)
            self.envs_test.ret_rms = deepcopy(self.envs.ret_rms)
        except AttributeError:
            # Obs normalization not used
            pass

        for i in range(self.eval_runs):
            # info_print = collections.defaultdict(lambda: np.zeros(1, ))
            not_dones = np.ones((self.num_envs_test,), bool)
            obs = self.envs_test.reset()
            while np.any(not_dones):
                # ep_lengths[i, not_dones] += 1
                if render:
                    img = self.envs_test.render(mode=render_mode)
                    # if not hasattr(self, "base_img"):
                    #     from matplotlib import pyplot as plt
                    #     plt.ion()
                    #     self.fig = plt.figure()
                    #     ax = self.fig.add_subplot(1, 1, 1)
                    #     # ax.set_title(infos[0])
                    #     self.base_img = ax.imshow(img)
                    #     self.fig.show()
                    #
                    # self.base_img.set_data(img)
                    # self.fig.canvas.draw()
                    # self.fig.canvas.flush_events()
                with ch.no_grad():
                    obs = tensorize(obs, self.cpu, self.dtype)
                    p = policy(obs)
                    if needs_proj:
                        # TODO fix 0 for entropy bound
                        p = projection(policy, p, old_policy(obs), 0)
                    actions = p[0] if deterministic else policy.sample(p)
                    actions = policy.squash(actions)
                unscaled_action = self._maybe_unscale_actions(get_numpy(actions))
                obs, rews, dones, infos = self.envs_test.step(unscaled_action)
                # ep_rewards[i, not_dones] += rews[not_dones]

                # for k, v in infos[0].items():
                #     if k not in ['episode', 'terminal_observation']:
                #         info_print[k] += v

                # check whether there is a new termination, the global not_dones is updated afterwards
                newly_done = dones & not_dones
                if np.any(newly_done):
                    episode = [info.get('episode') for info in infos]
                    ep_rewards[i, newly_done] = np.array([elem.get('r') for elem in episode])[newly_done]
                    ep_lengths[i, newly_done] = np.array([elem.get('l') for elem in episode])[newly_done]
                    rate = []
                    for elem in infos:
                        sr = elem.get('success', elem.get('is_success', -1))
                        rate.append(sr[-1] if isinstance(sr, list) else sr)

                    ep_success_rates[i, newly_done] = np.array(rate)[newly_done]

                # only set to False when env has never terminated before
                # Otherwise, we introduce bias for shorter/worse sampling
                not_dones = ~dones & not_dones
            # print(info_print)

        return self.get_performance_dict(ep_rewards, ep_lengths), self.get_performance_dict(ep_success_rates)

    def _maybe_unscale_actions(self, actions):
        """
        Scale actions from [-1,1] back to original action space.
        """
        lb = self.action_space.low
        ub = self.action_space.high
        if self.scale_actions:
            # working with actions that are scaled -1 to 1
            actions = lb + (actions + 1.) * 0.5 * (ub - lb)
        return np.clip(actions, lb, ub)

    def _maybe_scale_actions(self, actions: ch.Tensor):
        """
        Scale actions from original action space to [-1,1].
        """
        lb = actions.new(self.action_space.low)
        ub = actions.new(self.action_space.high)
        if self.scale_actions:
            actions = 2.0 * ((actions - lb) / (ub - lb)) - 1.0
            actions = actions.clip(-1, 1)
        return actions

    def get_exploration_performance(self):
        ep_reward = np.array(self.total_rewards)
        ep_length = np.array(self.total_steps)
        ep_success_rate = np.array(self.total_success_rate)
        return self.get_performance_dict(ep_reward, ep_length), self.get_performance_dict(ep_success_rate, None)

    @staticmethod
    def get_performance_dict(ep_performance, ep_length=None):
        stats = {
            'mean': ep_performance.mean().item(),
            'median': np.median(ep_performance).item(),
            'std': ep_performance.std().item(),
            'max': ep_performance.max().item(),
            'min': ep_performance.min().item(),
        }
        if ep_length is not None:
            stats.update({
                'step_reward': (ep_performance / ep_length).mean().item(),
                'length': ep_length.mean().item(),
                'length_std': ep_length.std().item(),
            })
        return stats

    @property
    def observation_space(self):
        return self.envs.observation_space

    @property
    def observation_shape(self):
        return self.observation_space.shape

    @property
    def action_space(self):
        return self.envs.action_space

    @property
    def action_shape(self):
        return self.action_space.shape

    @property
    def num_envs(self):
        return self.envs.num_envs

    @property
    def num_envs_test(self):
        return self.envs_test.num_envs

    @property
    def spec(self):
        return self.envs.unwrapped.envs[0].spec

#
# import collections
# import logging
# from copy import deepcopy
# from typing import Union
#
# import fancy_gym
# import gym
# import numpy as np
# import torch as ch
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
#
# from trust_region_projections.models.policy.abstract_gaussian_policy import AbstractGaussianPolicy
# from trust_region_projections.models.value.critic import BaseCritic
# from trust_region_projections.models.value.vf_net import VFNet
# from trust_region_projections.sampling.dataclass import TrajectoryOffPolicyLogpacsRaw, TrajectoryOffPolicyRaw, \
#     TrajectoryOnPolicyRaw
# from trust_region_projections.utils.torch_utils import get_numpy, tensorize, to_gpu
#
#
# def make_env(env_id: str, seed: int, rank: int, disable_timelimit: bool = False, use_time_feature_wrapper: bool = False,
#              wrap_monitor: bool = False, **kwargs) -> callable:
#     """
#     returns callable to create gym environment or monitor
#
#     Args:
#         env_id: gym env ID
#         seed: seed for env
#         rank: rank if multiple env are used
#         wrap_monitor: Whether to use a Monitor for episode stats or not
#         disable_timelimit: use gym env without artificial termination signal
#
#     Returns: callable for env constructor
#
#     """
#
#     assert not (disable_timelimit and use_time_feature_wrapper), \
#         "Cannot disable TimeLimit and use TimeFeatureWrapper at the same time. "
#
#     def _get_env():
#         env = fancy_gym.make_rank(env_id, seed=seed, rank=rank, return_callable=False, **kwargs)
#
#         # Remove env from gym TimeLimitWrapper
#         if disable_timelimit:
#             env = env.env if isinstance(env, gym.wrappers.TimeLimit) else env
#         elif use_time_feature_wrapper:
#             # env = sb3_contrib.common.wrappers.TimeFeatureWrapper(env)
#             raise NotImplementedError()
#
#         # if log_dir is not None:
#         #     import os
#         #     env = gym.wrappers.Monitor(env=env, directory=os.path.join(log_dir, str(rank)))
#
#         return Monitor(env) if wrap_monitor else env
#
#     return _get_env
#
#
# class TrajectorySampler(object):
#     def __init__(self,
#                  env_id: str,
#                  n_envs: int = 1,
#                  n_test_envs: int = 1,
#                  eval_runs: int = 5,
#                  discount_factor: float = 0.99,
#                  norm_observations: Union[bool, None] = False,
#                  clip_observations: Union[float, None] = 0.0,
#                  norm_rewards: Union[bool, None] = False,
#                  clip_rewards: Union[float, None] = 0.0,
#                  scale_actions: bool = False,
#                  disable_timelimit: bool = False,
#                  # use_time_feature_wrapper: bool = False,
#                  handle_timelimit: bool = True,
#                  cpu: bool = True,
#                  dtype=ch.float32,
#                  seed: int = 1, **kwargs):
#
#         """
#         Instance that takes care of generating Trajectory samples.
#         Args:
#            env_id: ID of training env
#            n_envs: Number of parallel envs to run for more efficient sampling.
#            n_test_envs: Number of environments to use during occasional testing of the current policy.
#            discount_factor: Discount factor for return computation.
#            norm_observations: If `True`, keeps moving mean and variance of observations and normalizes
#                    incoming observations. Additional optimization proposed in (Ilyas et al., 2018).
#            clip_observations: Value above and below to clip normalized observation.
#                    Additional optimization proposed in (Ilyas et al., 2018) set to `5` or `10`.
#            norm_rewards: If true, keeps moving variance of rewards and normalizes incoming rewards.
#                    Reward normalization was implemented in OpenAI baselines.
#            clip_rewards: Value above and below to clip normalized reward.
#                        Additional optimization proposed in (Ilyas et al., 2018) set to `5` or `10`.
#            dtype: Data type to use, either of float32 or float64. The later might be necessary for higher
#                    dimensions in order to learn the full covariance.
#            cpu: Compute on CPU only.
#            seed: Seed for generating envs
#         """
#
#         self.dtype = dtype
#         self.cpu = cpu
#
#         self.total_rewards = collections.deque(maxlen=100)
#         self.total_steps = collections.deque(maxlen=100)
#         self.total_success_rate = collections.deque(maxlen=100)
#
#         self.eval_runs = eval_runs
#         self.scale_actions = scale_actions
#
#         self.handle_time_limit = handle_timelimit
#
#         clip_observations = clip_observations if clip_observations else np.inf
#         clip_rewards = clip_rewards if clip_rewards else np.inf
#         self.env_id = env_id
#         if env_id == 'ALRHopperJump-v4':
#             env_id_train = 'ALRHopperJump-v4'
#             env_kwargs = {'height_scale': kwargs['height_scale'], 'dist_scale': kwargs['dist_scale'],
#                           'healthy_scale': kwargs['healthy_scale']}
#             env_id_test = 'ALRHopperJump-v3'
#         else:
#             env_id_train = env_id
#             env_id_test = env_id
#             env_kwargs = {}
#
#         vec_env_fun = SubprocVecEnv if n_envs > 1 else DummyVecEnv
#         self.env_fns = [make_env(env_id_train, seed, i, disable_timelimit, wrap_monitor=True, **env_kwargs) for i in range(n_envs)]
#         envs = vec_env_fun(self.env_fns)
#         # envs = VecMonitor(envs)
#         self.envs = VecNormalize(envs, training=True,
#                                  norm_obs=norm_observations, norm_reward=norm_rewards, clip_obs=clip_observations,
#                                  clip_reward=clip_rewards, gamma=discount_factor)
#         # reset once in the beginning to have non empty old observation
#         self.envs.reset()
#
#         vec_env_fun = SubprocVecEnv if n_test_envs > 1 else DummyVecEnv
#         self.env_fns_test = [make_env(env_id_test, seed + n_envs, i, wrap_monitor=True) for i in range(n_envs)]
#         envs_test = vec_env_fun(self.env_fns_test)
#         # envs_test = VecMonitor(envs_test)
#         self.envs_test = VecNormalize(envs_test, training=False, norm_obs=norm_observations, norm_reward=False,
#                                       clip_obs=clip_observations, clip_reward=np.inf, gamma=discount_factor)
#
#         self._logger = logging.getLogger("trajectory_sampler")
#
#     def __call__(self, rollout_steps, policy: AbstractGaussianPolicy, critic: Union[None, VFNet, BaseCritic] = None,
#                  reset_envs: bool = False, is_on_policy: bool = True, off_policy_logpacs: bool = False):
#         self.run(rollout_steps, policy, critic, reset_envs, is_on_policy, off_policy_logpacs)
#
#     def run(self, rollout_steps, policy: AbstractGaussianPolicy, critic: Union[None, VFNet, BaseCritic] = None,
#             reset_envs: bool = False, is_on_policy: bool = True, off_policy_logpacs: bool = False,
#             random_actions: bool = False) -> Union[
#         TrajectoryOnPolicyRaw, TrajectoryOffPolicyRaw, TrajectoryOffPolicyLogpacsRaw]:
#         """
#         Args:
#             rollout_steps: Number of rollouts to generate, total number of samples returned is n_envs x rollout_steps
#             policy: Policy model to generate samples for
#             critic: Value model instance for on_policy
#             reset_envs: Whether to reset all envs in the beginning.
#             is_on_policy: if True returns relevant trajectory information for on policy learning,
#                           if False for off policy learning
#             off_policy_logpacs: returns logpacs for the off policy setting, e.g. when using retrace
#
#         Returns:
#             NamedTuple Trajectory with the respective data as torch tensors.
#         """
#
#         needs_proj = False
#         if isinstance(policy, tuple):
#             needs_proj = True
#             policy, old_policy, projection = policy
#
#         # Here, we init the containers that will contain the experiences
#
#         base_shape = (rollout_steps, self.num_envs)
#         base_shape_p1 = (rollout_steps + 1, self.num_envs)
#         base_action_shape = base_shape + self.envs.action_space.shape
#
#         mb_obs = ch.zeros(base_shape_p1 + self.envs.observation_space.shape, dtype=self.dtype)
#         mb_actions = ch.zeros(base_action_shape, dtype=self.dtype)
#         mb_rewards = ch.zeros(base_shape, dtype=self.dtype)
#         mb_dones = ch.zeros(base_shape, dtype=ch.bool)
#
#         # proper timelimit handling
#         mb_time_limit_dones = ch.zeros(base_shape, dtype=ch.bool)
#         mb_terminal_obs = ch.zeros(base_shape_p1 + self.envs.observation_space.shape, dtype=self.dtype)
#
#         ep_reward = []
#         ep_length = []
#         ep_success_rate = []
#
#         if is_on_policy or off_policy_logpacs:
#             mb_means = ch.zeros(base_action_shape, dtype=self.dtype)
#             mb_stds = ch.zeros(base_action_shape + self.envs.action_space.shape, dtype=self.dtype)
#
#         # continue from last state
#         # Before first step we already have self.obs because env calls self.obs = env.reset() on init
#         obs = self.envs.reset() if reset_envs else self.envs.normalize_obs(self.envs.get_original_obs())
#         obs = tensorize(obs, self.cpu, self.dtype)
#
#         # For n in range number of steps
#         for i in range(rollout_steps):
#             # Given observations, get action value and logpacs
#             if random_actions:
#                 actions = tensorize([self.action_space.sample()], self.cpu, self.dtype)
#             else:
#                 pds = policy(obs, train=False)
#                 if needs_proj:
#                     # TODO fix 0 for entropy bound
#                     pds = projection(policy, pds, old_policy(pds, train=False), 0)
#                 actions = policy.sample(pds)
#                 actions = policy.squash(actions)
#
#             actions = self._maybe_scale_actions(actions)
#             mb_obs[i] = obs
#             mb_actions[i] = actions
#
#             # maybe unscale squashed action back to original action space
#             unscaled_action = self._maybe_unscale_actions(get_numpy(actions))
#             obs, rewards, dones, infos = self.envs.step(unscaled_action)
#             obs = tensorize(obs, self.cpu, self.dtype)
#
#             if (is_on_policy or off_policy_logpacs) and not random_actions:
#                 mb_means[i] = pds[0]
#                 mb_stds[i] = pds[1]
#
#             for j, info in enumerate(infos):
#                 # store correct terminal observation when terminating due to time limit
#                 if self.handle_time_limit:
#                     mb_time_limit_dones[i, j] = info.get("TimeLimit.truncated", False)
#                     if mb_time_limit_dones[i, j]:
#                         mb_terminal_obs[i, j] = tensorize(info["terminal_observation"], self.cpu, self.dtype)
#                 ep_i = info.get("episode")
#                 if ep_i is not None:
#                     ep_reward.append(ep_i['r'])
#                     ep_length.append(ep_i['l'])
#                     ep_success_rate.append([info.get('success', info.get('is_success', -1))])
#
#             mb_rewards[i] = tensorize(rewards, self.cpu, self.dtype)
#             mb_dones[i] = tensorize(dones, self.cpu, ch.bool)
#
#         # save last value for vf/qf prediction of next state
#         mb_obs[-1] = obs
#
#         if is_on_policy:
#             # compute all logpacs and value estimates at once --> less computation
#             mb_logpacs = policy.log_probability((mb_means, mb_stds), mb_actions)
#             mb_values = (critic if critic else policy.get_value)(mb_obs, train=False)
#
#             out = (mb_obs[:-1], mb_actions, mb_logpacs, mb_rewards, mb_values,
#                    mb_dones, mb_time_limit_dones, mb_terminal_obs, mb_means, mb_stds)
#         else:
#             out = (mb_obs, mb_actions, mb_rewards, mb_dones, mb_time_limit_dones, mb_terminal_obs)
#             if off_policy_logpacs:
#                 mb_logpacs = policy.log_probability((mb_means, mb_stds), mb_actions)
#                 out += (mb_logpacs,)
#
#         if not self.cpu:
#             out = tuple(map(to_gpu, out))
#
#         self.total_rewards.extend(ep_reward)
#         self.total_steps.extend(ep_length)
#         self.total_success_rate.extend(ep_success_rate)
#
#         if is_on_policy:
#             return TrajectoryOnPolicyRaw(*out)
#         elif off_policy_logpacs:
#             return TrajectoryOffPolicyLogpacsRaw(*out)
#         else:
#             return TrajectoryOffPolicyRaw(*out)
#
#     def evaluate_policy(self, policy: AbstractGaussianPolicy, render: bool = False, deterministic: bool = True,
#                         render_mode: str = "human"):
#
#         needs_proj = False
#         if isinstance(policy, tuple):
#             needs_proj = True
#             policy, old_policy, projection = policy
#
#         if self.num_envs_test == 0:
#             return {}
#
#         ep_rewards = np.zeros((self.eval_runs, self.num_envs_test,))
#         ep_lengths = np.zeros((self.eval_runs, self.num_envs_test,))
#         ep_success_rates = np.ones((self.eval_runs, self.num_envs_test)) * -1
#         ep_max_height = np.ones((self.eval_runs, self.num_envs_test)) * -1
#         ep_goal_dists = np.ones((self.eval_runs, self.num_envs_test)) * -1
#         ep_is_healthy = np.ones((self.eval_runs, self.num_envs_test)) * -1
#         ep_contact_dists = np.ones((self.eval_runs, self.num_envs_test)) * -1
#
#         # copy normalization stats from training env to test env
#         try:
#             self.envs_test.obs_rms = deepcopy(self.envs.obs_rms)
#             self.envs_test.ret_rms = deepcopy(self.envs.ret_rms)
#         except AttributeError:
#             # Obs normalization not used
#             pass
#         # if render:
#         #     self.envs_test.render()
#             # try:
#             #     # self.env_test.envs[0].render(mode="rgb_array")
#             #     self.envs_test.envs[0].render()
#             # except AttributeError:
#             #     from gym.vector import SyncVectorEnv
#             #     self.envs_test = SyncVectorEnv(self.envs_test.env_fns)
#             #     self.envs_test.envs[0].render()
#         for i in range(self.eval_runs):
#             # info_print = collections.defaultdict(lambda: np.zeros(1, ))
#             not_dones = np.ones((self.num_envs_test,), np.bool)
#             obs = self.envs_test.reset()
#             while np.any(not_dones):
#                 # ep_lengths[i, not_dones] += 1
#                 # if render:
#                 #     img = self.envs_test.render(mode=render_mode)
#                 #     # if not hasattr(self, "base_img"):
#                 #     #     from matplotlib import pyplot as plt
#                 #     #     plt.ion()
#                 #     #     self.fig = plt.figure()
#                 #     #     ax = self.fig.add_subplot(1, 1, 1)
#                 #     #     # ax.set_title(infos[0])
#                 #     #     self.base_img = ax.imshow(img)
#                 #     #     self.fig.show()
#                 #     #
#                 #     # self.base_img.set_data(img)
#                 #     # self.fig.canvas.draw()
#                 #     # self.fig.canvas.flush_events()
#                 with ch.no_grad():
#                     obs = tensorize(obs, self.cpu, self.dtype)
#                     p = policy(obs)
#                     if needs_proj:
#                         # TODO fix 0 for entropy bound
#                         p = projection(policy, p, old_policy(obs), 0)
#                     actions = p[0] if deterministic else policy.sample(p)
#                     actions = policy.squash(actions)
#                 unscaled_action = self._maybe_unscale_actions(get_numpy(actions))
#                 obs, rews, dones, infos = self.envs_test.step(unscaled_action)
#                 if render:
#                     self.envs_test.render()
#                 # ep_rewards[i, not_dones] += rews[not_dones]
#
#                 # for k, v in infos[0].items():
#                 #     if k not in ['episode', 'terminal_observation']:
#                 #         info_print[k] += v
#
#                 # check whether there is a new termination, the global not_dones is updated afterwards
#                 newly_done = dones & not_dones
#                 if np.any(newly_done):
#                     episode = [info.get('episode') for info in infos]
#                     ep_rewards[i, newly_done] = np.array([elem.get('r') for elem in episode])[newly_done]
#                     ep_lengths[i, newly_done] = np.array([elem.get('l') for elem in episode])[newly_done]
#                     rate = np.array([elem.get('success', elem.get('is_success', -1)) for elem in infos])
#                     ep_success_rates[i, newly_done] = np.array(rate)[newly_done]
#                     max_height = np.array([elem.get('max_height', -1) for elem in infos])
#                     ep_max_height[i, newly_done] = np.array(max_height)[newly_done]
#                     goal_dist = np.array([elem.get('goal_dist', -1) for elem in infos])
#                     ep_goal_dists[i, newly_done] = np.array(goal_dist)[newly_done]
#                     healthy = np.array([elem.get('healthy', -1) for elem in infos])
#                     ep_is_healthy[i, newly_done] = np.array(healthy)[newly_done]
#                     contact_dist = np.array([elem.get('contact_dist', -1) for elem in infos])
#                     ep_contact_dists[i, newly_done] = np.array(contact_dist)[newly_done]
#                 # only set to False when env has never terminated before
#                 # Otherwise, we introduce bias for shorter/worse sampling
#                 not_dones = ~dones & not_dones
#             # print(info_print)
#
#         return self.get_performance_dict(ep_rewards, ep_lengths), self.get_performance_dict(ep_success_rates), \
#                self.get_performance_dict(ep_max_height), self.get_performance_dict(ep_goal_dists),  \
#                self.get_performance_dict(ep_is_healthy), self.get_performance_dict(ep_contact_dists)
#
#     def _maybe_unscale_actions(self, actions):
#         """
#         Scale actions from [-1,1] back to original action space.
#         """
#         lb = self.action_space.low
#         ub = self.action_space.high
#         if self.scale_actions:
#             # working with actions that are scaled -1 to 1
#             actions = lb + (actions + 1.) * 0.5 * (ub - lb)
#         return np.clip(actions, lb, ub)
#
#     def _maybe_scale_actions(self, actions: ch.Tensor):
#         """
#         Scale actions from original action space to [-1,1].
#         """
#         lb = actions.new(self.action_space.low)
#         ub = actions.new(self.action_space.high)
#         if self.scale_actions:
#             actions = 2.0 * ((actions - lb) / (ub - lb)) - 1.0
#             actions = actions.clip(-1, 1)
#         return actions
#
#     def get_exploration_performance(self):
#         ep_reward = np.array(self.total_rewards)
#         ep_length = np.array(self.total_steps)
#         ep_success_rate = np.array(self.total_success_rate)
#         return self.get_performance_dict(ep_reward, ep_length), self.get_performance_dict(ep_success_rate, None)
#
#     @staticmethod
#     def get_performance_dict(ep_performance, ep_length=None):
#         stats = {
#             'mean': ep_performance.mean().item(),
#             'median': np.median(ep_performance).item(),
#             'std': ep_performance.std().item(),
#             'max': ep_performance.max().item(),
#             'min': ep_performance.min().item(),
#         }
#         if ep_length is not None:
#             stats.update({
#                 'step_reward': (ep_performance / ep_length).mean().item(),
#                 'length': ep_length.mean().item(),
#                 'length_std': ep_length.std().item(),
#             })
#         return stats
#
#     @property
#     def observation_space(self):
#         return self.envs.observation_space
#
#     @property
#     def observation_shape(self):
#         return self.observation_space.shape
#
#     @property
#     def action_space(self):
#         return self.envs.action_space
#
#     @property
#     def action_shape(self):
#         return self.action_space.shape
#
#     @property
#     def num_envs(self):
#         return self.envs.num_envs
#
#     @property
#     def num_envs_test(self):
#         return self.envs_test.num_envs
#
#     @property
#     def spec(self):
#         return self.envs.unwrapped.envs[0].spec
