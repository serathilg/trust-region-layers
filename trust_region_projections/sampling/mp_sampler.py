import collections
import logging
from copy import deepcopy
from typing import Union

import fancy_gym
import gym
import numpy as np
import torch as ch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from trust_region_projections.models.policy.abstract_gaussian_policy import AbstractGaussianPolicy
from trust_region_projections.models.policy.pro_mp_wrapper import ProMPWrapper
from trust_region_projections.models.value.critic import BaseCritic
from trust_region_projections.models.value.vf_net import VFNet
from trust_region_projections.sampling.dataclass import TrajectoryOffPolicyLogpacsRaw, TrajectoryOffPolicyRaw, \
    TrajectoryOnPolicyRaw
from trust_region_projections.utils.torch_utils import get_numpy, tensorize, to_gpu


def make_env(env_id: str, seed: int, rank: int, disable_timelimit: bool = False, use_time_feature_wrapper: bool = False,
             wrap_monitor: bool = False) -> callable:
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
        env = fancy_gym.make_rank(env_id, seed=seed, rank=rank, return_callable=False)

        # Remove env from gym TimeLimitWrapper
        if disable_timelimit:
            raise DeprecationWarning("Disabling the timelimit is no longer supported.")
            # env = env.env if isinstance(env, gym.wrappers.TimeLimit) else env
        elif use_time_feature_wrapper:
            env = gym.wrappers.TimeAwareObservation(env)

        # if log_dir is not None:
        #     import os
        #     env = gym.wrappers.Monitor(env=env, directory=os.path.join(log_dir, str(rank)))

        return Monitor(env) if wrap_monitor else env

    return _get_env


class MPSampler(object):
    def __init__(self, env_id: str, n_envs: int = 1, n_test_envs: int = 1, eval_runs: int = 5,
                 discount_factor: float = 0.99, norm_observations: Union[bool, None] = False,
                 clip_observations: Union[float, None] = 0.0, norm_rewards: Union[bool, None] = False,
                 clip_rewards: Union[float, None] = 0.0, scale_actions: bool = False, disable_timelimit: bool = False,
                 use_time_feature_wrapper: bool = False, cpu: bool = True, dtype=ch.float32, seed: int = 1):

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

        clip_observations = clip_observations if clip_observations else np.inf
        clip_rewards = clip_rewards if clip_rewards else np.inf

        assert n_envs == 1
        self.env_fns = [make_env(env_id, seed, i, disable_timelimit, wrap_monitor=True,
                                 use_time_feature_wrapper=use_time_feature_wrapper) for i in range(n_envs)]
        envs = DummyVecEnv(self.env_fns)
        self.envs = VecNormalize(envs, training=True,
                                 norm_obs=norm_observations, norm_reward=norm_rewards, clip_obs=clip_observations,
                                 clip_reward=clip_rewards, gamma=discount_factor)
        # reset once in the beginning to have non empty old observation
        self.envs.reset()

        assert n_test_envs == 1
        self.env_fns_test = [make_env(env_id, seed + n_envs, i, wrap_monitor=True,
                                      use_time_feature_wrapper=use_time_feature_wrapper) for i in range(n_test_envs)]
        envs_test = DummyVecEnv(self.env_fns_test)
        self.envs_test = VecNormalize(envs_test, training=False, norm_obs=norm_observations, norm_reward=False,
                                      clip_obs=clip_observations, clip_reward=np.inf, gamma=discount_factor)

        self.action_scale = self.envs.envs[0].action_scale

        self._logger = logging.getLogger("trajectory_sampler")

    def __call__(self, rollout_steps, policy: ProMPWrapper, critic: Union[None, VFNet, BaseCritic] = None,
                 reset_envs: bool = False, is_on_policy: bool = True, off_policy_logpacs: bool = False):
        self.run(rollout_steps, policy, critic, reset_envs, is_on_policy, off_policy_logpacs)

    def run(self, rollout_steps, policy: ProMPWrapper, critic: Union[None, VFNet, BaseCritic] = None,
            reset_envs: bool = False, is_on_policy: bool = True, off_policy_logpacs: bool = False) -> Union[
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

        # Here, we init the containers that will contain the experiences

        base_shape = (rollout_steps,)
        base_shape_p1 = (rollout_steps + 1,)
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

        done = True
        obs = self.envs.reset()
        obs = tensorize(obs, self.cpu, self.dtype)
        des_pos = des_vel = stds = start_pos = None

        i = 0
        # For n in range number of steps
        while i < rollout_steps:
            if done:
                # Given observations, get pro mp parametrization and corresponding trajectory
                # TODO: use masked obs here?
                # masked_obs = obs[self.envs.active_obs]

                r_close = self.envs.envs[0].data.get_joint_qpos("r_close")
                # start_pos = obs.new(
                #     np.hstack([self.envs.envs[0].data.mocap_pos.flatten() / self.action_scale, r_close]))
                # start_pos = obs.new(np.hstack([self.envs.envs[0].data.mocap_pos.flatten(), r_close]))
                start_pos = 0
                des_pos, des_vel, stds = policy(obs, train=False, start_pos=start_pos)

                done = False

            for pos, vel, std in zip(des_pos[0], des_vel[0], stds[0]):
                mb_obs[i] = obs
                action = policy.sample((pos, std))
                # cur_pos = des_pos.new(self.envs.envs[0].data.mocap_pos.flatten() / self.action_scale)
                # cur_pos = des_pos.new(self.envs.envs[0].data.mocap_pos.flatten())
                # controller_action = policy.act(action, vel, cur_pos, cur_vel=None)[None]
                controller_action = action
                mb_actions[i] = action - start_pos
                # mb_actions[i] = pos

                # maybe unscale squashed action back to original action space
                unscaled_action = self._maybe_unscale_actions(get_numpy(controller_action))
                obs, reward, done, info = self.envs.step([unscaled_action])
                obs = tensorize(obs, self.cpu, self.dtype)

                if is_on_policy or off_policy_logpacs:
                    # mb_means[i] = mean
                    mb_means[i] = pos - start_pos
                    mb_stds[i] = std

                for info in info:
                    # store correct terminal observation when terminating due to time limit
                    mb_time_limit_dones[i] = info.get("TimeLimit.truncated", False)
                    if mb_time_limit_dones[i]:
                        mb_terminal_obs[i] = tensorize(info["terminal_observation"], self.cpu, self.dtype)
                    ep_i = info.get("episode")
                    if ep_i is not None:
                        ep_reward.append(ep_i['r'])
                        ep_length.append(ep_i['l'])
                        ep_success_rate.append([info.get('success', info.get('is_success', -1))])

                mb_rewards[i] = tensorize(reward, self.cpu, self.dtype)
                mb_dones[i] = tensorize(done, self.cpu, ch.bool)

                i += 1
                if i == rollout_steps:
                    break

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
        if self.num_envs_test == 0:
            return {}

        ep_rewards = np.zeros((self.eval_runs,))
        ep_lengths = np.zeros((self.eval_runs,))
        ep_success_rates = np.ones((self.eval_runs,)) * -1

        # copy normalization stats from training policy to test policy
        self.envs_test.obs_rms = deepcopy(self.envs.obs_rms)
        self.envs_test.ret_rms = deepcopy(self.envs.ret_rms)

        for i in range(self.eval_runs):
            obs = tensorize(self.envs_test.reset(), self.cpu, self.dtype)

            # Given observations, get pro mp parametrization and corresponding trajectory
            # TODO: use masked obs here?
            # masked_obs = obs[self.envs.active_obs]

            with ch.no_grad():
                r_close = self.envs_test.envs[0].data.get_joint_qpos("r_close")
                # start_pos = obs.new(
                #     np.hstack([self.envs_test.envs[0].data.mocap_pos.flatten() / self.action_scale, r_close]))
                # start_pos = obs.new(np.hstack([self.envs_test.envs[0].data.mocap_pos.flatten(), r_close]))
                start_pos = 0
                des_pos, des_vel, stds = policy(obs, train=False, start_pos=start_pos)

            for pos, vel, std in zip(des_pos[0], des_vel[0], stds[0]):
                if render:
                    self.envs_test.render(mode=render_mode)

                action = pos if deterministic else policy.sample((pos, std))
                # cur_pos = des_pos.new(self.envs_test.envs[0].data.mocap_pos.flatten()) / self.action_scale
                # cur_pos = action.new(self.envs_test.envs[0].data.mocap_pos.flatten())
                # action = policy.act(action, vel, cur_pos, cur_vel=None)[None]

                unscaled_action = self._maybe_unscale_actions(get_numpy(action))
                _, _, done, info = self.envs_test.step([unscaled_action])

                if done[0]:
                    episode = info[0].get('episode')
                    ep_rewards[i] = episode.get('r')
                    ep_lengths[i] = episode.get('l')
                    ep_success_rates[i] = info[0].get('success', info[0].get('is_success', -1))
                    break

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

    def get_exploration_performance(self):
        ep_reward = np.array(self.total_rewards)
        ep_length = np.array(self.total_steps)
        ep_success_rate = np.array(self.total_success_rate)
        return self.get_performance_dict(ep_reward, ep_length), self.get_performance_dict(ep_success_rate, None)

    @staticmethod
    def get_performance_dict(ep_performance, ep_length=None):
        print(ep_performance)
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
