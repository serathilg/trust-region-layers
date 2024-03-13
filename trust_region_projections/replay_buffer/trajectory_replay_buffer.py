from collections import deque

import numpy as np
import torch as ch

from trust_region_projections.replay_buffer.abstract_replay_buffer import AbstractReplayBuffer
from trust_region_projections.sampling.dataclass import TrajectoryOffPolicyLogpacs, TrajectoryOffPolicyLogpacsRaw
from trust_region_projections.utils.torch_utils import flatten_batch


class TrajectoryReplayBuffer(AbstractReplayBuffer):

    def __init__(self, n_step: int, max_replay_buffer_size: int, observation_dim: int, action_dim: int,
                 discount_factor: float, period_length: int = 1, handle_timelimit: bool = False,
                 dtype: ch.dtype = ch.float32, device: ch.device = "cpu"):
        """
        Trajectory replay buffer for Retrace and V-trace.

        Args:
            n_step: partial trajectory length stored
            period_length: sliding windows step size, i.e. period_length < n_step generates overlapping entries
        """

        super().__init__(max_replay_buffer_size, observation_dim, action_dim, discount_factor, handle_timelimit)
        self.n_step = n_step
        self.period_length = period_length
        assert self.n_step >= self.period_length, \
            f"The number of steps per window are smaller than the stride ({self.n_step} < {self.period_length})"
        self._observation_dim = observation_dim
        self._action_dim = action_dim

        # also save first observation  the others are included
        self._obs = ch.zeros((max_replay_buffer_size, n_step + 1, observation_dim), dtype=dtype, device=device)
        # Need future action for Q(s_tp1, a_tp1)
        self._actions = ch.zeros((max_replay_buffer_size, n_step + 1, action_dim), dtype=dtype, device=device)
        self._rewards = ch.zeros((max_replay_buffer_size, n_step), dtype=dtype, device=device)
        self._terminals = ch.zeros((max_replay_buffer_size, n_step), dtype=ch.bool, device=device)
        self._logpacs = ch.zeros((max_replay_buffer_size, n_step), dtype=dtype, device=device)
        self._time_limit_dones = ch.zeros((max_replay_buffer_size, n_step), dtype=ch.bool, device=device)
        self._terminal_obs = ch.zeros((max_replay_buffer_size, n_step + 1, observation_dim), dtype=dtype, device=device)

        # save last samples that have not been added as initiating sample to the buffer
        self._buffer = None  # TrajectoryOffPolicyLogpacsRaw(*[deque(maxlen=n_step - 1) for _ in range(7)])

    def add_samples(self, samples: TrajectoryOffPolicyLogpacs):
        """
        Add n new samples to replay buffer.
        Args:
            samples: Namedtuple of samples to add

        Returns:

        """

        # append existing samples from buffer to new samples
        if self._buffer:
            samples = TrajectoryOffPolicyLogpacsRaw(*[ch.cat([b, s]) for b, s in zip(self._buffer, samples)])
            # print(samples)

        n_samples = samples.rewards.shape[0]
        if n_samples < self.n_step:
            # current entry does not provide sufficient amount of samples for one window
            self._update_buffer(samples, 0)  # TODO verify this update is correct here.
            return

        window_samples = self._transform_samples(samples)
        window_n_samples = window_samples.rewards.shape[0]

        indices = ch.arange(self._top, self._top + window_n_samples) % self._max_size

        self._obs[indices] = window_samples.obs
        self._actions[indices] = window_samples.actions
        self._rewards[indices] = window_samples.rewards
        self._terminals[indices] = window_samples.dones
        self._time_limit_dones[indices] = window_samples.time_limit_dones
        self._terminal_obs[indices] = window_samples.terminal_obs
        self._logpacs[indices] = window_samples.logpacs

        # get positive value for indexing, because negative value is
        self._update_buffer(samples, n_samples - n_samples % self.period_length)

        self._update_pointer(window_n_samples)

    def _transform_samples(self, samples: TrajectoryOffPolicyLogpacsRaw):
        # dones = ~samples.dones * self._discount

        args = (samples.rewards, samples.dones, samples.time_limit_dones, samples.logpacs)
        # add unused action in the end for correct sliding window sizes
        actions = samples.actions
        placeholder_a = ch.empty((1,) + actions.shape[1:], dtype=actions.dtype, device=actions.device)
        actions = ch.cat([actions, placeholder_a])
        args_p1 = (samples.obs, actions, samples.terminal_obs)

        reward, dones, time_limit_dones, logpacs = [self._sliding_window(elem, self.n_step) for elem in args]
        obs, actions, terminal_obs = [self._sliding_window(elem, self.n_step + 1) for elem in args_p1]

        return TrajectoryOffPolicyLogpacsRaw(obs, actions, reward, dones, time_limit_dones, terminal_obs, logpacs)

    def _update_buffer(self, samples, steps):
        # remove next_obs sample for last timestep as this is the first observation of the next batch, if not resetting.
        # Not removing this sample leads to two duplicate observations
        samples = samples._replace(obs=samples.obs[:-1], terminal_obs=samples.terminal_obs[:-1])
        # add overlapping samples to buffer
        self._buffer = TrajectoryOffPolicyLogpacsRaw(*[s[steps:] for s in samples])

    def _sliding_window(self, data, steps):
        # sliding window and switch of data dimension and window
        data = data.unfold(0, steps, self.period_length)
        if len(data.shape) == 4:
            data = data.permute([0, 1, 3, 2])
        # combine multi environment data
        return flatten_batch(data)

    def random_batch(self, batch_size) -> TrajectoryOffPolicyLogpacs:
        indices = np.random.randint(0, self._size, batch_size)
        obs = self._obs[indices]
        next_obs = obs[:, 1:]
        terminals = self._terminals[indices]
        terminal_done = self._time_limit_dones[indices]

        if self._handle_timelimit and terminal_done.any():
            next_obs = next_obs.clone()
            # correct terminal observation is stored alongside the current step observation
            next_obs[terminal_done] = self._terminal_obs[indices, :-1][terminal_done]

            # Account for correct termination signal
            terminals *= ~terminal_done

        return TrajectoryOffPolicyLogpacs(
            obs=obs[:, :-1],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_obs=next_obs,
            terminals=~terminals * self._discount,
            logpacs=self._logpacs[indices],
        )

    def reset(self, dones=None):
        # self._buffer = TrajectoryOffPolicyLogpacsRaw(*[deque(maxlen=self.n_step - 1) for _ in range(7)])
        self._buffer = None
