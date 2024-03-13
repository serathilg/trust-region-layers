import copy

import numpy as np
import torch as ch

from trust_region_projections.replay_buffer.abstract_replay_buffer import AbstractReplayBuffer
from trust_region_projections.sampling.dataclass import TrajectoryOffPolicy, TrajectoryOffPolicyRaw
from trust_region_projections.utils.torch_utils import flatten_batch


class ReplayBuffer(AbstractReplayBuffer):

    def __init__(self, max_replay_buffer_size, observation_dim, action_dim, discount_factor: float = 0.99,
                 handle_timelimit: bool = False, dtype: ch.dtype = ch.float32, device: ch.device = "cpu", **kwargs):
        super().__init__(max_replay_buffer_size, observation_dim, action_dim, discount_factor, handle_timelimit)

        self._obs = ch.zeros((max_replay_buffer_size, observation_dim), dtype=dtype, device=device)
        self._next_obs = ch.zeros((max_replay_buffer_size, observation_dim), dtype=dtype, device=device)
        self._actions = ch.zeros((max_replay_buffer_size, action_dim), dtype=dtype, device=device)
        self._rewards = ch.zeros(max_replay_buffer_size, dtype=dtype, device=device)
        # self._terminals[i] = a terminal was received at time i * discount factor
        self._terminals = ch.zeros(max_replay_buffer_size, dtype=dtype, device=device)

    def add_samples(self, samples: TrajectoryOffPolicyRaw):
        """
        Add n new samples to replay buffer.
        Args:
            samples: Namedtuple of samples to add. Elements should be formatted as [n_samples, n_envs, data_dim].

        Returns:

        """
        samples = self._transform_samples(samples)

        n_samples = samples.obs.shape[0]
        indices = ch.arange(self._top, self._top + n_samples) % self._max_size

        self._obs[indices] = samples.obs
        self._actions[indices] = samples.actions
        self._rewards[indices] = samples.rewards
        self._terminals[indices] = samples.terminals.to(self._terminals.dtype)
        self._next_obs[indices] = samples.next_obs

        self._update_pointer(n_samples)

    def _transform_samples(self, samples: TrajectoryOffPolicyRaw):
        next_obs = samples.obs[1:]
        dones = samples.dones

        if self._handle_timelimit and ch.any(samples.time_limit_dones):
            # Account for correct termination signal
            terminal_done = samples.time_limit_dones
            dones = dones * ~terminal_done

            # If we maintain the reference, the observation returned from the reset is lost and not included as state
            # of the next transition tuple
            next_obs = copy.deepcopy(next_obs)

            # the correct terminal observation is stored alongside the current step obs and not the next obs
            next_obs[terminal_done] = samples.terminal_obs[:-1][terminal_done]

        pcont = (~dones).to(self._terminals.dtype) * self._discount
        # combine multi environment data
        args = (
            samples.obs[:-1], samples.actions, samples.rewards, next_obs, pcont)
        flat_samples = TrajectoryOffPolicy(*map(flatten_batch, args))

        return flat_samples

    def random_batch(self, batch_size) -> TrajectoryOffPolicy:
        indices = ch.randint(self.size, (batch_size,))
        return TrajectoryOffPolicy(
            obs=self._obs[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_obs=self._next_obs[indices],
            terminals=self._terminals[indices],
        )
