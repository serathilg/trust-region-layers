import copy
from collections import deque

import numpy as np
import torch as ch

from trust_region_projections.replay_buffer.abstract_replay_buffer import AbstractReplayBuffer
from trust_region_projections.sampling.dataclass import TrajectoryOffPolicyLogpacs, TrajectoryOffPolicyRaw, \
    TrajectoryOffPolicy
from trust_region_projections.utils.torch_utils import flatten_batch


class NStepReplayBuffer(AbstractReplayBuffer):

    def __init__(self, n_step: int, max_replay_buffer_size: int, observation_dim: int, action_dim: int,
                 discount_factor: float, handle_timelimit: bool = False, dtype: ch.dtype = ch.float32,
                 device: ch.device = "cpu", **kwargs):
        """
        N-step replay buffer

        Args:
            n_step: number of steps to use for n-step return
        """

        super().__init__(max_replay_buffer_size, observation_dim, action_dim, discount_factor, handle_timelimit)
        self.n_step = n_step
        self._observation_dim = observation_dim
        self._action_dim = action_dim

        self._obs = ch.zeros((max_replay_buffer_size, observation_dim), dtype=dtype, device=device)
        self._next_obs = ch.zeros((max_replay_buffer_size, observation_dim), dtype=dtype, device=device)
        self._actions = ch.zeros((max_replay_buffer_size, action_dim), dtype=dtype, device=device)
        self._rewards = ch.zeros(max_replay_buffer_size, dtype=dtype, device=device)
        self._terminals = ch.zeros(max_replay_buffer_size, dtype=dtype, device=device)

        # save remaining n-1 samples
        self._buffer = TrajectoryOffPolicyRaw(*[deque(maxlen=n_step - 1) for _ in range(6)])

    def add_samples(self, samples: TrajectoryOffPolicyLogpacs):
        """
        Add n new samples to replay buffer.
        Args:
            samples: Namedtuple of samples to add

        Returns:

        """

        # append existing samples from buffer to new samples
        if self._buffer.obs:
            samples = TrajectoryOffPolicyRaw(*[ch.cat([ch.stack(tuple(b)), s]) for b, s in zip(self._buffer, samples)])
            # print(samples)

        if samples.rewards.shape[0] < self.n_step:
            # current entry does not provide sufficient amount of samples for one window
            self._update_buffer(samples)
            return

        n_step_samples = self._transform_samples(samples)
        n_samples = n_step_samples.rewards.shape[0]

        indices = ch.arange(self._top, self._top + n_samples) % self._max_size

        self._obs[indices] = n_step_samples.obs
        self._next_obs[indices] = n_step_samples.next_obs
        self._actions[indices] = n_step_samples.actions
        self._rewards[indices] = n_step_samples.rewards
        self._terminals[indices] = n_step_samples.terminals

        self._update_buffer(samples)

        self._update_pointer(n_samples)

    def _transform_samples(self, samples: TrajectoryOffPolicyRaw):

        next_obs = samples.obs[self.n_step:]
        done_window = samples.dones.unfold(0, self.n_step, 1)

        # append all_false for first entry, because done flags for rewards need information from previous step
        # In case the previous step was terminal, the reward is still use if it is the first of the sequence
        # In that setting this is the start of a new trajectory, if it is in the middle, the end of an trajectory
        # Hence, the first reward is always used.
        all_false = ch.zeros(done_window.shape[:-1]).unsqueeze(-1)
        done_window = ch.cat([all_false, done_window], dim=-1).to(ch.bool)

        # cumprod sets all dones=True for all timesteps following a termination (cumprod = 0) within that window.
        # Given the first timestep is always true it is not incorrectly setting the next_obs to done in case the first
        # observation is the terminal from the previous episode
        not_done_window = (~done_window).cumprod(-1)
        # The terminal state of the next_obs after n_steps `[..., -1]`
        # would be 0 if any previous state was terminal, hence pcont=0
        pcont = not_done_window[..., -1] * self._discount ** self.n_step

        discounts = self._discount ** ch.arange(self.n_step)
        reward_windows = samples.rewards.unfold(0, self.n_step, 1)
        # This gives at least the first reward of the n_steps as this done is always false
        rewards = (reward_windows * not_done_window[..., :-1] * discounts).sum(-1)

        if self._handle_timelimit and ch.any(samples.time_limit_dones):
            # Account for correct termination signal
            terminal_done = samples.time_limit_dones.unfold(0, self.n_step, 1)
            # Max automatically selects the first value if there are duplicates, hence chooses the first done
            # Theoretically this treats a correct done flag before a time limit incorrectly,
            # but in practice this could only happen when n_steps > max_episode length
            terminal_value, terminal_idx = terminal_done.max(-1)

            # if a time limit is present we take y^(index + 1) for bootstrapping instead of the y^n_steps
            # +1 is necessary to account as this is for the next step
            pcont[terminal_value] = self._discount ** (terminal_idx[terminal_value] + 1)

            # we need to copy next_obs because we only want to alter the "next observation" of the timeout transitions
            # If we maintain the reference, the observation returned from the reset is lost and not included as state
            # of the next transition tuple
            next_obs = copy.deepcopy(next_obs)

            # the correct terminal observation is stored alongside the current step obs and not the next obs
            next_obs[terminal_value] = samples.terminal_obs[self.n_step - 1:-1][terminal_value]

        pcont = pcont.to(self._terminals.dtype)
        # combine multi environment data
        args = (
            samples.obs[:-self.n_step], samples.actions[:-self.n_step + 1], rewards, next_obs, pcont)
        flat_samples = TrajectoryOffPolicy(*map(flatten_batch, args))

        return flat_samples

    def _update_buffer(self, samples: TrajectoryOffPolicyRaw):
        # remove next_obs sample for last timestep as this is the first observation of the next batch, if not resetting.
        # Not removing this sample leads to two duplicate observations
        samples = samples._replace(obs=samples.obs[:-1], terminal_obs=samples.terminal_obs[:-1])
        # add n_step - 1 latest samples to buffer
        [b.extend(s[-self.n_step + 1:]) for b, s in zip(self._buffer, samples)]

    def random_batch(self, batch_size: int) -> TrajectoryOffPolicy:
        indices = np.random.randint(0, self._size, batch_size)
        return TrajectoryOffPolicy(
            obs=self._obs[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_obs=self._obs[indices],
            terminals=self._terminals[indices]
        )

    def reset(self, dones=None):
        self._buffer = TrajectoryOffPolicyRaw(*[deque(maxlen=self.n_step - 1) for _ in range(6)])
