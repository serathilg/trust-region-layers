import copy

import numpy as np
import torch as ch

from trust_region_projections.replay_buffer.abstract_replay_buffer import AbstractReplayBuffer
from trust_region_projections.sampling.dataclass import TrajectoryOffPolicy, TrajectoryOffPolicyMixtureRaw, \
    TrajectoryOffPolicyLogpacs
from trust_region_projections.utils.torch_utils import flatten_batch


class LogpacReplayBuffer(AbstractReplayBuffer):

    def __init__(self, max_replay_buffer_size, observation_dim, action_dim, polyak_weight=0.1,
                 discount_factor: float = 0.99, handle_timelimit: bool = False, dtype: ch.dtype = ch.float32,
                 device: ch.device = "cpu", **kwargs):
        super().__init__(max_replay_buffer_size, observation_dim, action_dim, discount_factor, handle_timelimit)

        # self._top_log_pacs = 0
        # self._max_policy_history_size = 1  # 100
        # self._policies: Deque[AbstractGaussianPolicy] = deque(maxlen=self._max_policy_history_size)
        # self._policies_update_interval = 1  # 10
        # self._policies_update_counter = 0

        self.polyak_weight = ch.tensor(polyak_weight, dtype=dtype, device=device)

        self._obs = ch.zeros((max_replay_buffer_size, observation_dim), dtype=dtype, device=device)
        self._next_obs = ch.zeros((max_replay_buffer_size, observation_dim), dtype=dtype, device=device)
        self._actions = ch.zeros((max_replay_buffer_size, action_dim), dtype=dtype, device=device)
        self._rewards = ch.zeros(max_replay_buffer_size, dtype=dtype, device=device)
        self._logpacs = ch.zeros(max_replay_buffer_size, dtype=dtype, device=device)
        # self._logpacs = ch.zeros((max_replay_buffer_size, self._max_policy_history_size), dtype=dtype, device=device)
        # self._terminals[i] = a terminal was received at time i
        self._terminals = ch.zeros(max_replay_buffer_size, dtype=dtype, device=device)

    def add_samples(self, samples: TrajectoryOffPolicyMixtureRaw):
        """
        Add n new samples to replay buffer.
        Args:
            samples: Namedtuple of samples to add. Elements should be formatted as [n_samples, n_envs, data_dim].

        Returns:

        """

        new_policy = samples.policy
        samples = self._transform_samples(samples)

        n_samples = samples.obs.shape[0]
        indices = ch.arange(self._top, self._top + n_samples) % self._max_size

        self._obs[indices] = samples.obs
        self._actions[indices] = samples.actions
        self._rewards[indices] = samples.rewards
        self._terminals[indices] = samples.terminals
        self._next_obs[indices] = samples.next_obs

        with ch.no_grad():
            p = new_policy(samples.obs)
            self._logpacs[indices] = new_policy.log_probability(p, samples.actions)

        # online start
        # with ch.no_grad():
        #     p = new_policy(samples.obs)
        #     self._logpacs[indices] = new_policy.log_probability(p, samples.actions)[:, None]
        # online end

        # moving avg start
        # update before pointer gets updated, new samples already got a value assigned
        with ch.no_grad():
            for batch_idx in ch.split(ch.arange(self.size), 100000):
                p = new_policy(self._obs[batch_idx])
                self._logpacs[batch_idx] = ch.logaddexp(
                    ch.log(1 - self.polyak_weight) + self._logpacs[batch_idx],
                    ch.log(self.polyak_weight) + new_policy.log_probability(p, self._actions[batch_idx]))
                # self._logpacs[:self.size] *= self.tau
                # self._logpacs[:self.size] += (1 - self.tau) * new_policy.log_probability(p, self._actions[:self.size])[:,
                #                                               None]
        # moving avg end

        # update pointer before updating the logpac values, otherwise new samples get excluded
        self._update_pointer(n_samples)

        # if self._policies_update_counter % self._policies_update_interval == 0:
        #
        #     # Generate new logpacs based on all existing policies
        #     # TODO: Currently one unnecessary forward pass for policy that gets kicked out anyway
        #     if self._policies:
        #         with ch.no_grad():
        #             new_logpacs = [pi.log_probability(pi(samples.obs), samples.actions) for pi in self._policies]
        #             new_logpacs = ch.atleast_2d(ch.vstack(new_logpacs)).T
        #         self._logpacs[indices, :len(self._policies)] = new_logpacs
        #
        #     # Add new policy logpacs for all existing samples (exclude the empty placeholders)
        #     with ch.no_grad():
        #         p = new_policy(self._obs[:self.size])
        #         logpacs = new_policy.log_probability(p, self._actions[:self.size])
        #
        #     if len(self._policies) < self._max_policy_history_size:
        #         self._logpacs[:self.size, len(self._policies)] = logpacs
        #     else:
        #         # FIFO Tensor
        #         self._logpacs[:self.size] = ch.cat((self._logpacs[:self.size, 1:], logpacs[:, None]), dim=1)
        #
        #     # Add policy to stored ones
        #     self._policies.append(new_policy)
        #
        # self._policies_update_counter += 1

    def _transform_samples(self, samples: TrajectoryOffPolicyMixtureRaw):
        next_obs = samples.obs[1:]
        dones = samples.dones

        if self._handle_timelimit and ch.any(samples.time_limit_dones):
            # Account for correct termination signal
            terminal_done = samples.time_limit_dones
            dones = dones * ~terminal_done

            # If we maintain the reference, the observation returned from the reset is lost and not included as state
            # of the next transition tuple
            next_obs = copy.deepcopy(next_obs)

            # correct terminal observation is stored alongside the current step observation
            next_obs[terminal_done] = samples.terminal_obs[:-1][terminal_done]

        pcont = (~dones).to(self._terminals.dtype) * self._discount
        # combine multi environment data
        args = (
            samples.obs[:-1], samples.actions, samples.rewards, next_obs, pcont)
        return TrajectoryOffPolicy(*map(flatten_batch, args))

    def random_batch(self, batch_size) -> TrajectoryOffPolicyLogpacs:
        indices = np.random.randint(0, self.size, batch_size)
        return TrajectoryOffPolicyLogpacs(
            obs=self._obs[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_obs=self._next_obs[indices],
            terminals=self._terminals[indices],
            logpacs=self._logpacs[indices]
            # logpacs=ch.logsumexp(self._logpacs[indices, :len(self._policies)], dim=-1)
        )
