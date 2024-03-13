from typing import Union

from abc import ABC, abstractmethod

from trust_region_projections.sampling.dataclass import TrajectoryOffPolicyRaw, TrajectoryOffPolicyLogpacsRaw


class AbstractReplayBuffer(ABC):

    def __init__(self, max_replay_buffer_size: int, observation_dim: int, action_dim: int, discount_factor: float,
                 handle_timelimit: bool = False):
        self._max_size = max_replay_buffer_size
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._discount = discount_factor
        self._handle_timelimit = handle_timelimit

        self._top = 0
        self._size = 0

    @abstractmethod
    def add_samples(self, samples: Union[TrajectoryOffPolicyRaw, TrajectoryOffPolicyLogpacsRaw]):
        """
        Add new samples to replay buffer.
        Args:
            samples: Namedtuple of samples to add

        Returns:

        """

    def _transform_samples(self, samples):
        """
        Transform samples before adding them to the replay buffer
        Args:
            samples: Named tuple of samples to transform

        Returns:

        """
        return samples

    def _update_pointer(self, n_samples: int):
        """
        Update the pointer to the latest insert
        Args:
            n_samples: number of samples inserted

        Returns:

        """
        self._top = (self._top + n_samples) % self._max_size
        if self._size < self._max_size:
            self._size = min(self._size + n_samples, self._max_size)

    @abstractmethod
    def random_batch(self, batch_size):
        """
        Generate a random batch from the replay memory
        Args:
            batch_size: number of samples to generate

        Returns:
            batch as named tuple
        """

    def reset(self, dones=None):
        """
        Reset parts of the buffer, when starting new trajectories or similar.
        Args:
            dones: Flags which trajectories to reset

        Returns:

        """
        pass

    @property
    def size(self):
        return self._size

    @property
    def discount(self):
        return self._discount
