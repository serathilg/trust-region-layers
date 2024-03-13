from trust_region_projections.replay_buffer.n_step_replay_buffer import NStepReplayBuffer
from trust_region_projections.replay_buffer.abstract_replay_buffer import AbstractReplayBuffer
from trust_region_projections.replay_buffer.logpac_buffer import LogpacReplayBuffer
from trust_region_projections.replay_buffer.trajectory_replay_buffer import TrajectoryReplayBuffer
from trust_region_projections.replay_buffer.replay_buffer import ReplayBuffer


def get_replay_buffer(buffer_type: str, **kwargs) -> AbstractReplayBuffer:
    """
    Value loss and critic network factory
    Args:
        buffer_type: what type of buffer to use to use, one of 'double' (double Q-learning),
        'duelling' (Duelling Double Q-learning), and 'retrace' (Retrace by Munos et al. 2016)
        device: torch device
        dtype: torch dtype
        **kwargs: replay buffer arguments

    Returns:
        Gaussian Policy instance
    """

    if buffer_type == "default":
        buffer = ReplayBuffer(**kwargs)
    elif buffer_type == "trajectory":
        buffer = TrajectoryReplayBuffer(**kwargs)
    elif buffer_type == "logpacs":
        buffer = LogpacReplayBuffer(**kwargs)
    elif buffer_type == "n_step":
        buffer = NStepReplayBuffer(**kwargs)
    else:
        raise ValueError(f"Invalid replay buffer type {buffer_type}. Select one of 'default', 'trajectory', 'logpacs'"
                         f"or 'n_step'.")

    return buffer
