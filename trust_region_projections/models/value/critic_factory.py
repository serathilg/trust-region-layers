from trust_region_projections.models.value.critic import BaseCritic, TargetCritic, DoubleCritic
from trust_region_projections.models.value.qf_net import QFNet
import torch as ch

from trust_region_projections.models.value.vf_net import VFNet


def get_critic(critic_type: str, dim: int, device: ch.device = "cpu", dtype=ch.float32, **kwargs):
    """
    Critic network factory
    Args:
        critic_type: what type of critic, one of 'base' (standard Q-learning), 'double' (double Q-learning),
                     and 'duelling' (Duelling Double Q-learning).
        dim: input dimensionality.
        device: torch device
        dtype: torch dtype
        **kwargs: critic arguments

    Returns:
        Critic instance
    """
    # Value-networks
    if critic_type == "base_v" or critic_type == "value":
        vf = VFNet(dim, 1, **kwargs)
        critic = BaseCritic(vf)
        return critic.to(device, dtype)
    elif critic_type == "target_v" or critic_type == "vlearn" or critic_type == "vtrace":
        vf = VFNet(dim, 1, **kwargs)
        critic = TargetCritic(vf)
        return critic.to(device, dtype)
    elif critic_type == "double_v" or critic_type == "vlearn_double":
        vf1 = VFNet(dim, 1, **kwargs)
        vf2 = VFNet(dim, 1, **kwargs)
        critic = DoubleCritic(vf1, vf2)
        return critic.to(device, dtype)

    # Q-Networks
    qf = QFNet(dim, 1, **kwargs).to(device, dtype)

    if critic_type == "base":
        critic = BaseCritic(qf)
    elif critic_type == "target":
        critic = TargetCritic(qf)
    elif critic_type == "double":
        qf2 = QFNet(dim, 1, **kwargs).to(device, dtype)
        critic = DoubleCritic(qf, qf2)
    else:
        raise ValueError(f"Invalid value_loss type {critic_type}. Select one of 'base', 'double', 'duelling'.")

    return critic.to(device, dtype)
