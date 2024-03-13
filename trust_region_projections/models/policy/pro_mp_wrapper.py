import torch as ch
from torch import nn

from trust_region_projections.models.policy.abstract_gaussian_policy import AbstractGaussianPolicy
from trust_region_projections.mps.pro_mp import DeterministicProMP, MetaWorldController
from trust_region_projections.utils.torch_utils import inverse_softplus


class ProMPWrapper(nn.Module):
    def __init__(self, policy: AbstractGaussianPolicy, pro_mp: DeterministicProMP):
        super().__init__()

        self.policy = policy
        self.pro_mp = pro_mp
        self.controller = MetaWorldController()

        self._activation = nn.Softplus()
        self._shift = inverse_softplus(ch.tensor(1.0 - 1e-8))  # init - min
        self.log_std = nn.Parameter(ch.normal(0, 0.01, (self.pro_mp.action_dim,)))

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "policy":
                raise AttributeError()
            return getattr(self.policy, name)

    def forward(self, x, train=True, start_pos: ch.Tensor = None, sample=True):
        import numpy as np
        x = x[:, np.hstack([
            # Current observation
            [False] * 3,  # end-effector position
            [False] * 1,  # normalized gripper open distance
            [True] * 3,  # main object position
            [False] * 4,  # main object quaternion
            [False] * 3,  # secondary object position
            [False] * 4,  # secondary object quaternion
            # Previous observation
            # TODO: Include previous values? According to their source they might be wrong for the first iteration.
            [False] * 3,  # previous end-effector position
            [False] * 1,  # previous normalized gripper open distance
            [False] * 3,  # previous main object position
            [False] * 4,  # previous main object quaternion
            [False] * 3,  # previous second object position
            [False] * 4,  # previous second object quaternion
            # Goal
            [True] * 3,  # goal position
        ])]

        weights, weight_std = self.policy(x, train=train)
        # TODO sample weights here
        if train:
            weights = self.policy.rsample((weights, weight_std))
        des_pos, des_vel = self.pro_mp(weights)

        # batch_shape = weights.shape[0]
        # des_pos = des_vel = ch.ones(batch_shape, 500, 5).to(weights.dtype) @ weights.reshape(batch_shape, 5, 4)

        if self.pro_mp.zero_start and start_pos is not None:
            des_pos += start_pos

        # std in action space
        # std = 0.1 * ch.eye(des_pos.shape[-1])[None].expand(des_pos.shape[:-1] + (-1, -1))

        std = (self._activation(self.log_std + self._shift) + 1e-8)
        std = std.diag_embed().expand(des_pos.shape[:-1] + (-1, -1))
        return des_pos, des_vel, std.to(des_pos.device, des_pos.dtype)

    def act(self, des_pos, des_vel, cur_pos, cur_vel):
        return self.controller(des_pos, des_vel, cur_pos, cur_vel)
