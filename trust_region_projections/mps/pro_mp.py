import torch as ch
from torch import nn


class MetaWorldController(object):
    """
    A Metaworld Controller. Using position and velocity information from a provided environment,
    the controller calculates a response based on the desired position and velocity.
    Unlike the other Controllers, this is a special controller for MetaWorld environments.
    They use a position delta for the xyz coordinates and a raw position for the gripper opening.
    """

    def __call__(self, des_pos: ch.Tensor, des_vel: ch.Tensor, cur_pos: ch.Tensor, cur_vel: ch.Tensor):
        return self.get_action(des_pos, des_vel, cur_pos, cur_vel)

    def get_action(self, des_pos: ch.Tensor, des_vel: ch.Tensor, cur_pos: ch.Tensor, cur_vel: ch.Tensor):
        gripper_pos = des_pos[-1]

        # cur_pos = env.data.mocap_pos.flatten()
        # cur_pos = des_pos.new(cur_pos)
        xyz_pos = des_pos[:-1]

        assert xyz_pos.shape == cur_pos.shape, \
            f"Mismatch in dimension between desired position {xyz_pos.shape} and current position {cur_pos.shape}"
        trq = ch.hstack([(xyz_pos - cur_pos), gripper_pos])
        return trq


class DeterministicProMP(nn.Module):

    def __init__(self, n_basis: int, action_dim: int, duration: float, dt: float, kernel_width: float = None,
                 center_offset: float = 0.01, zero_start: bool = True, zero_goal: bool = False, n_zero_bases: int = 2,
                 weights_scale: float = 1.0):
        """

        Args:
            n_basis: Number of Basis functions
            action_dim: Number of degrees of freedom/action dimensionality
            duration: length of the trajectory
            dt: control frequency of the trajectory
            kernel_width: width of the kernel
            center_offset: initial offset of the kernel centers
            zero_start: zero padding in the beginning to start from current position instead of 0
            zero_goal: zero padding in the end to end at current position instead of 0
            n_zero_bases: number of zeros bases for beginning or end when using
        """
        super().__init__()

        self.dt = dt
        self.action_dim = action_dim
        self.n_basis = n_basis
        self.duration = duration
        self._n_zero_bases = n_zero_bases
        self._weights_scale = weights_scale
        self._total_basis = n_basis + (zero_start + zero_goal) * n_zero_bases

        self._centers = ch.linspace(center_offset, 1. + center_offset, self._total_basis)
        self._widths = ch.ones(self._total_basis)
        self._widths *= ((1. + center_offset) / (2. * self._total_basis)) if kernel_width is None else kernel_width

        N = int(self.duration * 1 / self.dt)
        # TODO phase decay here
        t = ch.linspace(0, 1, N)
        # self.phase =

        self._features = self._exponential_kernel(t)
        self.pad = ch.nn.ConstantPad2d((0, 0, int(zero_start) * n_zero_bases, int(zero_goal) * n_zero_bases), 0)

        self.zero_start = zero_start
        self.zero_goal = zero_goal

    def forward(self, weights):
        # weights = ch.atleast_2d(weights)
        batch_shape = weights.shape[0]
        weights = weights.reshape(batch_shape, self.n_basis, self.action_dim) * self._weights_scale
        # Padding for leading or ending zeros
        weights = self.pad(weights)

        # N = int(self.duration * 1 / self.dt)
        # t = ch.linspace(0, 1, N)
        #

        # TODO integration
        # pos_features, vel_features = self._exponential_kernel(t)
        pos_features, vel_features = self._features

        # acceleration
        # (acc_features @ weights) / (self.duration ** 2)
        des_pos = pos_features.to(weights.dtype) @ weights
        des_vel = (vel_features.to(weights.dtype) @ weights) / self.duration

        return des_pos, des_vel

    def _exponential_kernel(self, z):
        diffs = z[:, None] - self._centers[None, :]

        w = ch.exp(-(ch.square(diffs) / (2 * self._widths[None, :])))
        w_der = -(diffs / self._widths[None, :]) * w
        # w_der2 = -(1 / self._widths[None, :]) * w + ch.square(diffs / self._widths[None, :]) * w

        sum_w = ch.sum(w, dim=1)[:, None]
        sum_w_der = ch.sum(w_der, dim=1)[:, None]
        # sum_w_der2 = ch.sum(w_der2, dim=1)[:, None]

        tmp = w_der * sum_w - w * sum_w_der

        # acceleration
        # ((w_der2 * sum_w - sum_w_der2 * w) * sum_w - 2 * sum_w_der * tmp) / (sum_w ** 3)
        return w / sum_w, tmp / (sum_w ** 2)
