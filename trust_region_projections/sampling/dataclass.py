import torch as ch
from typing import NamedTuple, Union

from trust_region_projections.models.policy.abstract_gaussian_policy import AbstractGaussianPolicy


class TrajectoryOnPolicyRaw(NamedTuple):
    obs: ch.Tensor
    actions: ch.Tensor
    logpacs: ch.Tensor
    rewards: ch.Tensor
    values: ch.Tensor
    dones: ch.Tensor
    time_limit_dones: ch.Tensor
    terminal_obs: ch.Tensor
    means: ch.Tensor
    stds: ch.Tensor


class TrajectoryOnPolicy(NamedTuple):
    obs: ch.Tensor
    actions: ch.Tensor
    logpacs: ch.Tensor
    rewards: ch.Tensor
    returns: ch.Tensor
    advantages: ch.Tensor
    values: ch.Tensor
    dones: ch.Tensor
    time_limit_dones: ch.Tensor
    terminal_obs: ch.Tensor
    q: tuple


class TrajectoryOffPolicyRaw(NamedTuple):
    obs: ch.Tensor
    actions: ch.Tensor
    rewards: ch.Tensor
    dones: ch.Tensor
    time_limit_dones: ch.Tensor
    terminal_obs: ch.Tensor


class TrajectoryOffPolicy(NamedTuple):
    obs: ch.Tensor
    actions: ch.Tensor
    rewards: ch.Tensor
    next_obs: ch.Tensor
    terminals: ch.Tensor


class TrajectoryOffPolicyLogpacsRaw(NamedTuple):
    obs: ch.Tensor
    actions: ch.Tensor
    rewards: ch.Tensor
    dones: ch.Tensor
    time_limit_dones: ch.Tensor
    terminal_obs: ch.Tensor
    logpacs: ch.Tensor


class TrajectoryOffPolicyLogpacs(NamedTuple):
    obs: ch.Tensor
    actions: ch.Tensor
    rewards: ch.Tensor
    next_obs: ch.Tensor
    terminals: ch.Tensor
    logpacs: ch.Tensor


class TrajectoryOffPolicyMixtureRaw(NamedTuple):
    obs: ch.Tensor
    actions: ch.Tensor
    rewards: ch.Tensor
    dones: ch.Tensor
    time_limit_dones: ch.Tensor
    terminal_obs: ch.Tensor
    policy: AbstractGaussianPolicy
