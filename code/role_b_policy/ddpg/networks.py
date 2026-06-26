"""Actor/Critic networks for DDPG on DroneControl-v0.

Action space is Box(low=[0, -1], high=[1, 1]): index 0 is speed in [0,1],
index 1 is a heading delta in [-1,1] (scaled to +-pi by the env itself, not
here). The actor's output activation is chosen per-dimension to match this
box exactly rather than using a single tanh and rescaling both dims the same
way, which would waste half the speed dimension's range on outputs that get
clipped to 0 anyway.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _mlp(in_dim, hidden_sizes, out_dim):
    layers = []
    prev = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int] = (128, 128)):
        super().__init__()
        self.net = _mlp(obs_dim, hidden_sizes, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        speed = torch.sigmoid(raw[..., 0:1])          # -> [0, 1]
        heading_delta = torch.tanh(raw[..., 1:2])      # -> [-1, 1]
        return torch.cat([speed, heading_delta], dim=-1)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 2, hidden_sizes: Sequence[int] = (128, 128)):
        super().__init__()
        self.net = _mlp(obs_dim + action_dim, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


# TD3 uses two independent critics and takes their minimum when building the
# regression target -- this is the direct fix for the overestimation bias
# that a single critic exhibits (visible in our DDPG run as a steadily
# growing |actor_loss|, the DDPG analogue of DQN's diverging Q-values).
# Same class, just instantiated twice in agent.py; no separate class needed.


class TwinCritic(nn.Module):
    """Two independent critics (TD3, Fujimoto et al. 2018). Taking the min
    of the two when computing the TD target is the direct fix for DDPG's
    single-critic overestimation: a single critic's error is, on average,
    optimistic (the actor is explicitly trained to find actions the critic
    over-values), and min(Q1, Q2) cancels out the part of that error that
    isn't shared by both independently-initialized networks."""

    def __init__(self, obs_dim: int, action_dim: int = 2, hidden_sizes: Sequence[int] = (128, 128)):
        super().__init__()
        self.q1 = Critic(obs_dim, action_dim, hidden_sizes)
        self.q2 = Critic(obs_dim, action_dim, hidden_sizes)

    def forward(self, obs, action):
        return self.q1(obs, action), self.q2(obs, action)
