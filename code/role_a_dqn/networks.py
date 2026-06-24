"""Q-networks for Role A (DQN / Double DQN share this; Dueling DQN uses the
second class). Both take the flattened observation vector (see
common/obs_encoding.py, dim = flat_dim(cfg)) and output one Q-value per
discrete action (cfg.n_actions = 169 on the standard config).

Design decisions (log for the report / oral defense):

- Plain MLP, no conv/attention. The grid is already flattened into the input
  vector (see obs_encoding.py); a CNN over the grid would likely help but is
  out of scope for the required baseline -- note this as a limitation in the
  report rather than silently skipping it.
- Hidden sizes are a config knob (cfg list, e.g. [256, 256]), not hardcoded
  here, so configs/dqn.yaml controls capacity without touching this file.
- We deliberately do NOT clip/clamp Q-values inside the network. Stability
  (avoiding the "absurd loss" failure mode) is handled in agent.py via:
    (a) masking invalid actions to -inf before argmax / before computing the
        target's max, so the network is never trained toward a Q-value for
        an action that was never legal,
    (b) Huber loss (smooth_l1) instead of MSE, which is far less sensitive to
        the occasional large TD-error spike,
    (c) gradient norm clipping,
    (d) a target network updated on a fixed schedule (not every step).
  Putting all of that here would hide it from the ablation (target-network
  on/off) and from train.py, where it needs to be visible and toggleable.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _mlp_trunk(in_dim: int, hidden_sizes: Sequence[int]) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    return nn.Sequential(*layers), prev


class QNetwork(nn.Module):
    """Standard (non-dueling) Q-network. Used by both vanilla DQN and
    Double DQN -- Double DQN changes how the *target* is computed in
    agent.py, not the network architecture."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: Sequence[int] = (256, 256)):
        super().__init__()
        self.trunk, last = _mlp_trunk(obs_dim, hidden_sizes)
        self.head = nn.Linear(last, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


class DuelingQNetwork(nn.Module):
    """Splits into a state-value stream V(s) (scalar) and an advantage
    stream A(s,a) (n_actions), recombined as

        Q(s,a) = V(s) + (A(s,a) - mean_a' A(s,a'))

    The mean-subtraction (rather than max-subtraction) is the standard
    choice from Wang et al. 2016 -- it's more stable because it doesn't let
    a single action's advantage estimate dominate identifiability between
    V and A; we use it here and note this choice explicitly in the report's
    method-origin section.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: Sequence[int] = (256, 256)):
        super().__init__()
        self.trunk, last = _mlp_trunk(obs_dim, hidden_sizes)
        self.value_head = nn.Linear(last, 1)
        self.advantage_head = nn.Linear(last, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value_head(h)                                   # (B, 1)
        a = self.advantage_head(h)                                # (B, n_actions)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
