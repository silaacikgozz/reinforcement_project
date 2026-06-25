"""Policy+value network for the discrete DroneDispatch-v0 policy-gradient
methods (REINFORCE+GAE and A2C share this exact architecture; what differs
between them is the rollout/update strategy in agent.py, not the network).

Design mirrors Role A's networks.py on purpose (same hidden sizes, same
input dim from code/common/obs_encoding.py) so the two roles are comparable
in the report -- the only architectural difference that matters for the
comparison is value-based vs policy-based, not network capacity.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

_MASK_NEG = -1e8  # same constant as Role A's agent.py, same rationale:
                   # large-but-finite so it never produces NaN gradients.


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: Sequence[int] = (256, 256)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, n_actions)
        self.value_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """mask: (B, n_actions) float, 1 = legal, 0 = illegal.
        Returns (masked_logits, value)."""
        h = self.trunk(x)
        logits = self.policy_head(h)
        logits = logits.masked_fill(mask == 0, _MASK_NEG)
        value = self.value_head(h).squeeze(-1)
        return logits, value
