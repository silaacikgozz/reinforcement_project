"""Core DQN agent. One class covers all three required variants:

  - DQN          : dueling=False, double=False
  - Double DQN   : dueling=False, double=True
  - Dueling DQN  : dueling=True   (commonly combined with double=True too;
                                    keep them as separate flags so the
                                    ablation can toggle each independently)

`use_target_network` is the Section-4 ablation knob (target-network on/off).
When off, `self.target_net is self.q_net` -- i.e. there is no delayed copy,
the "target" is just whatever the online network outputs right now. This is
intentionally the textbook way to demonstrate why target networks help:
without one, the regression target moves every single gradient step, which
is the classic source of the instability/"absurd loss" failure mode.

Stability measures (all here, all visible -- nothing hidden inside networks.py):
  1. Action masking applied BOTH at decision time (`act`) AND at
     target-computation time (`learn`, for both the online argmax in Double
     DQN and the max in vanilla DQN). Masking only at decision time and
     forgetting it at target time was the bug behind the previous session's
     DQN-worse-than-random numbers: the target would bootstrap off Q-values
     for actions that are never actually legal (e.g. assigning a busy drone),
     which has no reason to be well-calibrated and can be arbitrarily large.
  2. Huber loss (smooth_l1) instead of MSE -- bounded gradient w.r.t.
     occasional large TD-error outliers (the reward scale here spans
     -50 .. +15 per step, an MSE on that range overweights rare extreme
     transitions, e.g. battery depletion, by squaring them).
  3. Gradient norm clipping.
  4. Target network with periodic hard sync (when enabled).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from networks import QNetwork, DuelingQNetwork

_MASK_NEG = -1e8  # large-but-finite "invalid action" value; avoids -inf propagating NaNs


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int,
                 hidden_sizes: Sequence[int] = (256, 256),
                 dueling: bool = False, double: bool = False,
                 use_target_network: bool = True,
                 gamma: float = 0.99, lr: float = 1e-4,
                 target_update_every: int = 1000, grad_clip: float = 10.0,
                 device: str = "cpu", seed: int = 0):
        NetClass = DuelingQNetwork if dueling else QNetwork
        self.q_net = NetClass(obs_dim, n_actions, hidden_sizes).to(device)
        self.use_target_network = use_target_network
        if use_target_network:
            self.target_net = NetClass(obs_dim, n_actions, hidden_sizes).to(device)
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.target_net.eval()
        else:
            self.target_net = self.q_net  # ablation: no delayed target at all

        self.double = double
        self.gamma = gamma
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.target_update_every = target_update_every
        self.grad_clip = grad_clip
        self.n_actions = n_actions
        self.device = device
        self._learn_steps = 0
        self.rng = np.random.default_rng(seed)

    # ---------- acting ----------
    def greedy_action(self, obs_vec: np.ndarray, mask: np.ndarray) -> int:
        with torch.no_grad():
            x = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_net(x).squeeze(0).cpu().numpy()
        q = np.where(mask.astype(bool), q, -np.inf)
        return int(np.argmax(q))

    def act(self, obs_vec: np.ndarray, mask: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            valid = np.flatnonzero(mask)
            return int(self.rng.choice(valid))
        return self.greedy_action(obs_vec, mask)

    def q_values(self, obs_vec: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Masked Q-values (invalid actions -> NaN). Used by the
        Introspectable adapter / visualizer, not by training."""
        with torch.no_grad():
            x = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_net(x).squeeze(0).cpu().numpy()
        return np.where(mask.astype(bool), q, np.nan)

    # ---------- learning ----------
    def learn(self, batch) -> float:
        obs, action, reward, next_obs, next_mask, done = batch
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(action, dtype=torch.int64, device=self.device)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        next_mask_t = torch.as_tensor(next_mask, dtype=torch.float32, device=self.device)

        q_values = self.q_net(obs_t)
        q_sa = q_values.gather(1, action_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            invalid = (next_mask_t == 0)
            if self.double:
                # Double DQN: action chosen by the ONLINE net, value read from
                # the TARGET net -- decouples selection from evaluation to
                # curb overestimation (van Hasselt et al. 2016).
                next_q_online = self.q_net(next_obs_t).masked_fill(invalid, _MASK_NEG)
                next_actions = next_q_online.argmax(dim=1)
                next_q_target_all = self.target_net(next_obs_t)
                next_q = next_q_target_all.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_target_all = self.target_net(next_obs_t).masked_fill(invalid, _MASK_NEG)
                next_q = next_q_target_all.max(dim=1).values
            target = reward_t + self.gamma * (1.0 - done_t) * next_q

        loss = F.smooth_l1_loss(q_sa, target)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.opt.step()

        self._learn_steps += 1
        if self.use_target_network and self._learn_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())
