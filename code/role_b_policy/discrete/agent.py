"""One agent class serves both required methods:

  - REINFORCE + GAE: call `finish_rollout()` only when the episode ends
    (rollout_steps=None in train.py) -- GAE then degenerates to a
    whole-trajectory advantage estimate, the direct policy-gradient analogue
    of "Monte-Carlo return minus a learned baseline".
  - A2C: call `finish_rollout()` every `rollout_steps` environment steps
    regardless of episode boundaries, bootstrapping the cut-off state's
    value from the critic -- the standard online actor-critic update.

Stability measures mirror Role A's agent.py on purpose (same justification):
Huber value loss (not MSE) given this env's reward scale, gradient-norm
clipping, and explicit masking everywhere a probability is computed --
masking is applied inside PolicyValueNet.forward(), so both `act()` and the
log-prob computation during `finish_rollout()` automatically respect it
since they reuse the same masked logits.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from networks import PolicyValueNet
from gae import compute_gae


class PGAgent:
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(256, 256),
                 gamma: float = 0.99, lam: float = 0.95, lr: float = 1e-4,
                 value_coef: float = 0.5, entropy_coef: float = 0.01,
                 normalize_advantages: bool = True, grad_clip: float = 10.0,
                 device: str = "cpu", seed: int = 0):
        self.net = PolicyValueNet(obs_dim, n_actions, hidden_sizes).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.lam = gamma, lam
        self.value_coef, self.entropy_coef = value_coef, entropy_coef
        self.normalize_advantages = normalize_advantages
        self.grad_clip = grad_clip
        self.device = device
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self._buf = []  # list of dicts: obs, mask, action, logprob, value, reward, done

    # ---------- acting ----------
    def act(self, obs_vec: np.ndarray, mask: np.ndarray):
        x = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(x, m)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return int(action.item()), float(logprob.item()), float(value.item())

    def greedy_action(self, obs_vec: np.ndarray, mask: np.ndarray) -> int:
        """argmax action, for evaluation (no sampling noise)."""
        x = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(x, m)
        return int(torch.argmax(logits, dim=1).item())

    def store(self, obs_vec, mask, action, logprob, value, reward, done):
        self._buf.append(dict(obs=obs_vec, mask=mask, action=action,
                               logprob=logprob, value=value, reward=reward, done=done))

    # ---------- learning ----------
    def finish_rollout(self, last_obs_vec: np.ndarray, last_mask: np.ndarray, last_done: bool) -> dict:
        """Consumes everything stored since the last call, does one gradient
        step, clears the buffer. `last_*` describe the state the rollout was
        cut off AT (i.e. the state the agent has NOT yet acted in) -- used
        only to bootstrap the value for GAE if that state isn't terminal."""
        if not self._buf:
            return {}

        rewards = [b["reward"] for b in self._buf]
        values = [b["value"] for b in self._buf]
        dones = [b["done"] for b in self._buf]

        if last_done:
            last_value = 0.0
        else:
            x = torch.as_tensor(last_obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            m = torch.as_tensor(last_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, v = self.net(x, m)
            last_value = float(v.item())

        advantages, returns = compute_gae(rewards, values, dones, last_value, self.gamma, self.lam)
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.as_tensor(np.stack([b["obs"] for b in self._buf]), dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(np.stack([b["mask"] for b in self._buf]), dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor([b["action"] for b in self._buf], dtype=torch.int64, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        logits, values_pred = self.net(obs_t, mask_t)
        dist = Categorical(logits=logits)
        logprobs = dist.log_prob(action_t)
        entropy = dist.entropy().mean()

        policy_loss = -(logprobs * adv_t.detach()).mean()
        value_loss = F.smooth_l1_loss(values_pred, ret_t.detach())
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.opt.step()

        self._buf = []
        return {"policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item())}
