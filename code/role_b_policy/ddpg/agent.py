"""DDPG agent for DroneControl-v0, stabilized with the three TD3 fixes
(Fujimoto, van Hoof & Meger, 2018) -- this is what actually solved the
"stuck against the wall / stands still" degenerate policy we found by
diagnosing a vanilla-DDPG run (see code/role_b_policy/ddpg/README.md,
Engineering Notes):

  1. Twin critics, target = min(Q1, Q2) -- directly counters the
     overestimation bias that was visible as a steadily growing
     |actor_loss| in the vanilla run; an inflated, wrong Q-value is exactly
     what makes "stand still forever" look falsely attractive.
  2. Target policy smoothing -- small clipped noise added to the action
     used inside the TARGET Q computation (not the behavior action), so the
     critic can't be exploited by a razor-thin action-value peak.
  3. Delayed policy + target updates -- the actor and both target networks
     are only updated every `policy_delay` critic updates, giving the
     critic time to settle before the policy chases it.

Exploration: additive Gaussian noise on the actor's output during
*behavior* (training-time) action selection, decayed but with a floor that
never reaches zero (see configs/ddpg.yaml's noise_end) -- a deliberate
change from the original vanilla-DDPG attempt, where noise decayed low
enough that a policy which had collapsed into a fixed point (identical
observation forever, e.g. pinned against the grid boundary) could no
longer be perturbed out of it during training.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from networks import Actor, Critic


class DDPGAgent:
    def __init__(self, obs_dim: int, action_low, action_high,
                 hidden_sizes=(128, 128), gamma: float = 0.99, tau: float = 0.005,
                 actor_lr: float = 1e-4, critic_lr: float = 1e-3,
                 policy_delay: int = 2, target_noise_std: float = 0.2, target_noise_clip: float = 0.5,
                 grad_clip: float = 10.0, device: str = "cpu", seed: int = 0):
        self.actor = Actor(obs_dim, hidden_sizes).to(device)
        self.actor_target = Actor(obs_dim, hidden_sizes).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(obs_dim, 2, hidden_sizes).to(device)
        self.critic1_target = Critic(obs_dim, 2, hidden_sizes).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(obs_dim, 2, hidden_sizes).to(device)
        self.critic2_target = Critic(obs_dim, 2, hidden_sizes).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr)

        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=device)
        self.gamma, self.tau, self.grad_clip = gamma, tau, grad_clip
        self.policy_delay = policy_delay
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.device = device
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self._update_count = 0

    def act(self, obs_vec: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        x = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(x).squeeze(0).cpu().numpy()
        if noise_std > 0:
            a = a + self.rng.normal(0, noise_std, size=a.shape).astype(np.float32)
        return np.clip(a, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())

    def soft_update(self, target: torch.nn.Module, source: torch.nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tp.data * (1.0 - self.tau) + sp.data * self.tau)

    def learn(self, batch) -> dict:
        obs, action, reward, next_obs, done = batch
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        # ---- critics: target = r + gamma * min(Q1', Q2')(s', smoothed a') ----
        with torch.no_grad():
            next_action = self.actor_target(next_obs_t)
            noise = (torch.randn_like(next_action) * self.target_noise_std).clamp(
                -self.target_noise_clip, self.target_noise_clip)
            next_action = torch.clamp(next_action + noise, self.action_low, self.action_high)
            target_q1 = self.critic1_target(next_obs_t, next_action)
            target_q2 = self.critic2_target(next_obs_t, next_action)
            target_q = torch.min(target_q1, target_q2)
            y = reward_t + self.gamma * (1.0 - done_t) * target_q

        q1 = self.critic1(obs_t, action_t)
        q2 = self.critic2(obs_t, action_t)
        critic_loss = F.smooth_l1_loss(q1, y) + F.smooth_l1_loss(q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), self.grad_clip)
        self.critic_opt.step()

        stats = {"critic_loss": float(critic_loss.item()), "actor_loss": float("nan")}
        self._update_count += 1

        # ---- delayed actor + target update ----
        if self._update_count % self.policy_delay == 0:
            actor_loss = -self.critic1(obs_t, self.actor(obs_t)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_opt.step()

            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic1_target, self.critic1)
            self.soft_update(self.critic2_target, self.critic2)
            stats["actor_loss"] = float(actor_loss.item())

        return stats
