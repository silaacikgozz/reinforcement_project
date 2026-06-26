"""Dyna-Q agent: wraps Role A's DQNAgent unchanged, adds a learned model +
planning steps (extra Q-updates from model-simulated transitions, on top of
the normal real-transition update)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "role_a_dqn"))
import torch, torch.nn.functional as F
import numpy as np
from agent import DQNAgent          # noqa: E402  (Role A, reused as-is)
from model import DynamicsModel


class DynaQAgent(DQNAgent):
    def __init__(self, obs_dim, n_actions, planning_steps=5, model_lr=1e-3, **kw):
        super().__init__(obs_dim, n_actions, **kw)
        self.model = DynamicsModel(obs_dim, n_actions).to(self.device)
        self.model_opt = torch.optim.Adam(self.model.parameters(), lr=model_lr)
        self.planning_steps = planning_steps
        self.n_actions = n_actions

    def train_model(self, batch):
        obs, action, reward, next_obs, next_mask, done = batch
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(action, dtype=torch.int64, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        pred_next, pred_r, pred_done_logit = self.model(obs_t, act_t)
        loss = (F.mse_loss(pred_next, next_obs_t) + F.mse_loss(pred_r, reward_t) +
                F.binary_cross_entropy_with_logits(pred_done_logit, done_t))
        self.model_opt.zero_grad(); loss.backward(); self.model_opt.step()
        return float(loss.item())

    def plan(self, buffer, next_mask_batch_for_planning):
        """`planning_steps` extra Q-updates using model-predicted (r, s')
        for real (s,a) pairs sampled from the buffer. Reuses the real
        next_mask of whatever transition we sampled the (s,a) from, as a
        cheap stand-in for the model predicting the mask too."""
        for _ in range(self.planning_steps):
            obs, action, _, _, next_mask, _ = buffer.sample(64)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            act_t = torch.as_tensor(action, dtype=torch.int64, device=self.device)
            with torch.no_grad():
                pred_next, pred_r, pred_done_logit = self.model(obs_t, act_t)
                pred_done = (torch.sigmoid(pred_done_logit) > 0.5).float()
            sim_batch = (obs, action, pred_r.numpy(), pred_next.numpy(), next_mask, pred_done.numpy())
            self.learn(sim_batch)
