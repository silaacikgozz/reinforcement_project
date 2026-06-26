"""Learned dynamics model for Dyna-Q: predicts (reward, done, next_obs) from
(obs, action). Trained on real transitions; used to generate simulated
transitions for extra Q-updates (planning)."""
import torch, torch.nn as nn

class DynamicsModel(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(256, 256)):
        super().__init__()
        layers, prev = [], obs_dim + n_actions
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]; prev = h
        self.trunk = nn.Sequential(*layers)
        self.next_obs_head = nn.Linear(prev, obs_dim)
        self.reward_head = nn.Linear(prev, 1)
        self.done_head = nn.Linear(prev, 1)
        self.n_actions = n_actions

    def forward(self, obs, action_idx):
        a_onehot = torch.nn.functional.one_hot(action_idx, self.n_actions).float()
        h = self.trunk(torch.cat([obs, a_onehot], dim=-1))
        return self.next_obs_head(h), self.reward_head(h).squeeze(-1), self.done_head(h).squeeze(-1)
