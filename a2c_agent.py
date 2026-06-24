import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCriticNet, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = self.common(state)
        action_logits = self.actor(x)
        state_value = self.critic(x)
        return action_logits, state_value

class A2CAgent:
    """Rol B: Policy-Based / Actor-Critic Ajanı (Nisa)"""
    def __init__(self, state_dim: int, action_dim: int, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = ActorCriticNet(state_dim, action_dim, config.get("hidden_dim", 128)).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.get("lr", 0.0003))
        self.epsilon = 0.0  
        
        self.reset_memory()

    def reset_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def store(self, log_prob: torch.Tensor, value: torch.Tensor, reward: float, done: bool):
        self.rewards.append(torch.tensor([reward], dtype=torch.float32, device=self.device))
        self.dones.append(torch.tensor([done], dtype=torch.float32, device=self.device))
        self.log_probs.append(log_prob)
        self.values.append(value)

    def _flatten_obs(self, obs: dict) -> torch.Tensor:
        flat_list = []
        for k, v in obs.items():
            if k != "action_mask":
                if isinstance(v, np.ndarray): flat_list.extend(v.flatten())
                elif isinstance(v, (int, float)): flat_list.append(v)
        return torch.tensor(flat_list, dtype=torch.float32, device=self.device).unsqueeze(0)

    def act(self, obs: dict):
        state_t = self._flatten_obs(obs)
        mask = torch.tensor(obs["action_mask"], dtype=torch.float32, device=self.device)
        
        logits, value = self.network(state_t)
        logits = logits.squeeze(0)
        
        masked_logits = logits + (mask - 1.0) * 1e9
        probs = torch.softmax(masked_logits, dim=-1)
        
        dist = Categorical(probs)
        action = dist.sample()
        
        return int(action.item()), dist.log_prob(action), value.squeeze(0)

    def update(self):
        """GAE HESAPLAMASI VE MODEL GÜNCELLEME SÜRECİ"""
        
        rewards = torch.stack(self.rewards)
        dones = torch.stack(self.dones)
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)

        gamma = self.config.get("gamma", 0.99)
        gae_lambda = self.config.get("gae_lambda", 0.95)
        value_coef = self.config.get("value_coef", 0.5)
        entropy_coef = self.config.get("entropy_coef", 0.01)

        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        gae = 0.0
        # --- BURASI CRITICAL GAE DÖNGÜSÜ ---
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            # TD Error hesaplaması
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            
            # GAE Formülü (Düzeltilmiş Satır kanka)
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        # Normalizasyon ile kararlılık sağlama
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Loss Hesaplamaları
        value_loss = nn.MSELoss()(values, returns)
        policy_loss = -(log_probs * advantages.detach()).mean()
        entropy_loss = -log_probs.mean()

        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.reset_memory()
        return total_loss.item()