import numpy as np
import random
from typing import Any, Optional

class DynaQAgent:
    """Rol C: Şarj ve Yeniden Konumlandırma için Dyna-Q Planlama Ajanı"""
    def __init__(self, state_dim: int, action_dim: int, config: dict):
        self.config = config
        self.action_dim = action_dim
        
        # Tablo tabanlı Q-learning yapısı
        self.q_table = {}
        self.lr = config.get("lr", 0.1)
        self.gamma = config.get("gamma", 0.95)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.1)
        self.epsilon_decay = config.get("epsilon_decay", 0.98)
        self.planning_steps = config.get("planning_steps", 50) # n steps
        
        # Dyna-Q Dünya Modeli: Model[state][action] = (reward, next_state, done)
        self.model = {}

    def _get_str_state(self, obs: dict[str, Any]) -> str:
        """Gözlem sözlüğünü tablodan okuyabilmek için string bir anahtara çevirir."""
        flat_list = []
        for k, v in obs.items():
            if k != "action_mask":
                if isinstance(v, np.ndarray): flat_list.extend(v.flatten().round(1))
                elif isinstance(v, (int, float)): flat_list.append(round(v, 1))
        return str(flat_list)

    def act(self, obs: dict[str, Any]) -> int:
        mask = obs["action_mask"]
        valid_actions = np.where(mask == 1)[0]
        if len(valid_actions) == 0: return 0
        
        if random.random() < self.epsilon:
            return int(random.choice(valid_actions))
            
        state_str = self._get_str_state(obs)
        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(self.action_dim)
            
        q_values = self.q_table[state_str]
        masked_q = np.where(mask == 1, q_values, -np.inf)
        return int(np.argmax(masked_q))

    def action_values(self, obs: dict[str, Any]) -> Optional[np.ndarray]:
        state_str = self._get_str_state(obs)
        if state_str not in self.q_table: return np.zeros(self.action_dim)
        mask = obs["action_mask"]
        return np.where(mask == 1, self.q_table[state_str], np.nan)

    def action_probs(self, obs: dict[str, Any]) -> Optional[np.ndarray]: return None
    def state_values(self, obs: dict[str, Any]) -> Optional[np.ndarray]: return None

    def store_and_train(self, obs, action, reward, next_obs, done):
        s_str = self._get_str_state(obs)
        next_s_str = self._get_str_state(next_obs)
        
        if s_str not in self.q_table: self.q_table[s_str] = np.zeros(self.action_dim)
        if next_s_str not in self.q_table: self.q_table[next_s_str] = np.zeros(self.action_dim)
        
        # 1. Doğrudan Öğrenme (Direct RL)
        best_next_q = np.max(self.q_table[next_s_str])
        self.q_table[s_str][action] += self.lr * (reward + self.gamma * best_next_q * (1 - done) - self.q_table[s_str][action])
        
        # 2. Çevre Modelini Güncelleme (Model Learning)
        if s_str not in self.model: self.model[s_str] = {}
        self.model[s_str][action] = (reward, next_s_str, done)
        
        # 3. Hayali Planlama Adımları (Dyna-Q Rüyası)
        for _ in range(self.planning_steps):
            sim_s = random.choice(list(self.model.keys()))
            sim_a = random.choice(list(self.model[sim_s].keys()))
            sim_r, sim_next_s, sim_done = self.model[sim_s][sim_a]
            
            best_sim_next_q = np.max(self.q_table[sim_next_s])
            self.q_table[sim_s][sim_a] += self.lr * (sim_r + self.gamma * best_sim_next_q * (1 - sim_done) - self.q_table[sim_s][sim_a])
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay