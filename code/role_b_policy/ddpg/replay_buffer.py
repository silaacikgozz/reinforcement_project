"""Same fixed-capacity, uniform-sample replay buffer pattern as Role A's
replay_buffer.py, adapted for continuous (float-vector) actions instead of
discrete indices, and with no action mask (DroneControl-v0 has none)."""
from __future__ import annotations
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self._ptr, self._size = 0, 0
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return self._size

    def add(self, obs, action, reward, next_obs, done):
        i = self._ptr
        self.obs[i], self.action[i], self.reward[i] = obs, action, reward
        self.next_obs[i], self.done[i] = next_obs, float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self._rng.integers(0, self._size, size=batch_size)
        return (self.obs[idx], self.action[idx], self.reward[idx],
                self.next_obs[idx], self.done[idx])
