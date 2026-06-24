"""Fixed-capacity replay buffer, uniform sampling. Stores already-flattened
observation vectors (not the raw Dict obs) so sampling is just array slicing
-- no re-flattening at sample time.

We also store next_mask (the action mask for the *next* state) because
computing the Double-DQN / vanilla-DQN target requires masking
Q(s', .) before taking the max -- this is the exact piece that was missing in
the previous session's run and produced the "absurd loss" / worse-than-random
results. Storing it explicitly (rather than recomputing it later from
next_obs) keeps that masking unambiguous and cheap.
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, n_actions: int, seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_mask = np.zeros((capacity, n_actions), dtype=np.float32)
        self.action = np.zeros((capacity,), dtype=np.int64)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)   # terminated OR truncated
        self._ptr = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(self, obs, action, reward, next_obs, next_mask, done):
        i = self._ptr
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.next_mask[i] = next_mask
        self.done[i] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self._rng.integers(0, self._size, size=batch_size)
        return (self.obs[idx], self.action[idx], self.reward[idx],
                self.next_obs[idx], self.next_mask[idx], self.done[idx])
