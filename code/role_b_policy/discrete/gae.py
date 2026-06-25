"""Generalized Advantage Estimation (Schulman et al., 2016).

Both REINFORCE+GAE and A2C in this folder call the same function -- they
only differ in WHEN they call it (REINFORCE: once per finished episode,
lambda effectively averaging over the whole trajectory; A2C: every
`rollout_steps` environment steps, bootstrapping from the value of the
state the rollout was cut off at).
"""
from __future__ import annotations
import numpy as np


def compute_gae(rewards, values, dones, last_value: float, gamma: float, lam: float):
    """
    rewards, values, dones: 1-D sequences of length T (values[t] is V(s_t),
    the state BEFORE reward[t] was received). last_value is V(s_T), the
    value of the state one step after the last entry -- 0 if that state was
    terminal, otherwise the critic's bootstrap estimate.

    Returns (advantages, returns), both length T, both np.float32 arrays.
    returns[t] = advantages[t] + values[t]  (used as the value-loss target).
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(T)):
        not_done = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns
