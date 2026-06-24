"""Shared helpers for turning DroneDispatch-v0's Dict observation into a flat
vector usable by an MLP (DQN, A2C, etc.). Anyone on the team can import this so
state encoding stays consistent across roles.

obs (gym Dict) has keys: drones (n_drones,10), orders (k_max,5), grid (H,W),
time (1,), action_mask (n_actions,). See Simulator Spec Section 12 / README.

Design decisions (logged here, not just in code, so we can defend them):

1. Normalization. Positions (x, y) range [0, max(H,W)-1] -> divided by
   max(H,W) so they sit in [0,1), matching the already-normalized fields
   (soc, time, one-hot flags). Unnormalized large-magnitude inputs (raw
   pixel-like coords up to 19) next to 0/1 flags is a classic cause of
   slow/unstable training -- one input dimension dominates the gradient.
2. Grid. We feed the raw (H, W) grid, flattened and divided by 3.0 (max cell
   code), as the simplest *correct* starting point. It's redundant across
   steps within an episode (the grid is fixed at reset) but the MLP has no
   other way to "see" no-fly geometry. If training is too slow once we have
   numbers, the first optimization to try is compressing this (e.g. local
   patch around each drone) rather than touching anything else.
3. Action mask IS included as an input feature (not just used to mask the
   output). Rationale: it tells the network which (drone, order) pairs are
   even legal *right now* -- e.g. drone status, without it the net would
   have to re-derive "is this drone idle" indirectly from the one-hot status
   field across all 8 drones. Including it costs us 169 extra input dims,
   which is cheap, and it is NOT a substitute for masking the Q-values
   at decision time and at target-computation time -- that masking still
   happens explicitly in agent.py.

Final vector layout (fixed order, do not change without re-checking
agent.py's assumptions): [drones_flat | orders_flat | grid_flat | time | mask]
"""
from __future__ import annotations

import numpy as np


def flat_dim(cfg) -> int:
    """Size of the vector flatten_obs() produces, for sizing the first
    network layer without having to call flatten_obs on a dummy obs."""
    n_drone_feats = 4 + 5 + 1          # x,y,soc,lost + 5 status one-hot + has_order  (= cfg's N_STATUS=5)
    return (cfg.n_drones * n_drone_feats
            + cfg.k_max * 5
            + cfg.H * cfg.W
            + 1
            + cfg.n_actions)


def flatten_obs(obs: dict, cfg) -> np.ndarray:
    """obs -> float32 vector, see module docstring for layout/rationale."""
    scale = float(max(cfg.H, cfg.W))

    drones = np.asarray(obs["drones"], dtype=np.float32).copy()
    drones[:, 0] /= scale   # x
    drones[:, 1] /= scale   # y
    # soc (col 2), lost flag (col 3), status one-hot, has_order flag: already in [0,1]

    orders = np.asarray(obs["orders"], dtype=np.float32).copy()
    orders[:, 0:4] /= scale                  # ox, oy, dx, dy
    orders[:, 4] = orders[:, 4] / float(cfg.sla_steps)  # waiting time, scaled by SLA window

    grid = np.asarray(obs["grid"], dtype=np.float32).reshape(-1) / 3.0  # cell codes are 0..3

    time = np.asarray(obs["time"], dtype=np.float32).reshape(-1)        # already in [0,1]

    mask = np.asarray(obs["action_mask"], dtype=np.float32).reshape(-1)  # already 0/1

    vec = np.concatenate([drones.reshape(-1), orders.reshape(-1), grid, time, mask])
    return vec.astype(np.float32)
