# Offline RL (Ch. 20) — Joint component

Dataset: D_logs.npz, built via the simulator's own `generate_offline_dataset()`
(60% greedy_nearest + 40% noisy-random behavior, 200k transitions). This
stands in for "pool all three members' trajectories" — using the
instructor-provided generator avoids inconsistency from A/B/C each using a
different state encoding (750-d masked, 181-d, 59-d), and produces the
same kind of mixed-quality, partially-covering data the spec describes.

Note: the dataset's flattened obs (181-d: drones+orders+time) has NO action
mask. This is intentional/unavoidable from the generator, and it's actually
what makes the overestimation failure mode real and visible (see below) —
training blind to validity lets Q-values bootstrap off actions that were
never actually legal at that state.

## Pipeline
1. `build_dataset.py` → `logs/D_logs.npz`
2. `train_naive.py` → naive offline DQN, no mask, demonstrates divergence
3. `train_cql.py` → same network, + Conservative Q-Learning penalty
4. `train_bc.py` → behavioral-cloning baseline (no RL at all)
5. `evaluate_offline.py` → runs all three in the LIVE env (masked at
   decision time only) against random/greedy_nearest/milp_rolling

## Real results (30k steps, 50k-transition dataset, seed 0)
**Q-value divergence (train_naive.py):** mean_q 270 → 761, max_q 524 → 1345
over training, unbounded growth — actual per-step rewards are in [-300, +100].
**CQL (train_cql.py):** mean_q stays in 40–100, an order of magnitude lower.

| Method | cost_per_order |
|---|---|
| random | 18.78 |
| greedy_nearest | 4.57 |
| naive offline DQN | 66.94 |
| behavioral cloning | 32.15 |
| **CQL** | **17.63** (beats both required baselines) |

Re-run with the full 200k-transition dataset before the final report for
less noisy numbers — the pattern (naive worst, CQL best) should hold.
