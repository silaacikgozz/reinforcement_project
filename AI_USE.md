# AI Tool Use Declaration

Per the project spec (Section 2): graded for honesty, not penalized for use. Every
teammate logs their own usage below. Be specific — "used Claude for X" not "used AI".

## Kerem Ayyılmaz — Role A (Value-based / DQN family)

| Stage | What Claude was used for | What was verified independently |
|---|---|---|
| Setup | Read the simulator's source (`env_dispatch.py`, `agent_interface.py`, `config.py`) together to understand the obs/action/reward API before writing any policy code; planned the repo skeleton matching Section 6 of the spec. | Ran the shipped test suite and `reproduce.sh` against the baselines myself before writing any training code. |
| State encoding | Designed `code/common/obs_encoding.py` (flattening the Dict obs to a 750-d vector) with Claude. | Verified output shape and NaN-freedom against the real env directly. |
| DQN/Double/Dueling | Implemented `networks.py`, `agent.py`, `train.py` with Claude. An early version had a masking bug (mask applied at decision time but not when computing the TD target), which produced a DQN that performed worse than random — diagnosed this together by re-deriving the Bellman target step by step. | Ran the actual 200k-step, 3-seed training myself; the loss/eval curves and the divergence pattern in the report are from my own runs, not generated text. |
| Ablation, evaluation, reporting | Designed the target-network on/off ablation, `evaluate_agent.py`, `plot_curves.py`, and the individual Word report with Claude. | All numbers in the report (70.98±22.72, 26.83±2.36, 26.17±1.24, 228.53±232.50) are copied from my own terminal output / `*_comparison.csv` files, not invented. |

## Nisa Üstünakın — Role B (Policy-based)

| Stage | What Claude was used for | What was verified independently |
|---|---|---|
| REINFORCE+GAE / A2C | Implemented `code/role_b_policy/discrete/` (shared `PolicyValueNet`, GAE computation, the rollout-length distinction between the two methods) with Claude, reusing Role A's obs encoding and masking pattern by design. | Ran the 60k-step, 3-seed training and the advantage-normalization ablation myself. |
| DDPG → TD3 | A first DDPG attempt on `DroneControl-v0` converged to a degenerate, permanently-stuck policy (success rate 0.00 across every checkpoint). Diagnosed the cause with Claude by stepping a trained actor through an episode (found it pinned at the grid boundary, repeating a rejected move) and identified single-critic overestimation as the likely mechanism. Implemented the TD3 fix (twin critics, target policy smoothing, delayed updates) with Claude. | Re-ran the exact same seeds after the fix myself; the 0.00 → 0.70±0.08 success-rate improvement is a measured before/after comparison from my own two training runs, not a claimed result. |
| Reporting | Individual Word report drafted with Claude using my own logged numbers and PNGs. | Reviewed the report against my own terminal output before submission. |

## Sıla Açıkgöz — Role C (Planning / Dyna-Q)

| Stage | What Claude was used for | What was verified independently |
|---|---|---|
| Dyna-Q | Implemented `code/role_c_planning/` with Claude: a learned dynamics model (`model.py`) plus a `DynaQAgent` that subclasses Role A's `DQNAgent` unchanged and adds model training + planning steps, to keep the comparison to plain DQN as close to apples-to-apples as possible. | Ran the 60k-step, 3-seed training for both `planning_steps=2` and the `planning_steps=0` ablation myself. |
| Ablation finding | Discussed with Claude why planning underperformed the no-planning ablation (63.97±3.71 vs. 30.06±9.79) — attributed to the dynamics model's prediction error on the full 750-d state being large enough, this early in training, that simulated transitions were not yet trustworthy. | This is reported as a genuine negative result from my own runs, not adjusted after the fact to look better. |
| Reporting | Individual Word report drafted with Claude using my own logged numbers and PNGs. | Reviewed against my own terminal output before submission. |

## Joint — Offline RL (Ch. 20) & Multi-Agent (Ch. 21)

| Stage | What Claude was used for | What was verified independently |
|---|---|---|
| Offline RL | Built `code/offline_rl/` with Claude: dataset generation via the simulator's own `generate_offline_dataset()`, a naive offline DQN (no action mask, to demonstrate Q-value divergence), a CQL fix, and a behavioral-cloning baseline. | Ran the full 200k-transition dataset build and all three training runs myself; the divergence curve (naive mean-Q growing past 650 vs. CQL staying under 120) and the final cost_per_order numbers (54.94 / 30.70 / 16.87) are from my own logs. |
| Multi-Agent | Built `code/multi_agent/` with Claude: IDQN with full parameter sharing, reusing Role A's `QNetwork`/`DQNAgent` unmodified. An initial version ran at ~4 steps/s on my machine; Claude identified the cause (8 separate per-agent forward passes instead of one batched call) and fixed it. | Ran the training and the head-to-head evaluation against the centralized Dueling DQN myself; the −754.02±317.70 vs. −680.67 comparison is from my own run. |
