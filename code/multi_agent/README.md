# Multi-Agent (Ch. 21) — Joint component

IDQN with full parameter sharing on `DroneDispatchMA-v0`: every drone
acts via the SAME `QNetwork` (reused from Role A unchanged), and every
agent's transitions land in one shared replay buffer. Minimum bar per
spec ("runs end-to-end, compared head-to-head, non-stationarity discussed")
— full convergence is an explicit stretch goal, not a requirement.

## Non-stationarity discussion (for the report)
Even with parameter sharing, IDQN violates the stationarity assumption
single-agent Q-learning convergence relies on: from any one drone's local
view, the "environment" includes the other drones' evolving policies (all
driven by the *same* shared weights, updated from everyone's experience
simultaneously). A gradient step taken because of drone_3's transition
changes the Q-values drone_7 will act on next, even though drone_7's own
local state didn't change — the target each agent regresses toward keeps
shifting for reasons outside its own MDP. This is a structural source of
extra variance on top of everything Role A already found unstable about
DQN in this environment, not a separate, fixable bug.

## Head-to-head (evaluate_agent.py)
Compares IDQN's mean team `episode return` against the centralized
Dueling DQN's `episode_return` on the same reward weights — same units,
different credit-assignment structure (centralized: one stream; IDQN: summed
across 8 independently-acting, shared-weight agents).

Run after both Role A (dueling_dqn) and this are trained:
```
python evaluate_agent.py --config ../../configs/idqn.yaml
```
