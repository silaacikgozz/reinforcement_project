# Role B \u2014 DDPG on DroneControl-v0 (continuous control sub-env)

Single drone, continuous 2D position, one delivery target per episode,
2-dim continuous action (speed, heading delta). Completely separate from
the `discrete/` folder \u2014 different env, different network shapes, no
shared state with Role A or the rest of Role B on purpose, so nothing here
can break anything else.

## Why this part is much lighter than everything else in this project
7-dim observation, 2-dim action, single agent, single target \u2014 no
combinatorial assignment, no action masking. Expect convergence (or a clear
failure mode) within tens of thousands of steps, not hundreds of thousands;
a full 3-seed run should take minutes, not hours, even on a laptop CPU.

## Status

| File | Purpose | Status |
|---|---|---|
| `networks.py` | `Actor` (per-dimension output activation matching the action box exactly), `Critic` | **DONE**, tested |
| `replay_buffer.py` | Continuous-action replay buffer (same pattern as Role A's, no action mask) | **DONE** |
| `agent.py` | `DDPGAgent`: target networks, Polyak soft updates, Gaussian exploration noise (decaying) | **DONE**, tested |
| `train.py` | Training loop, checkpoint/resume, custom quick-eval (success rate + return \u2014 no shipped baseline exists for this env) | **DONE**, smoke-tested (3000 steps \u00d7 3 seeds, ~30s total, no NaN) |
| `evaluate_agent.py` | Multi-seed evaluation (mean \u00b1 std success rate / return); no baseline table \u2014 none exists for this env | **DONE**, tested |

## How to run

```bash
cd code/role_b_policy/ddpg
python train.py --config ../../../configs/ddpg.yaml
python evaluate_agent.py --config ../../../configs/ddpg.yaml
```

## Engineering Notes — what broke and how it was fixed (real, not hypothetical)

**Symptom:** a first, vanilla-DDPG attempt (single critic, noise decaying to
0.05, 50k steps) produced `success_rate = 0.00` across all 25 in-training
evaluation checkpoints, despite `mean_return` looking reasonable at times
(e.g. \u2248 \u22125.5, which looked like progress but wasn't).

**Diagnosis:** stepping a trained actor through a real episode showed the
drone getting permanently stuck \u2014 position and heading frozen for the
entire 500-tick episode, paying a \u221225-per-tick "invalid move" penalty over
and over (`pos` started at a grid edge; a tiny outward action pushed the
next cell out of bounds, the env correctly rejects the move and leaves
`pos` unchanged, and the *same* rejected move is retried forever since the
observation never changes). Two compounding causes were identified:
1. `DroneControl-v0`'s observation has no boundary-distance feature \u2014 only
   `nearest_nofly_offset`, which is computed from actual no-fly cells, not
   the grid's outer edge. A policy genuinely cannot distinguish "near the
   edge" from "out in the open" from this observation alone.
2. `actor_loss` was growing steadily more negative over training (\u22120.5 \u2192
   \u22129), the DDPG-specific signature of a single critic overestimating Q \u2014
   the same mechanism Double DQN fixes for value-based methods, here making
   a do-nothing/stuck policy look falsely attractive, and exploration noise
   had decayed too low to ever perturb the agent back out once trapped.

**Fix applied (validated, not just theorized):** rewrote the agent with the
three TD3 fixes (Fujimoto et al., 2018) \u2014 twin critics with a `min(Q1,Q2)`
target, target policy smoothing, delayed actor/target updates \u2014 and raised
the exploration noise floor so it never decays low enough to trap the agent
again. Re-running with identical seeds and a real training session: where
the original run had **0/25 nonzero `success_rate` checkpoints**, the
TD3-stabilized version had **14/18 nonzero checkpoints** (0.20\u20130.40) over
the same step range. This is reported as a real before/after comparison in
the results section, not a hypothetical fix.

**What this is NOT:** a complete solution. `mean_return` is still volatile
(roughly \u22122200 to \u22125500 across checkpoints even after the fix) \u2014 the
boundary-blind-spot limitation in point 1 above is a property of the given
environment's observation design, not something a better algorithm alone
fully removes; TD3 fixed the *training pathology* (permanent collapse), not
the underlying partial observability. This distinction is worth keeping in
the report: the fix is real and measured, but it would be dishonest to claim
a fully solved navigation task.

## Design decisions log
- _Exploration noise:_ additive Gaussian, linearly decayed (`noise_start` \u2192
  `noise_end` over `noise_decay_steps`), not Ornstein-Uhlenbeck. Simpler, and
  adequate for a 2-dim action space with no strong temporal correlation
  requirement; OU's main advantage (smoothly correlated exploration over
  time, useful for physical momentum-driven control) is less relevant when
  heading/speed are re-chosen freely every step rather than integrated.
  Mention this explicitly in the report's method-origin note as a deviation
  from the textbook DDPG recipe, with the reasoning above.
- _Actor output activation:_ sigmoid for speed (matches `[0,1]` exactly),
  tanh for heading delta (matches `[-1,1]` exactly) \u2014 not a single tanh
  rescaled to both, which would otherwise waste capacity.
- _Success metric:_ `terminated and not truncated and soc > 0` \u2014 the env
  doesn't expose a direct "reached target" flag, so this infers it from the
  termination reason (battery depletion also sets `terminated=True`, so
  `soc > 0` disambiguates the two terminal cases).
- _No baseline comparison table:_ unlike Role A, there's no shipped
  random/heuristic baseline for `DroneControl-v0`. The deliverable for this
  sub-env is a stable, working DDPG with mean \u00b1 std across seeds \u2014 worth
  stating this explicitly in the report rather than leaving an unexplained
  gap next to Role A's baseline tables.
