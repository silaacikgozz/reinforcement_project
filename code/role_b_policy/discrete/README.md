# Role B \u2014 Discrete Policy-Gradient (REINFORCE+GAE \u2192 A2C)

Env: `DroneDispatch-v0` \u2014 SAME environment as Role A's DQN family (169 masked
discrete actions, 750-dim flattened state via `code/common/obs_encoding.py`).
This is intentional: it lets the team report directly compare value-based vs
policy-based methods on identical ground, and it means **everything we
learned debugging Role A's masking applies here too** \u2014 just in a different
place (logits, not Q-values).

## Why this is NOT a copy-paste of Role A
Masking a policy is different from masking a Q-network:
- DQN: mask Q-values to \u2212\u221e before argmax / before the target's max.
- Policy gradient: mask **logits** to a large negative number before the
  softmax, so illegal actions get \u2248 0 probability and contribute \u2248 0 to the
  log-prob used in the policy gradient. Getting this right matters even more
  here, because an un-masked illegal action with nonzero probability can
  actually be *sampled* and crash the step (or silently corrupt training if
  the env doesn't validate it) \u2014 there is no equivalent of "just don't pick
  the max", sampling is sampling.

## Plan

| File | Purpose | Status |
|---|---|---|
| `networks.py` | `PolicyValueNet`: shared trunk, masked-categorical policy head, scalar value head | **DONE**, tested |
| `gae.py` | GAE(\u03bb) advantage + return computation, given a trajectory of rewards/values/dones | **DONE**, tested |
| `agent.py` | `PGAgent` \u2014 same class drives both REINFORCE+GAE (full-episode rollout) and A2C (n-step rollout), selected via config | **DONE**, tested |
| `train.py` | Config-driven training loop, checkpoint/resume (same pattern as Role A) | **DONE**, smoke-tested (3000 steps \u00d7 3 seeds, both variants, no NaN) |
| `evaluate_agent.py` | Wraps trained policy for `drone_dispatch_env.evaluate()` vs baselines | **DONE**, tested end-to-end |
| `plot_curves.py` | Learning-curve PNGs | **Reuse Role A's** (`code/role_a_dqn/plot_curves.py`) \u2014 it only reads `logs/<run>_seed*.csv` by column name, no Role-A-specific imports. Call it with `--run_names reinforce_gae,a2c`. |

## How to run (after `pip install -r requirements.txt` at repo root)

```bash
cd code/role_b_policy/discrete
python train.py --config ../../../configs/reinforce_gae.yaml
python train.py --config ../../../configs/a2c.yaml
python train.py --config ../../../configs/a2c_ablation_noadvnorm.yaml

python evaluate_agent.py --config ../../../configs/reinforce_gae.yaml
python evaluate_agent.py --config ../../../configs/a2c.yaml
python evaluate_agent.py --config ../../../configs/a2c_ablation_noadvnorm.yaml

# reuse Role A's plotting script -- it's generic
python ../../role_a_dqn/plot_curves.py --run_names reinforce_gae,a2c,a2c_ablation_noadvnorm
```

Same interruption-safety as Role A: re-running the same command resumes from
`weights/<run>_seed<k>_ckpt.pt` instead of restarting.

## Empirical notes (3000-step smoke test, both variants, all 3 seeds)
No NaN, no crash, in either variant. One single A2C policy-loss spike
(\u2248 \u221260, one seed, one checkpoint) was observed and is suspected to come from
advantage normalization over a very small batch (`rollout_steps: 20`) \u2014 when
a 20-step window happens to have near-zero variance in its rewards, dividing
by a near-zero std amplifies noise. Gradient clipping kept it from
propagating into the weights. Worth checking whether this recurs in the full
200k-step run, and a natural angle for the GAE-\u03bb-vs-advantage-normalization
ablation discussion in the report.

## Required deliverables (Section 4 of the spec, Role B)
- [ ] REINFORCE + GAE running end-to-end
- [ ] A2C running end-to-end
- [ ] \u22653 seeds, mean \u00b1 std learning curves
- [ ] Ablation: GAE \u03bb sweep, OR advantage-normalization on/off
- [ ] Baseline table vs random / greedy_nearest / milp_rolling

## Design decisions log
- _Shared trunk vs separate policy/value networks:_ TBD
- _Entropy bonus (encourages exploration, common in A2C):_ TBD \u2014 worth trying given Role A's no-op collapse finding; entropy regularization is a natural counter to a policy collapsing onto one safe action.
- _GAE \u03bb default:_ TBD
- _Masking implementation:_ mask logits to a large negative constant (mirroring `agent.py`'s `_MASK_NEG` from Role A) before softmax, both when sampling actions and when computing log-probs for the gradient.
