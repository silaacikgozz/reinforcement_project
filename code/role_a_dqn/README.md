# Role A — Value-based dispatcher (DQN → Double DQN → Dueling DQN)

Owner: Ömer Faruk Kul. Env: `DroneDispatch-v0` (discrete, masked, 169 actions).

## Plan (files to be filled in, in this order)

| File | Purpose | Status |
|---|---|---|
| `../common/obs_encoding.py` | Dict obs -> flat vector for the Q-network | **DONE** (dim=750, tested against real env, no NaN) |
| `networks.py` | `QNetwork` (vanilla MLP) and `DuelingQNetwork` (value + advantage streams) | **DONE** (forward-pass tested, output shape (B,169)) |
| `replay_buffer.py` | Fixed-size replay buffer, uniform sampling | **DONE** |
| `agent.py` | `DQNAgent`: epsilon-greedy `act()` w/ action masking, `learn()` step (masked target, Huber loss, grad clip), target-network sync, Double DQN flag | **DONE** — masking bug from the earlier session is fixed here (mask applied to the target computation too) |
| `train.py` | Config-driven training loop w/ checkpoint/resume (`checkpoint_every`), per-seed CSV logs, periodic in-training eval | **DONE** |
| `evaluate_agent.py` | Wraps trained weights as `agent_interface.Policy`; runs official `evaluate()` vs random/greedy_nearest/milp_rolling; writes `logs/<run>_comparison.csv` | **DONE**, tested |
| `plot_curves.py` | Learning-curve PNGs (mean±std over seeds, baseline reference lines) | **DONE**, tested |
| `ablation.py` | Compares `configs/dqn.yaml` (target on) vs `configs/dqn_ablation_notarget.yaml` (target off) | TODO — just runs train.py twice + evaluate_agent.py twice + a diff table; can also be done by hand with the commands below |

## How to run this on your own machine (after `pip install -r requirements.txt` at repo root)

```bash
cd code/role_a_dqn

# 1. Train each variant (3 seeds each, ~200k steps). This WILL take a while
#    on CPU -- budget real time, don't rush it (see empirical notes below for
#    why rushing produces exactly the instability we're trying to avoid).
python train.py --config ../../configs/dqn.yaml
python train.py --config ../../configs/double_dqn.yaml
python train.py --config ../../configs/dueling_dqn.yaml

# 2. Ablation arm (target network OFF)
python train.py --config ../../configs/dqn_ablation_notarget.yaml

# If a run gets interrupted (Ctrl+C, machine sleeps, crash): just re-run the
# exact same command. It resumes from weights/<run>_seed<k>_ckpt.pt
# automatically (checkpointed every 10k steps) instead of restarting.

# 3. Evaluate each against the baselines (writes logs/<run>_comparison.csv)
python evaluate_agent.py --config ../../configs/dqn.yaml
python evaluate_agent.py --config ../../configs/double_dqn.yaml
python evaluate_agent.py --config ../../configs/dueling_dqn.yaml
python evaluate_agent.py --config ../../configs/dqn_ablation_notarget.yaml

# 4. Plot learning curves (saves PNGs into logs/)
python plot_curves.py --run_names dqn,double_dqn,dueling_dqn
```

## Empirical notes (real data, from a 160k-step diagnostic run of plain DQN, seed 0)

`eval_cost_per_order` over training: 95 → 112 → 98 → ... → **31** (step 80k, the
best point) → fluctuates 40-70 → **220** (step 160k, sudden blow-up).

This is the textbook DQN overestimation-divergence pattern: a real
improvement phase, followed by Q-values feeding back on themselves and the
policy collapsing. **This is exactly the failure Double DQN exists to fix** --
don't be surprised if plain DQN's curve looks unstable; that instability,
correctly diagnosed, IS the point of the chapter, and exactly what the
ablation/method-origin section of the report should discuss. Do not chase a
"clean" DQN curve by accident-fixing hyperparameters without understanding
why it diverged -- the divergence itself is reportable evidence.

If, after real training, your method still doesn't beat `greedy_nearest`:
that is an acceptable outcome per the spec ("...or honestly diagnose why it
doesn't") -- greedy_nearest is a very strong baseline here (it nearly matches
milp_rolling), and a flat-MLP DQN losing to a problem-specific heuristic with
full map information is a legitimate, explainable result.

## Design decisions log (fill in as we make them, for the oral defense + report)
- _Obs encoding choice:_ flatten everything (drones + orders + grid/3 + time + action_mask) into one 750-dim vector, positions normalized by max(H,W). Action mask included as an *input feature* too (not just for output masking) so the net can see which (drone,order) pairs are legal without re-deriving it.
- _Action masking implementation (act-time and target-time):_ both done in `agent.py` — `greedy_action`/`act` mask before argmax; `learn` masks `Q(s',·)` (both the online argmax for Double DQN and the plain max) before bootstrapping. This was the suspected root cause of the earlier session's DQN-worse-than-random result.
- _Network sizes:_ hidden_sizes=[256,256] (config-driven, `configs/dqn.yaml`), plain MLP — no CNN over the grid (noted as a limitation, not hidden).
- _Double DQN target formula used:_ action selected by online net, value read from target net (`agent.py: learn()`, van Hasselt et al. 2016).
- _Dueling aggregation (mean vs subtract-max):_ mean-subtraction, `Q = V + (A - mean(A))`, per Wang et al. 2016 — more stable than max-subtraction for identifiability between V and A.
- _Stability measures added after observing the earlier session's instability:_ Huber loss (not MSE), gradient norm clipping, checkpoint/resume so long runs survive interruption.
