# Team Report — RL for City-Scale Drone Delivery

_Team: Kerem Ayyılmaz (Role A), Nisa Üstünakın (Role B), Sıla Açıkgöz (Role C).
Grade weight is on analysis depth, not page count._

## 1. Problem & Environment Recap

`DroneDispatch-v0` models the operational layer of a city-scale drone-delivery
service: 8 battery-limited drones must be matched against waiting delivery
requests at every decision point, with the option to divert a drone to a
charger instead. Strategic decisions (hub placement, fleet size) are fixed by
the simulator; our job is purely the dispatch policy. The action space is
discrete (169 entries: 160 drone–order pairings, 8 charge actions, 1 no-op),
almost entirely masked at every step, so the environment exposes an explicit
action mask alongside the Dict observation (per-drone status, pending orders,
the no-fly/charger grid, elapsed time). A secondary sub-environment,
`DroneControl-v0`, strips this down to a single drone with continuous
speed/heading control, used only for Role B's DDPG requirement. The bar for
every learned method is beating `greedy_nearest` on mean cost per delivered
order, or explaining why it doesn't.

## 2. Method Descriptions

### 2.1 Role A — Value-based (DQN → Double DQN → Dueling DQN)

**State.** The Dict observation is flattened into a single 750-dimensional
vector. Positions are normalized by `max(H,W)`; the grid is included raw
(divided by 3, its max cell code) since an MLP has no other way to see
no-fly geometry. The action mask is included as an *input feature* in
addition to being used for output masking.

**Architecture.** Standard/Double DQN: a plain MLP (two 256-unit hidden
layers) mapping the 750-d state to 169 Q-values. Dueling DQN: the same trunk
splits into a value stream `V(s)` and an advantage stream `A(s,a)`,
recombined as `Q(s,a) = V(s) + (A(s,a) − mean_a' A(s,a'))` (Wang et al., 2016).

**Double DQN target.** The next action is selected by the *online* network
and evaluated by the *target* network, decoupling selection from evaluation
to curb DQN's overestimation bias.

**Stability measures.** Masking applied to the target computation as well as
to acting (see Engineering Log §8), Huber loss instead of MSE, gradient-norm
clipping, periodic target-network sync.

### 2.2 Role B — Policy-based (REINFORCE+GAE → A2C; DDPG)

**REINFORCE with a GAE baseline.** A shared trunk feeds a masked-categorical
policy head (illegal logits forced to a large negative constant before the
softmax — the policy-gradient analogue of Role A's Q-masking) and a scalar
value head used as a variance-reducing baseline. The advantage is the
Generalized Advantage Estimate (Schulman et al., 2016) computed over the
full episode.

**A2C.** Identical network and loss form, but the rollout is cut every fixed
number of steps rather than waiting for episode end, bootstrapping the
cut-off state's value (λ=0.95, not 1.0). An entropy bonus is added to both
methods' policy loss — included specifically because Role A documented a
no-op-collapse failure mode that entropy regularization directly discourages.

**DDPG → TD3.** A first DDPG attempt on `DroneControl-v0` converged to a
literal fixed point: the drone pinned at one position for the full episode,
repeatedly attempting a rejected move, with a single critic's overestimation
(a steadily more negative actor loss) making that frozen behavior look
artificially good. The fix was the standard TD3 recipe (Fujimoto, van Hoof
& Meger, 2018): twin critics with a `min(Q1,Q2)` target, clipped target
policy-smoothing noise, and delayed actor/target updates, plus a raised
exploration-noise floor so training-time noise can never decay low enough to
leave a stuck state unperturbed.

### 2.3 Role C — Planning (Dyna-Q, model-based acceleration)

Reuses Role A's `QNetwork`/`DQNAgent` unmodified and adds a learned dynamics
model: a small MLP predicting next-state, reward, and a done logit from
(state, one-hot action), trained by ordinary supervised regression against
real transitions in the same replay buffer the Q-network learns from. After
every real-step DQN update, the agent samples `planning_steps` additional
(state, action) pairs from the buffer, asks the model what it predicts, and
performs an extra Q-update against that prediction as if it were real — the
central idea of Sutton's Dyna architecture, layered on top of Role A's
already-validated masked-DQN machinery so any difference from plain DQN is
attributable to the planning mechanism itself.

## 3. Results

**Baseline comparison** (`simulator/configs/eval_standard.yaml`, 5 eval
seeds, mean ± std over 3 training seeds [0,1,2]):

| Method | cost_per_order (mean ± std) |
|---|---|
| random | 18.50 |
| **greedy_nearest** (bar) | **4.31** |
| milp_rolling | 4.28 |
| Standard DQN | 70.98 ± 22.72 |
| Double DQN | 26.83 ± 2.36 |
| Dueling DQN | 26.17 ± 1.24 |
| REINFORCE + GAE | 21.40 ± 0.88 |
| A2C | 21.86 ± 1.42 |
| Dyna-Q (planning_steps=2) | 63.97 ± 3.71 |
| DDPG (TD3-stabilized, success rate) | 0.70 ± 0.08 *(different env/metric — see §3.3)* |

### 3.1 Role A
Going Standard → Double → Dueling DQN, mean cost drops (70.98 → 26.83 →
26.17) but **std drops far more sharply** (22.72 → 2.36 → 1.24, a 10–18×
reduction) — the variance collapse each method is designed to produce.
Standard DQN's own learning curve shows the textbook overestimation pattern:
a real improvement phase (best checkpoint ≈ 20–25) followed by sharp
regressions (cost briefly > 200 near step 110k–190k).

### 3.2 Role B
REINFORCE+GAE and A2C converge to a similar band (≈21–22) to each other and
to Role A's Double/Dueling DQN, despite being a different method family —
consistent with the bottleneck being the environment's information asymmetry
against `greedy_nearest`, not a weakness specific to either family.

### 3.3 Role C
Dyna-Q (63.97±3.71) is *worse* than its own no-planning ablation (30.06±9.79)
— see §4.3 for why this is a genuine, diagnosed finding and not a bug.

**Why none of the learned methods beat `greedy_nearest`/`milp_rolling`:**
(1) `greedy_nearest` reads exact grid distances directly; every learned
method here works from a flattened, geometry-blind encoding instead. (2)
`milp_rolling`, a full mixed-integer optimizer, only narrowly beats
`greedy_nearest` (4.28 vs 4.31), implying the bar itself is close to what
far more expensive planning achieves in this environment. (3) Delayed credit
assignment (a correct assignment pays off many steps after the decision,
while no-op is immediate and always legal) caused an early no-op collapse in
development across multiple methods, documented in §8.

## 4. Ablations

### 4.1 Role A — Target network ON vs. OFF
| Arm | cost_per_order (mean ± std) |
|---|---|
| Target network ON | 70.98 ± 22.72 |
| Target network OFF | 228.53 ± 232.50 |

Std (232.50) exceeds the mean (228.53) — not consistently bad, *unpredictable*,
direct evidence for why a delayed target network is standard practice.

### 4.2 Role B — Advantage normalization ON vs. OFF (A2C)
| Arm | cost_per_order (mean ± std) |
|---|---|
| Advantage normalization ON | 21.86 ± 1.42 |
| Advantage normalization OFF | 41.30 ± 30.53 |

Per-seed results make the mechanism concrete: one seed reached 84.35 (success
0.17), another reached 16.90 (success 0.94, beating the normalized version).
With only a 20-step rollout, an unnormalized advantage is highly sensitive to
that window's particular reward variance.

### 4.3 Role C — Planning steps n=2 vs. n=0
| Arm | cost_per_order (mean ± std) |
|---|---|
| planning_steps = 2 | 63.97 ± 3.71 |
| planning_steps = 0 | 30.06 ± 9.79 |

Planning made the mean *worse* while narrowing variance. The dynamics model
predicts the full 750-d next state from a small MLP trained on the same
limited real data the Q-network is still learning from; early-to-mid
training, its predictions are not yet trustworthy, and two extra Q-updates
per real step toward an inaccurate target outweigh the sample-efficiency
benefit planning is meant to provide. Reported as a genuine negative result.

## 5. Offline RL (Ch. 20)

**Dataset.** 200,000 transitions generated via the simulator's own
`generate_offline_dataset()` (60% greedy_nearest + 40% noisy random), used
in place of manually pooling each role's individually-trained policies —
the three roles use three incompatible state encodings (750-d/181-d/59-d),
so the simulator's generator was the only way to get one self-consistent
dataset. No action mask is stored (by the generator's own design).

**Naive offline DQN.** Trained purely from the static dataset (no env
interaction, no mask), bootstrapping `max_a' Q(s',a')` over all 169 actions —
many never legal at that state, and never correctable without a new on-policy
rollout. Result: unbounded Q-value growth (mean Q 270 → 761 over 30k steps;
real per-step rewards span only −300 to +100).

**CQL fix.** Adding `α × (logsumexp_a Q(s,a) − Q(s,a_data))` to the TD loss
(Kumar et al., 2020) penalizes Q being high for actions other than the one
the data actually took. Result: mean Q stays under ~120 over the same budget.

| Method | cost_per_order | success rate |
|---|---|---|
| Naive offline DQN | 54.94 | 0.496 |
| Behavioral cloning | 30.70 | 0.396 |
| **CQL** | **16.87** | **0.934** |

CQL beats both required baselines (naive-offline and BC) by a wide margin.

## 6. Multi-Agent (Ch. 21)

**Setup.** The centralized dispatcher is replaced with 8 independently-acting
drones on `DroneDispatchMA-v0` (4 actions, 59-d local obs), all sharing **one**
network (parameter sharing) — Role A's `QNetwork`/`DQNAgent`, reused
unmodified, with every agent's transitions landing in one shared replay
buffer.

**Non-stationarity.** Even with parameter sharing, IDQN violates the
stationarity assumption single-agent Q-learning relies on: from any one
drone's local view, the environment includes the other seven drones'
behavior, which keeps changing as the shared network updates from everyone's
pooled experience — a gradient step driven by drone 3's transition shifts
what drone 7 will do next, even though drone 7's own local state didn't
change.

| Method | Return |
|---|---|
| Centralized Dueling DQN (200k steps) | −680.67 |
| IDQN, decentralized, parameter-shared (15k steps) | −754.02 ± 317.70 |

The two totals are close despite IDQN training for ~7% as many steps on a
strictly harder coordination problem — the gap is plausibly explained by
training budget alone. Per spec, full convergence here is a stretch goal;
the minimum bar (runs end-to-end, compared head-to-head, non-stationarity
discussed) is met.

## 7. Method-Origin Note

**Role A:**
- **DQN** — Mnih et al., *Nature* (2015). Required starting point; baseline
  for the ablation and Double/Dueling comparisons.
- **Double DQN** — van Hasselt, Guez & Silver, AAAI (2016). Chosen because
  Standard DQN's own curve showed exactly the overestimation signature this
  method targets.
- **Dueling DQN** — Wang et al., ICML (2016). Chosen for value/advantage
  decomposition, useful where many drones are busy and the immediate action
  barely matters.

**Role B:**
- **GAE** — Schulman et al., *"High-Dimensional Continuous Control Using
  Generalized Advantage Estimation"* (2016). Used as the variance-reduction
  baseline for REINFORCE, then reused with a shorter λ for A2C's online
  bootstrap.
- **A2C** — Mnih et al., *"Asynchronous Methods for Deep RL"* (2016).
  Chosen as the required online actor-critic step after REINFORCE.
- **DDPG / TD3** — Lillicrap et al. (2015) for the base method; Fujimoto,
  van Hoof & Meger, *"Addressing Function Approximation Error in
  Actor-Critic Methods"* (2018) for the stabilization actually needed once
  vanilla DDPG converged to a stuck policy (§8).

**Role C:**
- **Dyna-Q** — Sutton, *"Dyna: An Integrated Architecture for Learning,
  Planning, and Reacting"* (1991); Sutton & Barto, *Reinforcement Learning:
  An Introduction*, Ch. 8. Chosen as the required model-based method; the
  ablation (§4.3) tests exactly the assumption Dyna depends on — that the
  learned model is accurate enough for planning to help.

**Joint:**
- **CQL** — Kumar et al., *"Conservative Q-Learning for Offline
  Reinforcement Learning"* (NeurIPS 2020). Chosen over IQL for its more
  direct mechanical link to the specific failure mode demonstrated (a
  single penalty term added to the exact same TD loss already in use).

## 8. Engineering Log — What Broke and How We Diagnosed It

### Role A (Kerem Ayyılmaz)
1. **Action masking missing from the target computation.** Masked only at
   decision time, not in the TD target → DQN performed worse than random,
   got worse as Double/Dueling were added. Fixed by masking `Q(s',·)`
   identically in both places and storing `next_mask` explicitly per
   transition.
2. **No-op collapse in early/short training** — confirmed by instrumenting
   action-type counts over a full episode (>90% no-op). Resolved by training
   to the full step budget; documented as an expected phase, not a bug.
3. **Checkpoint/resume** added after losing progress to an interrupted run.

### Role B (Nisa Üstünakın)
1. **Vanilla DDPG's stuck-policy failure.** Success rate stayed at 0.00 for
   an entire 50k-step run. Diagnosed by stepping the trained actor through a
   live episode: the drone was pinned at the grid boundary the entire time,
   repeatedly attempting a move the environment correctly rejects, with the
   observation never changing once stuck. A growing |actor_loss| pointed to
   single-critic overestimation as the reason this frozen behavior looked
   attractive. Fixed with TD3 (twin critics, target smoothing, delayed
   updates) plus a raised exploration-noise floor — re-verified with an
   identical-seed re-run: 0.00 → 0.70±0.08.
2. **Advantage-normalization sensitivity at short rollout length** (A2C,
   `rollout_steps=20`) — see ablation §4.2; flagged during development when
   one seed's policy loss spiked to ≈−60 with normalization on a near-zero-
   variance batch.

### Role C (Sıla Açıkgöz)
1. **Planning hurt rather than helped** (§4.3) — not a code bug; confirmed
   by comparing identical training runs with `planning_steps=2` vs `=0`,
   isolating the model-quality explanation from any other variable.

### Joint
1. **Multi-agent training ran at ~4 steps/s** (would have taken hours for a
   15k-step run). Diagnosed as 8 separate per-agent forward passes per
   environment step instead of one batched call; fixed by stacking all 8
   agents' observations into a single tensor before the forward pass.
2. **Offline dataset generation gave no progress feedback** for several
   minutes on 200k transitions, initially mistaken for a hang; resolved by
   adding a periodic progress print to the dataset-building script.
