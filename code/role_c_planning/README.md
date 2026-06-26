ROLE C - Dyna-Q (Ch.18). Reuses Role A's QNetwork/DQNAgent/replay_buffer/obs_encoding directly (imported, not copied).
model.py: learned dynamics model. dyna_agent.py: DynaQAgent(DQNAgent) + planning(). train.py/evaluate_agent.py: same pattern as Role A.
Ablation: planning_steps n=5 vs n=0 (configs/dyna_q.yaml vs dyna_q_ablation_n0.yaml).
Note: ~6x slower than plain DQN per step (1 real + 1 model-train + 5 simulated Q-updates per env step).
