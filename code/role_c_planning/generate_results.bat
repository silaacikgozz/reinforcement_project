@echo off
cd /d "%~dp0"
python evaluate_agent.py --config configs/dyna_q.yaml
python evaluate_agent.py --config configs/dyna_q_ablation_n0.yaml
python ../role_a_dqn/plot_curves.py --run_names dyna_q,dyna_q_n0
python plot_reward_curves.py --run_name dyna_q
python plot_reward_curves.py --run_name dyna_q_n0
pause
