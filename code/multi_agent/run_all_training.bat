@echo off
cd /d "%~dp0"
python train.py --config ../../configs/idqn.yaml
echo.
echo Head-to-head vs centralized Dueling DQN (needs weights/dueling_dqn_seed0.pt from Role A):
python evaluate_agent.py --config ../../configs/idqn.yaml
pause
