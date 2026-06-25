@echo off
cd /d "%~dp0"

echo [1/4] Evaluating REINFORCE+GAE...
python evaluate_agent.py --config configs/reinforce_gae.yaml

echo.
echo [2/4] Evaluating A2C...
python evaluate_agent.py --config configs/a2c.yaml

echo.
echo [3/4] Evaluating Ablation...
python evaluate_agent.py --config configs/a2c_ablation_noadvnorm.yaml

echo.
echo [4/4] Generating graphs (reusing Role A's plotting script)...
python ../../role_a_dqn/plot_curves.py --run_names a2c_noadvnorm
python ../../role_a_dqn/plot_curves.py --run_names reinforce_gae,a2c

echo.
echo DONE. Check the logs\ folder for *_comparison.csv and *_curve.png / all_methods_overlay.png
pause
