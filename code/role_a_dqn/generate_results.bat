@echo off
cd /d "%~dp0"

echo ============================================
echo  Generating baseline tables + graphs from trained weights.
echo  Run this AFTER run_all_training.bat has finished (or partially
echo  finished -- it'll just skip whatever isn't trained yet).
echo ============================================
echo.

echo [1/5] Evaluating Standard DQN vs baselines...
python evaluate_agent.py --config configs/dqn.yaml

echo.
echo [2/5] Evaluating Double DQN vs baselines...
python evaluate_agent.py --config configs/double_dqn.yaml

echo.
echo [3/5] Evaluating Dueling DQN vs baselines...
python evaluate_agent.py --config configs/dueling_dqn.yaml

echo.
echo [4/5] Evaluating Ablation (target network OFF) vs baselines...
python evaluate_agent.py --config configs/dqn_ablation_notarget.yaml

echo.
echo [5/5] Generating graphs...
python plot_curves.py --run_names dqn_ablation_notarget
python plot_curves.py --run_names dqn,double_dqn,dueling_dqn

echo.
echo ============================================
echo  DONE. Open the logs\ folder:
echo    dqn_curve.png              - Standard DQN learning curve vs baselines
echo    double_dqn_curve.png       - Double DQN learning curve vs baselines
echo    dueling_dqn_curve.png      - Dueling DQN learning curve vs baselines
echo    dqn_ablation_notarget_curve.png  - ablation arm (target net OFF)
echo    all_methods_overlay.png    - all 3 main variants overlaid (the "summary" graph)
echo    *_comparison.csv           - the numeric baseline-comparison tables
echo ============================================
pause