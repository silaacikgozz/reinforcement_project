@echo off
cd /d "%~dp0"
echo Evaluating DDPG (mean +/- std over 3 seeds, no baseline table for this env)...
python evaluate_agent.py --config configs/ddpg.yaml
echo.
echo DONE. Check logs\ddpg_comparison.csv
pause
