@echo off
cd /d "%~dp0"

echo ============================================
echo  IE306 Role A - DQN family - FULL OVERNIGHT RUN
echo  This runs 4 trainings back-to-back (3 seeds each, 200k steps each).
echo  Do NOT close this window. Leave the PC plugged in, sleep disabled.
echo ============================================
echo.

echo [1/4] Standard DQN...
python train.py --config ../../configs/dqn.yaml
if errorlevel 1 goto :error

echo.
echo [2/4] Double DQN...
python train.py --config ../../configs/double_dqn.yaml
if errorlevel 1 goto :error

echo.
echo [3/4] Dueling DQN...
python train.py --config ../../configs/dueling_dqn.yaml
if errorlevel 1 goto :error

echo.
echo [4/4] Ablation (target network OFF)...
python train.py --config ../../configs/dqn_ablation_notarget.yaml
if errorlevel 1 goto :error

echo.
echo ============================================
echo  ALL 4 TRAININGS FINISHED. Safe to close.
echo ============================================
pause
exit /b 0

:error
echo.
echo ============================================
echo  SOMETHING FAILED -- read the error above.
echo  Just re-run this script: completed/partial runs resume from checkpoint
echo  automatically, they will NOT restart from zero.
echo ============================================
pause
exit /b 1