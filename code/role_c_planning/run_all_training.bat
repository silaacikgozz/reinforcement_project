@echo off
cd /d "%~dp0"
echo Role C (Dyna-Q) - this is SLOWER than DQN (~6x, due to planning steps). Budget hours.
call :run "../../configs/dyna_q.yaml" "1/2 Dyna-Q (n=5)"
call :run "../../configs/dyna_q_ablation_n0.yaml" "2/2 Ablation (n=0)"
pause
exit /b 0
:run
set CFG=%~1
set LABEL=%~2
set N=0
:loop
python train.py --config %CFG%
if not errorlevel 1 exit /b 0
set /a N+=1
echo [%LABEL%] attempt %N% failed, retrying...
if %N% LSS 3 goto loop
exit /b 1
