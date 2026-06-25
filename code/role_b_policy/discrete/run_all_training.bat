@echo off
cd /d "%~dp0"

echo ============================================
echo  Role B (discrete) - REINFORCE+GAE, A2C, ablation
echo  Unattended-safe: retries via checkpoint, never blocks on a keypress.
echo ============================================

call :run_with_retry "../../../configs/reinforce_gae.yaml" "1/3 REINFORCE+GAE"
call :run_with_retry "../../../configs/a2c.yaml" "2/3 A2C"
call :run_with_retry "../../../configs/a2c_ablation_noadvnorm.yaml" "3/3 Ablation (no advantage norm)"

echo.
echo ALL 3 STAGES ATTEMPTED. Check logs\*.csv and weights\*.pt.
pause
exit /b 0

:run_with_retry
set CFG=%~1
set LABEL=%~2
set ATTEMPTS=0
echo.
echo [%LABEL%] starting...
:retry_loop
python train.py --config %CFG%
if not errorlevel 1 (
    echo [%LABEL%] finished OK.
    exit /b 0
)
set /a ATTEMPTS+=1
echo   [%LABEL%] attempt %ATTEMPTS% failed -- retrying (resumes from checkpoint)...
if %ATTEMPTS% LSS 3 goto :retry_loop
echo   [%LABEL%] WARNING: failed %ATTEMPTS% times. Moving on.
exit /b 1
