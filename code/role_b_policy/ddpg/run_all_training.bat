@echo off
cd /d "%~dp0"

echo ============================================
echo  Role B (DDPG) - DroneControl-v0
echo  This is FAST (minutes, not hours) -- much lighter than the other parts.
echo ============================================

python train.py --config ../../../configs/ddpg.yaml
if errorlevel 1 (
  echo Training reported an error -- re-run this script, it resumes from checkpoint.
) else (
  echo DDPG training finished.
)
pause
