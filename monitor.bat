@echo off
setlocal
cd /d "%~dp0"
title LT-1 Training Monitor

echo ======================================================
echo                LT-1 Training Monitor
echo ======================================================
echo Streaming live training logs from train.log...
echo Press Ctrl+C to stop monitoring.
echo.

if not exist "train.log" (
    echo [WARNING] train.log does not exist yet. Waiting for training process...
)

powershell -NoExit -Command "Get-Content -Path 'train.log' -Wait -Tail 30"
