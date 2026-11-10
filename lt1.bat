@echo off
setlocal
cd /d "%~dp0"
title LT-1 Control Panel

:MENU
cls
echo ======================================================
echo                LT-1 Control Panel
echo ======================================================
echo [1] Monitor Live Training Logs (train.log)
echo [2] Monitor GPU VRAM and Temperature (nvidia-smi)
echo [3] Generate Text from Latest Checkpoint
echo [4] Run Unit Test Suite
echo [5] Start New Base Pre-Training
echo [6] Run Smoke Pre-Training (50 steps)
echo [0] Exit
echo ======================================================
set /p CHOICE="Select an option [0-6]: "

if "%CHOICE%"=="1" goto MONITOR
if "%CHOICE%"=="2" goto GPU
if "%CHOICE%"=="3" goto GENERATE
if "%CHOICE%"=="4" goto TESTS
if "%CHOICE%"=="5" goto TRAIN
if "%CHOICE%"=="6" goto SMOKE
if "%CHOICE%"=="0" goto EXIT
goto MENU

:MONITOR
call monitor.bat
goto MENU

:GPU
cls
echo ======================================================
echo             NVIDIA GPU Status Monitor
echo ======================================================
nvidia-smi -l 2 --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu --format=csv
pause
goto MENU

:GENERATE
call generate.bat
goto MENU

:TESTS
cls
echo ======================================================
echo             Running Unit Test Suite
echo ======================================================
"C:\Users\LTadmin\AppData\Local\Programs\Python\Python311\python.exe" -m unittest discover -s tests -t . -v
pause
goto MENU

:TRAIN
call train.bat
goto MENU

:SMOKE
cls
echo ======================================================
echo             Running Smoke Verification
echo ======================================================
"C:\Users\LTadmin\AppData\Local\Programs\Python\Python311\python.exe" -m src.train --stage smoke --steps 50 --batch-size 4 --seq-len 64
pause
goto MENU

:EXIT
exit /b 0
