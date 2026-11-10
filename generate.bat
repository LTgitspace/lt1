@echo off
setlocal
cd /d "%~dp0"
title LT-1 Interactive Text Generation

set "PYTHON_EXE=C:\Users\LTadmin\AppData\Local\Programs\Python\Python311\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

set "CKPT=checkpoints\latest.pt"
if not "%~1"=="" set "CKPT=%~1"

if not exist "%CKPT%" (
    echo [ERROR] Checkpoint not found: %CKPT%
    pause
    exit /b 1
)

"%PYTHON_EXE%" -u -m src.generate --checkpoint "%CKPT%" --interactive --loops 2 --max-tokens 80 --temperature 0.7
pause
