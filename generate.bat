@echo off
setlocal
cd /d "%~dp0"
title LT-1 Text Generation

set "PYTHON_EXE=C:\Users\LTadmin\AppData\Local\Programs\Python\Python311\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

set "CKPT=checkpoints\latest.pt"
if not "%~1"=="" set "CKPT=%~1"

echo ======================================================
echo                LT-1 Text Generation
echo ======================================================
echo Checkpoint: %CKPT%
echo.

if not exist "%CKPT%" (
    echo [ERROR] Checkpoint not found: %CKPT%
    echo Wait until training saves checkpoints to checkpoints\latest.pt
    pause
    exit /b 1
)

set /p PROMPT="Enter prompt (default: Once upon a time): "
if "%PROMPT%"=="" set "PROMPT=Once upon a time"

set /p LOOPS="Enter recurrent loops [1, 2, or 4] (default: 2): "
if "%LOOPS%"=="" set "LOOPS=2"

echo.
echo Generating text...
echo ------------------------------------------------------
"%PYTHON_EXE%" -m src.generate --checkpoint "%CKPT%" --prompt "%PROMPT%" --loops %LOOPS% --max-tokens 50 --temperature 0.7
echo ------------------------------------------------------
echo.

pause
