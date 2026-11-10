@echo off
setlocal
cd /d "%~dp0"
title LT-1 1.00B Parameter Pre-Training and Conversational SFT

set "PYTHON_EXE=C:\Users\LTadmin\AppData\Local\Programs\Python\Python311\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo ======================================================
echo    LT-1 1.00B Parameter Pre-Training and Auto-SFT
echo ======================================================
echo Python: %PYTHON_EXE%
echo Directory: %CD%
echo Dataset: Cosmopedia v2 + Finance Alpaca + Daily Conversation
echo Target: 10,000 Pre-train steps + 1,000 SFT steps
echo Memory Guard: CPU-offloaded Engram tables with Factored Row-RMS
echo ======================================================
echo.

"%PYTHON_EXE%" -u -m src.train --stage pretrain --scale 1b --chain-sft --resume %*

pause
