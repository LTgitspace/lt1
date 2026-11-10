@echo off
setlocal
cd /d "%~dp0"
title LT-1 Pre-Training Runner

set "PYTHON_EXE=C:\Users\LTadmin\AppData\Local\Programs\Python\Python311\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo ======================================================
echo                LT-1 Pre-Training Runner
echo ======================================================
echo Python: %PYTHON_EXE%
echo Directory: %CD%
echo.

"%PYTHON_EXE%" -u -m src.train --stage pretrain --steps 5000 --batch-size 8 --seq-len 256 --grad-accum-steps 2 --loops 2 --lr 6e-4 --save-interval 500 --eval-interval 250 --log-file train.log %*

pause
