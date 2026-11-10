@echo off
setlocal
cd /d "%~dp0"
title LT-1 Inference Model Exporter

set "PYTHON_EXE=C:\Users\LTadmin\AppData\Local\Programs\Python\Python311\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo ======================================================
echo          LT-1 Accelerated Inference Exporter
echo ======================================================
echo Compacting Engram tables, fusing MoE 3D tensors,
echo and caching NCM codebook norms for zero-sync inference.
echo ======================================================
echo.

set "CKPT=%~1"
set "OUT=%~2"
set "TOK=%~3"

if "%CKPT%"=="" (
    if exist "checkpoints_1b\latest.pt" (
        set "CKPT=checkpoints_1b\latest.pt"
        set "OUT=checkpoints_1b\lt1_1b_inference.pt"
        set "TOK=data\tokenizer_cosmo.json"
    ) else (
        set "CKPT=checkpoints\latest.pt"
        set "OUT=checkpoints\lt1_inference.pt"
        set "TOK=data\tokenizer.json"
    )
)

echo Input Checkpoint: %CKPT%
echo Output Target:   %OUT%
echo Tokenizer:       %TOK%
echo.

"%PYTHON_EXE%" src\export.py --checkpoint "%CKPT%" --out "%OUT%" --tokenizer "%TOK%"

pause
