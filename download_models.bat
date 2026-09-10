@echo off
title PEDiT - Model Downloader
cd /d "%~dp0"

echo ================================================================
echo           PEDiT-100M - Hugging Face Model Downloader
echo ================================================================
echo Repository: https://huggingface.co/AIT-FT/PEDiT-100M
echo.

if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH!
    pause
    exit /b 1
)

echo Select which models to download from Hugging Face:
echo   1. FP16 (Recommended, 206 MB)
echo   2. FP8  (Fast ^& Light, 103 MB)
echo   3. INT8 (Maximum Speed, 103 MB)
echo   4. ALL  (FP16 + FP8 + INT8 + VAE + Text Encoder)
echo   5. Check local checkpoints status
echo.

set /p CHOICE="Enter choice (1-5) [default: 1]: "
if "%CHOICE%"=="" set CHOICE=1

if "%CHOICE%"=="1" (
    python download_utils.py --model fp16
) else if "%CHOICE%"=="2" (
    python download_utils.py --model fp8
) else if "%CHOICE%"=="3" (
    python download_utils.py --model int8
) else if "%CHOICE%"=="4" (
    python download_utils.py --all
) else if "%CHOICE%"=="5" (
    python download_utils.py --check
) else (
    echo [INFO] Invalid choice, downloading default FP16...
    python download_utils.py --model fp16
)

echo.
pause
