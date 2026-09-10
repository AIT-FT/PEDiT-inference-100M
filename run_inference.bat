@echo off
title PEDiT - Inference CLI
cd /d "%~dp0"

echo ================================================================
echo                  PEDiT - CLI Inference
echo ================================================================
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

set /p PROMPT_TEXT="Enter prompt (press Enter for default): "
if "%PROMPT_TEXT%"=="" set PROMPT_TEXT=A cinematic portrait of a cybernetic cat, neon lighting, 8k

echo.
echo [INFO] Generating image...
python inference.py --prompt "%PROMPT_TEXT%" --steps 8 --cfg 4.0 --output outputs/sample.png %*

if exist "outputs\sample.png" (
    echo [SUCCESS] Saved to outputs\sample.png
    start "" "outputs\sample.png"
)

pause
