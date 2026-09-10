@echo off
title PEDiT Server
cd /d "%~dp0"

echo ================================================================
echo                    PEDiT - Web Server
echo ================================================================
echo.

:: 1. Activate virtual environment if present
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment: venv...
    call "venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment: .venv...
    call ".venv\Scripts\activate.bat"
)

:: 2. Check Python availability
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH!
    echo Please make sure Python 3.10+ is installed and added to PATH.
    echo.
    pause
    exit /b 1
)

:: 3. Launch server
echo [INFO] Starting server (checking dependencies and launching)...
echo [INFO] Web UI will be available at: http://localhost:8000
echo.

python server.py %*

if errorlevel 1 (
    echo.
    echo [ERROR] Server exited with an error.
    pause
)
