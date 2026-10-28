@echo off
setlocal
cd /d "%~dp0"

echo ===================================================
echo   Explainable Credit Risk API - Backend Launcher
echo ===================================================

if not exist "%~dp0Model\.venv\Scripts\python.exe" (
    echo [INFO] Virtual environment not found. Setting up Model\.venv ...
    python -m venv "%~dp0Model\.venv"
    if errorlevel 1 (
        echo [ERROR] Python is not installed or not on your PATH.
        echo Please install Python 3.10+ from https://www.python.org/
        pause
        exit /b 1
    )
    echo [INFO] Installing required Python packages from requirements.txt ...
    "%~dp0Model\.venv\Scripts\pip.exe" install -r "%~dp0requirements.txt"
)

echo [INFO] Starting FastAPI Backend on http://127.0.0.1:8000 ...
"%~dp0Model\.venv\Scripts\python.exe" -m uvicorn Backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
