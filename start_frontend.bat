@echo off
setlocal
cd /d "%~dp0Frontend"

echo ===================================================
echo   Credit Risk Web App - Frontend Launcher
echo ===================================================

if not exist "%~dp0Frontend\node_modules" (
    echo [INFO] node_modules not found. Installing npm dependencies ...
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed. Please ensure Node.js is installed.
        pause
        exit /b 1
    )
)

echo [INFO] Starting Vite Frontend on http://localhost:5173 ...
call npm run dev
pause
