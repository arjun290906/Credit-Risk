Set-Location -Path $PSScriptRoot

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  Explainable Credit Risk API - Backend Launcher" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

$venvPython = "$PSScriptRoot\Model\.venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[INFO] Virtual environment not found. Setting up Model\.venv ..." -ForegroundColor Yellow
    python -m venv "$PSScriptRoot\Model\.venv"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Python 3.10+ must be installed and on PATH." -ForegroundColor Red
        exit 1
    }
    Write-Host "[INFO] Installing required Python packages from requirements.txt ..." -ForegroundColor Yellow
    & "$PSScriptRoot\Model\.venv\Scripts\pip.exe" install -r "$PSScriptRoot\requirements.txt"
}

Write-Host "[INFO] Starting FastAPI Backend on http://127.0.0.1:8000 ..." -ForegroundColor Green
& "$PSScriptRoot\Model\.venv\Scripts\python.exe" -m uvicorn Backend.main:app --host 127.0.0.1 --port 8000 --reload
