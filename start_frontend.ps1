Set-Location -Path "$PSScriptRoot\Frontend"

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  Credit Risk Web App - Frontend Launcher" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

if (-not (Test-Path "$PSScriptRoot\Frontend\node_modules")) {
    Write-Host "[INFO] node_modules not found. Installing npm dependencies ..." -ForegroundColor Yellow
    npm install
}

Write-Host "[INFO] Starting Vite Frontend on http://localhost:5173 ..." -ForegroundColor Green
npm run dev
