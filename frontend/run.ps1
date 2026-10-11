# Startup script for Cementitious Sorptivity Segmentation & XAI Workbench
param (
    [int]$Port = 8000,
    [string]$HostIP = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

$workspaceRoot = Split-Path -Parent $PSScriptRoot
Set-Location $workspaceRoot

$pythonExe = Join-Path $workspaceRoot ".venv-cuda\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = Join-Path $workspaceRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Starting Cementitious Sorptivity XAI Workbench" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Python: $pythonExe" -ForegroundColor Yellow
Write-Host "URL:    http://${HostIP}:${Port}" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

& $pythonExe -m frontend.app.server --host $HostIP --port $Port
