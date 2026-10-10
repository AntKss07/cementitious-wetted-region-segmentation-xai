param(
    [int]$Port = 8765,
    [string]$Python = $env:RESEARCH_PYTHON
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$candidates = @()
if ($Python) { $candidates += $Python }
$candidates += Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCommand -and $pythonCommand.Source -notlike '*WindowsApps*') {
    $candidates += $pythonCommand.Source
}
# Codex desktop supplies a Python runtime with NumPy and Pillow on this host.
# This fallback is optional; a normal project virtual environment takes priority.
$candidates += Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        & $candidate -c 'import numpy, PIL' 2>$null
        if ($LASTEXITCODE -eq 0) {
            & $candidate -m research_app.server --port $Port
            exit $LASTEXITCODE
        }
    }
}
throw 'Python with NumPy and Pillow was not found. Follow FRONTEND.md to create .venv and install requirements-frontend.txt, or pass -Python <python.exe>.'
