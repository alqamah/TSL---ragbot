# PowerShell script to create virtual environment (if missing) and launch frontend
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$VenvDir = Join-Path $ScriptDir ".venv_frontend"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Industrial SOP RAG Assistant - Streamlit Frontend" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

if (-not (Test-Path $VenvDir)) {
    Write-Host "[1/2] Creating dedicated virtual environment at $VenvDir..." -ForegroundColor Yellow
    python -m venv $VenvDir
    Write-Host "[2/2] Installing frontend dependencies..." -ForegroundColor Yellow
    & "$VenvDir\Scripts\python.exe" -m pip install --upgrade pip
    & "$VenvDir\Scripts\pip.exe" install -r (Join-Path $ScriptDir "requirements.txt")
}

Write-Host "`n[START] Launching Streamlit on http://localhost:8501..." -ForegroundColor Green
$env:PYTHONPATH = $ProjectRoot
& "$VenvDir\Scripts\streamlit.exe" run (Join-Path $ScriptDir "app.py") --server.port 8501
