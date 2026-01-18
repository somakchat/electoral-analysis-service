# Political Strategy Maker - Streamlit Frontend
# Run this script from the frontend directory

$ErrorActionPreference = "Stop"

Write-Host "Political Strategy Maker - Starting Streamlit Frontend" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

# Check if virtual environment exists
if (Test-Path "venv") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    .\venv\Scripts\Activate.ps1
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

Write-Host "`nStarting Streamlit on http://localhost:8501" -ForegroundColor Green
Write-Host "Make sure the backend is running on http://localhost:8000" -ForegroundColor Yellow
Write-Host "`nPress Ctrl+C to stop" -ForegroundColor Yellow

# Run Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

