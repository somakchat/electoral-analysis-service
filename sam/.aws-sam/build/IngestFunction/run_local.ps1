# Political Strategy Maker - Local Development Server
# Run this script from the backend directory

param(
    [switch]$IngestData,
    [switch]$SkipIngest
)

$ErrorActionPreference = "Stop"

Write-Host "Political Strategy Maker - Starting Local Server" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

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

# Check for .env file
if (-not (Test-Path ".env")) {
    if (Test-Path "env.template") {
        Write-Host "Creating .env from template..." -ForegroundColor Yellow
        Copy-Item "env.template" ".env"
        Write-Host "IMPORTANT: Edit .env with your OpenAI API key!" -ForegroundColor Red
    } else {
        Write-Host "WARNING: .env file not found!" -ForegroundColor Red
    }
}

# Create data directories
New-Item -ItemType Directory -Force -Path ".\data" | Out-Null
New-Item -ItemType Directory -Force -Path ".\index" | Out-Null
New-Item -ItemType Directory -Force -Path ".\data\uploads" | Out-Null
New-Item -ItemType Directory -Force -Path ".\data\memory" | Out-Null

# Check if index is empty and offer to ingest data
$indexFiles = Get-ChildItem -Path ".\index" -File -ErrorAction SilentlyContinue
$politicalDataPath = "..\political-data"

if ($IngestData -or ((-not $SkipIngest) -and ($indexFiles.Count -eq 0) -and (Test-Path $politicalDataPath))) {
    Write-Host "`nPolitical data folder found. Ingesting data..." -ForegroundColor Green
    Write-Host "This may take a few minutes on first run." -ForegroundColor Yellow
    
    python ..\scripts\ingest_political_data.py
    
    Write-Host "`nData ingestion complete!" -ForegroundColor Green
}

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "Starting FastAPI server on http://localhost:8000" -ForegroundColor Green
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "WebSocket: ws://localhost:8000/ws/chat" -ForegroundColor Green
Write-Host "`nTo ingest data manually: python ..\scripts\ingest_political_data.py" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan

# Run the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
