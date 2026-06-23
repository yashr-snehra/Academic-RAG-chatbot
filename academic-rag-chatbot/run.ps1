<#
.SYNOPSIS
  One-command setup + launch for the Academic RAG Chatbot.

  It will:
    1. Create the Python venv and install dependencies (first run only)
    2. Start Qdrant + Redis via Docker (starting Docker Desktop if needed)
    3. Verify Ollama is running and pull the required models if missing
    4. Start the API server and open the chat UI in your browser

.EXAMPLE
  ./run.ps1
  ./run.ps1 -Port 8001
#>
[CmdletBinding()]
param(
  [int]$Port = 8000,
  [string]$OllamaModel = "llama3.2:3b",
  [string]$EmbedModel  = "nomic-embed-text",
  [string]$IpexOllamaDir = "C:\ipex-ollama"   # IPEX-LLM Ollama portable (Intel Arc GPU)
)

# Native CLIs (docker/ollama) print progress to stderr; under -EA Stop that wraps as a
# terminating error and aborts the script, so we keep Continue and check $LASTEXITCODE.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

function Info($m) { Write-Host "[setup] $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "[warn]  $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "[error] $m" -ForegroundColor Red; exit 1 }

function Test-DockerUp {
  docker info *> $null
  return ($LASTEXITCODE -eq 0)
}

# ── 1. Python venv + dependencies ─────────────────────────────────────────────
if (-not (Test-Path ".venv")) {
  Info "Creating virtual environment..."
  $py = (Get-Command py -ErrorAction SilentlyContinue)
  if ($py) { py -3.13 -m venv .venv 2>$null; if (-not $?) { py -3 -m venv .venv } }
  else { python -m venv .venv }
  Info "Installing dependencies (one-time, ~2 min)..."
  .\.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
  .\.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt
  if ($LASTEXITCODE -ne 0) { Die "pip install failed - see output above." }
} else {
  Info "Virtual environment found."
}
$python = ".\.venv\Scripts\python.exe"

# ── 2. Docker (Qdrant + Redis) ────────────────────────────────────────────────
$dockerBin = "C:\Program Files\Docker\Docker\resources\bin"
if (Test-Path $dockerBin) { $env:Path += ";$dockerBin" }

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Warn "docker CLI not found. Install Docker Desktop, then re-run. Skipping Qdrant/Redis."
} else {
  if (-not (Test-DockerUp)) {
    Info "Starting Docker Desktop (this can take a minute)..."
    $dd = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dd) { Start-Process $dd } else { Warn "Docker Desktop.exe not found; start Docker manually." }
    Info "Waiting for the Docker daemon..."
    for ($i=0; $i -lt 60; $i++) { if (Test-DockerUp) { break }; Start-Sleep 3 }
    if (-not (Test-DockerUp)) { Die "Docker daemon did not start in time. Start Docker Desktop, then re-run." }
  }
  Info "Starting Qdrant + Redis..."
  docker compose -f docker/docker-compose.yml up -d 2>&1 | Out-Null
}

# ── 3. Ollama (Intel Arc via IPEX-LLM portable, else standard) ────────────────
$ollamaUrl = "http://localhost:11434"
function Test-OllamaUp { try { Invoke-RestMethod "$ollamaUrl/api/tags" -TimeoutSec 3 | Out-Null; return $true } catch { return $false } }

if (-not (Test-OllamaUp)) {
  $serveBat = Join-Path $IpexOllamaDir "ollama-serve.bat"
  if (Test-Path $serveBat) {
    Info "Starting IPEX-LLM Ollama on the Intel Arc GPU ($IpexOllamaDir)..."
    # Perf: Intel's recommended Level-Zero flag for the SYCL backend (~8% faster, measured
    # on Arc 140V). Flash attention / quantized KV are NOT supported on SYCL, so we don't set
    # them. This env var is inherited by ollama-serve.bat, which doesn't override it.
    $env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS = "1"
    # ollama-serve.bat sets the GPU env (OLLAMA_NUM_GPU=999, ZES_ENABLE_SYSMAN=1) then serves.
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$serveBat`"" -WorkingDirectory $IpexOllamaDir -WindowStyle Minimized
    # First Arc launch compiles SYCL kernels and can take a couple of minutes; later starts are fast.
    Info "Waiting for Ollama on $ollamaUrl (first GPU start can take ~2 min)..."
    for ($i=0; $i -lt 90; $i++) { if (Test-OllamaUp) { break }; Start-Sleep 2 }
  } else {
    Warn "IPEX-LLM portable not found at $IpexOllamaDir (no ollama-serve.bat)."
  }
}

if (-not (Test-OllamaUp)) {
  Warn "Ollama is not responding at $ollamaUrl."
  Warn "Start it manually: '$IpexOllamaDir\start-ollama.bat' (Intel Arc) or 'ollama serve' (standard)."
} else {
  $tags = (Invoke-RestMethod "$ollamaUrl/api/tags" -TimeoutSec 5).models.name
  $ollamaExe = Join-Path $IpexOllamaDir "ollama.exe"
  foreach ($m in @($OllamaModel, $EmbedModel)) {
    if ($tags -notcontains $m -and $tags -notcontains "$m`:latest") {
      if (Test-Path $ollamaExe) { Info "Pulling $m..."; & $ollamaExe pull $m }
      elseif (Get-Command ollama -ErrorAction SilentlyContinue) { Info "Pulling $m..."; ollama pull $m }
      else { Warn "Model '$m' missing; pull it manually." }
    }
  }
  Info "Ollama ready."
}

# ── 4. Start API + open browser ───────────────────────────────────────────────
# Free the port if something is already listening on it.
try { Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop |
        Select-Object -ExpandProperty OwningProcess -Unique |
        ForEach-Object { Stop-Process -Id $_ -Force } } catch {}

Info "Starting API server on http://localhost:$Port ..."
$server = Start-Process -FilePath $python `
  -ArgumentList "-m","uvicorn","app.main:app","--port","$Port" `
  -PassThru -NoNewWindow

for ($i=0; $i -lt 40; $i++) {
  try { if ((Invoke-RestMethod "http://localhost:$Port/api/v1/health" -TimeoutSec 2).status -eq "ok") { break } } catch {}
  Start-Sleep -Milliseconds 750
}

Info "Opening chat UI..."
Start-Process "http://localhost:$Port"
Write-Host ""
Write-Host "  Chat UI : http://localhost:$Port"      -ForegroundColor Green
Write-Host "  API docs: http://localhost:$Port/docs" -ForegroundColor Green
Write-Host "  Ingest PDFs:  $python scripts/ingest.py <folder-or-file>" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop the server." -ForegroundColor DarkGray
Write-Host ""
Wait-Process -Id $server.Id
