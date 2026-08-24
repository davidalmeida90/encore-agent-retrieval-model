<#
.SYNOPSIS
  Start Encore locally: backend, frontend, browser. Ctrl+C stops everything.

.DESCRIPTION
  Both servers run as child processes of this window. The finally block kills
  them on Ctrl+C or when the window closes, so nothing is left holding port 8000
  the next time you start (which is exactly what happened during development:
  a stale uvicorn kept the port and the new one exited silently, leaving the old
  code serving requests).

.EXAMPLE
  .\start.ps1
  .\start.ps1 -NoBrowser
  .\start.ps1 -BackendPort 8001 -FrontendPort 5174
#>
param(
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5173,
  [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = Join-Path $root 'backend\.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
  Write-Error "No virtualenv at $python. Create it before running this."
}

function Stop-Port([int]$Port) {
  $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  foreach ($c in $conns) {
    Write-Host "  freeing port $Port (PID $($c.OwningProcess))" -ForegroundColor DarkYellow
    Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
  }
}

$procs = @()

try {
  Write-Host "Encore" -ForegroundColor Cyan
  Stop-Port $BackendPort
  Stop-Port $FrontendPort

  Write-Host "  backend  -> http://localhost:$BackendPort"
  $procs += Start-Process -FilePath $python `
    -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--port', "$BackendPort") `
    -WorkingDirectory (Join-Path $root 'backend') -PassThru -NoNewWindow

  Write-Host "  frontend -> http://localhost:$FrontendPort"
  $procs += Start-Process -FilePath 'cmd.exe' `
    -ArgumentList @('/c', 'npm', 'run', 'dev', '--', '--port', "$FrontendPort") `
    -WorkingDirectory (Join-Path $root 'frontend') -PassThru -NoNewWindow

  # The backend warms the reranker on startup (a few seconds of torch loading),
  # so wait for /health rather than guessing with a fixed sleep.
  Write-Host "  waiting for backend..." -NoNewline
  $ready = $false
  foreach ($i in 1..60) {
    Start-Sleep -Milliseconds 500
    try {
      $r = Invoke-WebRequest "http://localhost:$BackendPort/health" -UseBasicParsing -TimeoutSec 2
      if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
  }
  if ($ready) { Write-Host " ready" -ForegroundColor Green }
  else { Write-Host " no /health yet, continuing anyway" -ForegroundColor Yellow }

  if (-not $NoBrowser) { Start-Process "http://localhost:$FrontendPort" }

  Write-Host ""
  Write-Host "Running. Press Ctrl+C to stop both servers." -ForegroundColor Cyan
  while ($true) { Start-Sleep -Seconds 1 }
}
finally {
  Write-Host ""
  Write-Host "Shutting down..." -ForegroundColor Cyan
  foreach ($p in $procs) {
    if ($p -and -not $p.HasExited) {
      Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
  }
  # npm spawns vite as a grandchild, so killing cmd.exe alone can orphan it.
  Stop-Port $BackendPort
  Stop-Port $FrontendPort
  Write-Host "Stopped." -ForegroundColor Green
}
