$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$pythonExe = Join-Path (Split-Path -Parent $projectRoot) "env\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found at $pythonExe"
}

$backendUri = "sqlite:///C:/Users/Rovin/mlfow/timeseries/mlflow.db"
$artifactRoot = "file:///C:/Users/Rovin/mlfow/timeseries/mlruns"

Write-Host "Starting MLflow server on http://127.0.0.1:5001" -ForegroundColor Cyan
Write-Host "Backend: $backendUri"
Write-Host "Artifacts: $artifactRoot"

& $pythonExe -m mlflow server `
    --host 127.0.0.1 `
    --port 5001 `
    --backend-store-uri $backendUri `
    --default-artifact-root $artifactRoot
