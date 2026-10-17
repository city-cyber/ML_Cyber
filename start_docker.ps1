param(
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [switch]$SkipBuild,
    [switch]$ExportModel
)

$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot

function Invoke-Docker {
    param([string[]]$DockerArguments)
    & docker @DockerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Docker failed (exit $LASTEXITCODE): docker $($DockerArguments -join ' ')"
    }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker is not installed. Install and start Docker Desktop with Linux containers.'
}

Push-Location -LiteralPath $projectDirectory
$previousPort = $env:ANTISPAM_PORT
try {
    & docker info --format '{{.OSType}}' 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw 'Docker Engine is unavailable. Start Docker Desktop and retry.'
    }
    $modelFile = Join-Path $projectDirectory 'microservice/app/model/model.joblib'
    $metadataFile = Join-Path $projectDirectory 'microservice/app/model/metadata.json'
    if ($ExportModel -or -not (Test-Path -LiteralPath $modelFile) -or -not (Test-Path -LiteralPath $metadataFile)) {
        $projectPython = Join-Path $projectDirectory '.venv/Scripts/python.exe'
        if (-not (Test-Path -LiteralPath $projectPython)) {
            $projectPython = 'python'
        }
        & $projectPython (Join-Path $projectDirectory 'export_model.py')
        if ($LASTEXITCODE -ne 0) {
            throw 'Model export failed. See README and install the project requirements.'
        }
        if ($SkipBuild) {
            throw 'Model was exported. Run again without -SkipBuild to include it in the image.'
        }
    }
    $env:ANTISPAM_PORT = [string]$Port
    if (-not $SkipBuild) {
        Invoke-Docker -DockerArguments @('compose', '-f', 'compose.yaml', 'build', 'antispam')
    }
    Invoke-Docker -DockerArguments @('compose', '-f', 'compose.yaml', 'up', '-d', '--no-build', '--wait', '--wait-timeout', '120', 'antispam')
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 10
    if ($health.status -ne 'ok') {
        throw 'The service did not pass its health check.'
    }
    $localMetadata = Get-Content -LiteralPath $metadataFile -Raw | ConvertFrom-Json
    $runningModel = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/model" -TimeoutSec 10
    if ($runningModel.model_sha256 -ne $localMetadata.model_sha256) {
        throw 'The container has an older model. Run this script without -SkipBuild.'
    }
    Write-Host "API is ready: http://127.0.0.1:$Port/docs"
    Write-Host "Health check: http://127.0.0.1:$Port/health"
    Write-Host 'Stop: docker compose -f compose.yaml down'
}
finally {
    $env:ANTISPAM_PORT = $previousPort
    Pop-Location
}
