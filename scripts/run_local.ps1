param(
    [switch]$RunPipeline,
    [switch]$SkipInstall,
    [switch]$ForceRestart
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
$RunDir = Join-Path $Root ".local_run"
$PidFile = Join-Path $RunDir "pids.json"

Set-Location $Root
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

function Find-PythonLauncher {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return @($py.Source, "-3.10")
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @($python.Source, "")
    }

    throw "Python was not found. Install Python 3.10+ or make the py launcher available."
}

function Ensure-Venv {
    $venvPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        $launcher = Find-PythonLauncher
        if ($launcher[1]) {
            & $launcher[0] $launcher[1] -m venv .venv
        } else {
            & $launcher[0] -m venv .venv
        }
    }
    return $venvPython
}

function Test-Port {
    param([int]$Port)
    return Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
}

function Stop-Port {
    param([int]$Port)
    $listeners = Test-Port -Port $Port
    foreach ($listener in $listeners) {
        if ($listener.OwningProcess) {
            Stop-Process -Id $listener.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
}

function Start-ServiceProcess {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$Arguments,
        [int]$Port
    )

    if ($ForceRestart) {
        Stop-Port -Port $Port
        Start-Sleep -Seconds 1
    }

    $listener = Test-Port -Port $Port
    if ($listener) {
        Write-Host "$Name already listening on port $Port; leaving it running."
        return @{
            name = $Name
            port = $Port
            pid = $listener[0].OwningProcess
            reused = $true
        }
    }

    $process = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $Arguments `
        -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -PassThru

    Start-Sleep -Seconds 3
    return @{
        name = $Name
        port = $Port
        pid = $process.Id
        reused = $false
    }
}

function Exit-WithMessage {
    param([string]$Message)
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

$python = Ensure-Venv

if (-not $SkipInstall) {
    & $python -m pip install -r requirements.txt
}

$rawPath = Join-Path $Root "data\raw\DieCasting_Quality_Raw_Data_product1.csv"
$processedTrainPath = Join-Path $Root "data\processed\train.csv"
$modelPath = Join-Path $Root "artifacts\models\model.joblib"

if ($RunPipeline) {
    if (-not (Test-Path $rawPath)) {
        Exit-WithMessage "Raw data is missing: data\raw\DieCasting_Quality_Raw_Data_product1.csv. Download it from KAMP datasetSeq=55, place it there, then rerun scripts\run_local.ps1 -RunPipeline."
    }
    & $python -m src.data.prepare_data
    & $python -m src.models.train_binary
    & $python -m src.models.compare_baselines_xai
} elseif ((-not (Test-Path $modelPath)) -and (Test-Path $rawPath)) {
    & $python -m src.data.prepare_data
    & $python -m src.models.train_binary
} elseif (-not (Test-Path $modelPath)) {
    Exit-WithMessage "Model artifact is missing and raw data is not available. Add the raw CSV or run training first."
}

if (-not (Test-Path $processedTrainPath)) {
    Write-Host "Processed CSV files are not present. API/UI will use committed model artifacts; full retraining needs the KAMP raw CSV."
}

$services = @()
$services += Start-ServiceProcess `
    -Name "MLflow UI" `
    -FilePath $python `
    -Arguments @("-m", "mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db", "--host", "127.0.0.1", "--port", "5000") `
    -Port 5000

$services += Start-ServiceProcess `
    -Name "FastAPI" `
    -FilePath $python `
    -Arguments @("-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000") `
    -Port 8000

$streamlit = Join-Path $Root ".venv\Scripts\streamlit.exe"
$services += Start-ServiceProcess `
    -Name "Streamlit UI" `
    -FilePath $streamlit `
    -Arguments @("run", "src/ui/app.py", "--server.address", "127.0.0.1", "--server.port", "8501") `
    -Port 8501

$services | ConvertTo-Json -Depth 5 | Set-Content -Path $PidFile -Encoding UTF8

Write-Host ""
Write-Host "Local project is running."
Write-Host "MLflow UI:   http://127.0.0.1:5000"
Write-Host "FastAPI:     http://127.0.0.1:8000/docs"
Write-Host "Streamlit:   http://127.0.0.1:8501"
Write-Host ""
Write-Host "Stop services with: powershell -ExecutionPolicy Bypass -File scripts\stop_local.ps1"
