$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
$PidFile = Join-Path $Root ".local_run\pids.json"

if (-not (Test-Path $PidFile)) {
    Write-Host "No .local_run\pids.json file found. Nothing to stop."
    exit 0
}

$services = Get-Content $PidFile -Encoding UTF8 | ConvertFrom-Json
foreach ($service in $services) {
    if ($service.pid -and -not $service.reused) {
        Stop-Process -Id $service.pid -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped $($service.name) pid=$($service.pid)"
    } elseif ($service.reused) {
        Write-Host "Skipped reused service $($service.name) pid=$($service.pid)"
    }
}

Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
