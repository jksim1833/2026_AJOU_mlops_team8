$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir

$command = Get-Command pycharm64 -ErrorAction SilentlyContinue
if ($command) {
    Start-Process -FilePath $command.Source -ArgumentList @($Root)
    exit 0
}

$candidates = @(
    "C:\Program Files\JetBrains\PyCharm 2025.3.2\bin\pycharm64.exe",
    "C:\Program Files\JetBrains\PyCharm Community Edition 2025.3.2\bin\pycharm64.exe"
)

$found = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $found) {
    $found = Get-ChildItem `
        -Path "C:\Program Files\JetBrains", "$env:LOCALAPPDATA\Programs" `
        -Recurse `
        -Filter pycharm64.exe `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}

if (-not $found) {
    Write-Host "ERROR: PyCharm executable was not found." -ForegroundColor Red
    exit 1
}

Start-Process -FilePath $found -ArgumentList @($Root)

