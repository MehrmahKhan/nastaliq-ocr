# One shot setup for Windows PowerShell.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#
# Creates the virtual environment, installs the lean dependency set, puts
# the two files that could not be written remotely into place, and runs the
# test suite. Safe to run more than once.

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
Write-Host "project root: $projectRoot" -ForegroundColor Cyan

# --- python check ---------------------------------------------------------
$versionText = (python --version 2>&1)
Write-Host "found $versionText"

$versionMatch = [regex]::Match($versionText, '(\d+)\.(\d+)')
if ($versionMatch.Success) {
    $major = [int]$versionMatch.Groups[1].Value
    $minor = [int]$versionMatch.Groups[2].Value
    if ($major -ne 3 -or $minor -lt 10) {
        Write-Host "this project wants python 3.10 or newer" -ForegroundColor Red
        exit 1
    }
    if ($minor -ge 13) {
        Write-Host "note: python 3.13 works but 3.11 or 3.12 is the smoother path" -ForegroundColor Yellow
    }
}

# --- the two files that remote tools are not allowed to write -------------
$makefileTemplate = Join-Path $projectRoot "scripts\_makefile_template.txt"
if (Test-Path $makefileTemplate) {
    Copy-Item $makefileTemplate (Join-Path $projectRoot "Makefile") -Force
    Write-Host "placed Makefile"
}

$ciTemplate = Join-Path $projectRoot "scripts\_ci_template.yml"
if (Test-Path $ciTemplate) {
    $workflowDir = Join-Path $projectRoot ".github\workflows"
    New-Item -ItemType Directory -Force -Path $workflowDir | Out-Null
    Copy-Item $ciTemplate (Join-Path $workflowDir "ci.yml") -Force
    Write-Host "placed .github\workflows\ci.yml"
}

# --- virtual environment --------------------------------------------------
if (-not (Test-Path (Join-Path $projectRoot ".venv"))) {
    Write-Host "creating the virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
}

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

Write-Host "upgrading pip..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip setuptools wheel --quiet

Write-Host "installing the lean dependency set, this takes a few minutes..." -ForegroundColor Cyan
& $venvPython -m pip install -r requirements-local.txt

# --- env file -------------------------------------------------------------
if (-not (Test-Path (Join-Path $projectRoot ".env"))) {
    Copy-Item (Join-Path $projectRoot ".env.example") (Join-Path $projectRoot ".env")
    Write-Host "created .env from the example, fill in your keys" -ForegroundColor Yellow
}

# --- tests ----------------------------------------------------------------
Write-Host "running the test suite..." -ForegroundColor Cyan
$env:PYTHONPATH = "src"
& $venvPython -m pytest tests -q

Write-Host ""
Write-Host "setup finished." -ForegroundColor Green
Write-Host "next, in this terminal:" -ForegroundColor Green
Write-Host "  .venv\Scripts\Activate.ps1"
Write-Host "  `$env:PYTHONPATH='src'"
Write-Host "  dvc repro split"
