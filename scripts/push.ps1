# Commit and push both the code and the data, in the right order.
#
#   powershell -ExecutionPolicy Bypass -File scripts\push.ps1 "what changed"
#
# DVC goes first on purpose. If the code lands on GitHub before the data is
# on the remote, anyone cloning the repo gets .dvc pointer files referring
# to blobs that are not there yet.

param(
    [Parameter(Position = 0)]
    [string]$Message = "update"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path (Join-Path $projectRoot ".git"))) {
    Write-Host "this folder is not a git repository yet, run the git init steps first" -ForegroundColor Red
    exit 1
}

# --- data first -----------------------------------------------------------
if (Test-Path (Join-Path $projectRoot ".dvc")) {
    Write-Host "pushing data to the dvc remote..." -ForegroundColor Cyan
    dvc push
    if ($LASTEXITCODE -ne 0) {
        Write-Host "dvc push failed, stopping before the git push" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "no .dvc folder, skipping the data push" -ForegroundColor Yellow
}

# --- then the code --------------------------------------------------------
Write-Host "staging changes..." -ForegroundColor Cyan
git add -A

$pending = git status --porcelain
if ([string]::IsNullOrWhiteSpace($pending)) {
    Write-Host "nothing to commit" -ForegroundColor Yellow
} else {
    git commit -m $Message
}

$branch = git rev-parse --abbrev-ref HEAD
Write-Host "pushing $branch to origin..." -ForegroundColor Cyan
git push -u origin $branch

Write-Host ""
Write-Host "done. https://github.com/MehrmahKhan/nastaliq-ocr" -ForegroundColor Green
