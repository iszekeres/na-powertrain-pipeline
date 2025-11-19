Param()

$ErrorActionPreference = 'Stop'

Write-Host "=== RUN CLEAN_FULL ON test1.csv AND test2.csv ===" -ForegroundColor Cyan

# Ensure we are running from the repo root (where this script lives)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$NEWLOGS  = 'newlogs'
$CLEANED  = Join-Path $NEWLOGS 'cleaned'
$OUT_ROOT = Join-Path $NEWLOGS 'output'
$STAGING  = Join-Path $NEWLOGS '_staging_Tests'

# Ensure required directories exist
foreach ($dir in @($NEWLOGS, $CLEANED, $OUT_ROOT, $STAGING)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
}

$test1 = Join-Path $NEWLOGS 'test1.csv'
$test2 = Join-Path $NEWLOGS 'test2.csv'

if (-not (Test-Path $test1)) {
    Write-Host "[ERROR] Missing raw log: $test1" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $test2)) {
    Write-Host "[ERROR] Missing raw log: $test2" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Cleaning test1.csv..." -ForegroundColor Yellow
python ".\clean_full_only_wrapper__PB_MP__QUIET.py" `
    --raw-glob    $test1 `
    --cleaner     ".\trans_clean_analyze__SAFE_REBUILD.py" `
    --staging     $STAGING `
    --cleaned-dir $CLEANED `
    --out-root    $OUT_ROOT `
    --workers     1
Write-Host "[OK] Finished test1.csv" -ForegroundColor Green

Write-Host "[INFO] Cleaning test2.csv..." -ForegroundColor Yellow
python ".\clean_full_only_wrapper__PB_MP__QUIET.py" `
    --raw-glob    $test2 `
    --cleaner     ".\trans_clean_analyze__SAFE_REBUILD.py" `
    --staging     $STAGING `
    --cleaned-dir $CLEANED `
    --out-root    $OUT_ROOT `
    --workers     1
Write-Host "[OK] Finished test2.csv" -ForegroundColor Green

Write-Host "`nCLEAN_FULL files should now be in $CLEANED as __trans_focus__clean_FULL__*.csv" -ForegroundColor Cyan

