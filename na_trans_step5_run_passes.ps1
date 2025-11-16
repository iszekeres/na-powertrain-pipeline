Set-StrictMode -Version Latest

$NEWLOGS   = "newlogs"
$OUT_ROOT  = Join-Path $NEWLOGS "output"
$PASS_ROOT = Join-Path $OUT_ROOT "02_passes"

$CLEANED = Join-Path $NEWLOGS "cleaned"
$logsGlob = Join-Path $CLEANED "__trans_focus__clean_FULL__*.csv"

Write-Host "`n[STEP 5] Running passes (STOPGO, KICKDOWN, CORNER, LAT, INTENT)..." -ForegroundColor Cyan

New-Item -ItemType Directory -Path $PASS_ROOT -Force | Out-Null

$STOPGO_DIR = Join-Path $PASS_ROOT "STOPGO"
$KICK_DIR   = Join-Path $PASS_ROOT "KICKDOWN"
$CORNER_DIR = Join-Path $PASS_ROOT "CORNER"
$LAT_DIR    = Join-Path $PASS_ROOT "LAT"
$INTENT_DIR = Join-Path $PASS_ROOT "INTENT"
$CONSIST_DIR = Join-Path $PASS_ROOT "CONSIST"

New-Item -ItemType Directory -Path $STOPGO_DIR,$KICK_DIR,$CORNER_DIR,$LAT_DIR,$INTENT_DIR,$CONSIST_DIR -Force | Out-Null

# STOPGO
if (Test-Path ".\stopgo_pass_weighted.py") {
    Write-Host "  [RUN] STOPGO (weighted)" -ForegroundColor Yellow
    python ".\stopgo_pass_weighted.py" --logs-glob $logsGlob --out (Join-Path $STOPGO_DIR "STOPGO__SHIFT_DOWN__DELTA.tsv")
} elseif (Test-Path ".\stopgo_pass.py") {
    Write-Host "  [RUN] STOPGO (legacy)" -ForegroundColor Yellow
    python ".\stopgo_pass.py" --logs-glob $logsGlob --out (Join-Path $STOPGO_DIR "STOPGO__SHIFT_DOWN__DELTA.tsv")
} else {
    Write-Host "  [WARN] STOPGO script not found; skipping STOPGO pass." -ForegroundColor DarkYellow
}

# KICKDOWN
if (Test-Path ".\kickdown_pass_weighted.py") {
    Write-Host "  [RUN] KICKDOWN" -ForegroundColor Yellow
    python ".\kickdown_pass_weighted.py" --logs-glob $logsGlob --out (Join-Path $KICK_DIR "KICKDOWN__SHIFT_DOWN__DELTA.tsv")
} else {
    Write-Host "  [WARN] kickdown_pass_weighted.py not found; skipping KICKDOWN pass." -ForegroundColor DarkYellow
}

# CORNER core + chassis + combine
if (Test-Path ".\corner_exit_pass_weighted__shim.py") {
    Write-Host "  [RUN] CORNER CORE" -ForegroundColor Yellow
    python ".\corner_exit_pass_weighted__shim.py" --logs-glob $logsGlob --out (Join-Path $CORNER_DIR "CORNER_CORE__SHIFT_DOWN__DELTA.tsv")
} else {
    Write-Host "  [WARN] corner_exit_pass_weighted__shim.py not found; skipping CORNER CORE." -ForegroundColor DarkYellow
}

if (Test-Path ".\corner_exit_pass_weighted__chassis_shim.py") {
    Write-Host "  [RUN] CORNER CHASSIS" -ForegroundColor Yellow
    python ".\corner_exit_pass_weighted__chassis_shim.py" --logs-glob $logsGlob --out (Join-Path $CORNER_DIR "CORNER_CHASSIS__SHIFT_DOWN__DELTA.tsv")
} else {
    Write-Host "  [WARN] corner_exit_pass_weighted__chassis_shim.py not found; skipping CORNER CHASSIS." -ForegroundColor DarkYellow
}

if (Test-Path ".\corner_combine_core_chassis.py") {
    Write-Host "  [RUN] CORNER COMBINE (CORE+CHASSIS)" -ForegroundColor Yellow
    $CORE_T = Join-Path $CORNER_DIR "CORNER_CORE__SHIFT_DOWN__DELTA.tsv"
    $CH_T   = Join-Path $CORNER_DIR "CORNER_CHASSIS__SHIFT_DOWN__DELTA.tsv"
    $OUT_T  = Join-Path $CORNER_DIR "CORNER__SHIFT_DOWN__DELTA__COMBINED.tsv"
    python ".\corner_combine_core_chassis.py" $CORE_T $CH_T $OUT_T
} else {
    Write-Host "  [WARN] corner_combine_core_chassis.py not found; skipping CORNER COMBINE." -ForegroundColor DarkYellow
}

# LAT (shift latency)
if (Test-Path ".\shift_latency_pass_weighted.py") {
    Write-Host "  [RUN] LAT (shift latency)" -ForegroundColor Yellow
    python ".\shift_latency_pass_weighted.py" --logs-glob $logsGlob --out-prefix (Join-Path $LAT_DIR "LAT")
} else {
    Write-Host "  [WARN] shift_latency_pass_weighted.py not found; skipping LAT pass." -ForegroundColor DarkYellow
}

# INTENT (SHIFT_UP + TCC_RELEASE)
if (Test-Path ".\driver_intent_pass_weighted__PLUS_v3.py") {
    Write-Host "  [RUN] INTENT (PLUS_v3)" -ForegroundColor Yellow
    python ".\driver_intent_pass_weighted__PLUS_v3.py" --logs-glob $logsGlob --out-dir $INTENT_DIR
} elseif (Test-Path ".\driver_intent_pass_weighted__PLUS_v3__CORE.py") {
    Write-Host "  [RUN] INTENT (PLUS_v3 CORE wrapper)" -ForegroundColor Yellow
    python ".\driver_intent_pass_weighted__PLUS_v3__CORE.py" --logs-glob $logsGlob --out-dir $INTENT_DIR
} else {
    Write-Host "  [WARN] INTENT script not found; skipping INTENT pass." -ForegroundColor DarkYellow
}

# CONSIST (consistency pass, UP/DOWN deltas)
if (Test-Path ".\consist_pass_weighted.py") {
    Write-Host "  [RUN] CONSIST (weighted)" -ForegroundColor Yellow
    # Build a clean_list.txt from logsGlob for CONSIST
    $cleanList = Join-Path $CLEANED "clean_list_for_CONSIST.txt"
    Get-ChildItem $CLEANED -Filter "__trans_focus__clean_FULL__*.csv" | ForEach-Object { $_.FullName } | Set-Content $cleanList
    python ".\consist_pass_weighted.py" --clean-list $cleanList --out-dir $CONSIST_DIR
} else {
    Write-Host "  [WARN] consist_pass_weighted.py not found; skipping CONSIST pass." -ForegroundColor DarkYellow
}

Write-Host "`n[STEP 5 RESULTS] Pass deltas present:" -ForegroundColor Yellow
Get-ChildItem $PASS_ROOT -Recurse -File | Select-Object Directory,Name,Length
