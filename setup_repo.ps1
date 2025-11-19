Param(
    [string]$RepoRoot = "."
)

Write-Host ">>> NA Powertrain Pipeline repo setup starting..." -ForegroundColor Cyan

# Move to the repo root
Set-Location $RepoRoot

# 1) Create standard directories
$dirs = @("src", "tools", "docs", "examples")
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        Write-Host "Creating directory: $d"
        New-Item -ItemType Directory -Path $d | Out-Null
    } else {
        Write-Host "Directory already exists: $d"
    }
}

# 2) Create README.md if missing
$readmePath = Join-Path (Get-Location) "README.md"
if (-not (Test-Path $readmePath)) {
    Write-Host "Creating README.md"
@"
# NA Powertrain Pipeline

This repo contains scripts and docs for analyzing transmission logs and building shift/TCC tables for the Tahoe L83/6L80 project.

## Project assumptions

- Vehicle: Tahoe L83 with 6L80 (T43)
- Final drive: **3.08**
- Tire diameter: **32.5"**
- Torque converter: stock **EC³ single-disc**
- Units: **mph**
- Throttle axis: **17-point TPS**  
  \`0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100\`

## Directory layout

- \`src/\` – main Python scripts:
  - \`trans_clean_analyze__SAFE_REBUILD.py\` – log cleaner (CLEAN_FULL)
  - \`clean_full_only_wrapper__PB_MP__QUIET.py\` – wrapper to run cleaner in FULL-only mode
  - \`shift_table_builder_Throttle17.py\` – builds SHIFT UP/DOWN tables
  - \`tcc_table_builder_Throttle17__FIX.py\` – builds TCC APPLY/RELEASE tables
  - \`overlay_polish_v3.py\` – neutral overlay polish (SHIFT only)
  - \`table_audit_and_fix.py\` – audits SHIFT/TCC tables and enforces policies
  - \`force_1dp__STRICT_SHIFT.py\`, \`force_1dp__STRICT_TCC.py\` – enforce 0.1 mph & 317/318 rules
  - \`tsv_escape_guard.py\` – TSV cleanup and header normalization

- \`tools/\` – one-paste helper scripts (PowerShell, bash) to run standard flows.

- \`docs/\` – pipeline docs and notes:
  - \`README__Neutral_First_Pipeline.md\`
  - \`RUNBOOK__Neutral_Build_and_Overlay.md\`
  - \`INTENT_Pass__Locked__2025-11-08.md\`
  - any other notes about passes (CONSIST, CORNER, LAT, STOPGO, KICKDOWN, INTENT, TCC_EDGE, etc.)

- \`examples/\` – small example tables and log snippets; **no large logs**.

## High-level workflow

1. Put raw logs in \`./newlogs/\`.
2. Run the CLEAN_FULL-only pipeline to create \`__trans_focus__clean_FULL__*.csv\` in \`./newlogs/cleaned/\`.
3. Run table builders to produce SHIFT and TCC tables in \`./newlogs/output/01_tables/{shift,tcc}/\`.
4. Run neutral overlay + audit tools to produce neutral candidate packs.
5. (Optional) Run style overlays and Mode B blending.

See files under \`docs/\` for detailed runbooks when they are added.
"@ | Set-Content -Path $readmePath -Encoding UTF8
} else {
    Write-Host "README.md already exists, leaving it unchanged."
}

# 3) Create stub docs if missing
$docsToCreate = @{
    "docs\README__Neutral_First_Pipeline.md" = @"
# Neutral-First Pipeline (Stub)

This file should describe the neutral-first, strict/no-fallback pipeline for the Tahoe L83/6L80 project.

TODO:
- Copy in or summarize the canonical neutral-first pipeline doc.
- Document CLEAN_FULL-only policy.
- Document SHIFT/TCC table builder usage and outputs.
"@;

    "docs\RUNBOOK__Neutral_Build_and_Overlay.md" = @"
# Neutral Build and Overlay Runbook (Stub)

This file should contain step-by-step instructions for building neutral SHIFT/TCC tables and applying overlays.

TODO:
- Add step-by-step commands for:
  - Running the CLEAN_FULL-only cleaner.
  - Building SHIFT and TCC tables.
  - Running overlay_polish_v3.py (SHIFT only).
  - Running table_audit_and_fix.py.
  - Enforcing 17-pt TPS axis and 0.1 mph formatting with 317/318 preserved.
"@;

    "docs\INTENT_Pass__Locked__2025-11-08.md" = @"
# INTENT Pass – Locked Configuration (Stub)

This stub should eventually describe the INTENT pass configuration as of 2025-11-08.

TODO:
- Add thresholds for warmed filters (ECT/TFT ~100°F).
- Describe thr-rate-pedal, thr-rate-throttle, brake-release windows.
- Describe how INTENT affects SHIFT UP and TCC RELEASE tables.
"@
}

foreach ($kvp in $docsToCreate.GetEnumerator()) {
    $path = $kvp.Key
    $content = $kvp.Value
    if (-not (Test-Path $path)) {
        Write-Host "Creating stub doc: $path"
        $dir = Split-Path $path -Parent
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir | Out-Null
        }
        $content | Set-Content -Path $path -Encoding UTF8
    } else {
        Write-Host "Doc already exists, leaving it unchanged: $path"
    }
}

Write-Host ""
Write-Host ">>> Repo layout setup complete." -ForegroundColor Green
Write-Host "Next steps:"
Write-Host "  1) Move your Python scripts into .\src\"
Write-Host "  2) Move your docs/markdown into .\docs\ (replacing stub content)"
Write-Host "  3) Optionally add helper .ps1 scripts into .\tools\"
Write-Host "  4) Commit and push:"
Write-Host "       git status"
Write-Host "       git add ."
Write-Host "       git commit -m 'Set up repo layout for NA powertrain pipeline'"
Write-Host "       git push"

