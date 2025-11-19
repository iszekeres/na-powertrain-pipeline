param(
    [string]$ExportZip = "NA_Trans_Project_Export__2025-11-19.zip"
)

Write-Host ">>> NA Trans project export import starting..." -ForegroundColor Cyan

# Repo root = current directory
$repoRoot = Get-Location
$zipPath  = Join-Path $repoRoot $ExportZip

if (-not (Test-Path $zipPath)) {
    Write-Host "ERROR: Could not find export zip at: $zipPath" -ForegroundColor Red
    Write-Host "Put the project export zip in the repo root or pass -ExportZip <path>."
    exit 1
}

# Temp directory for extraction
$tempDir = Join-Path $repoRoot "_import_na_trans_temp"
if (Test-Path $tempDir) {
    Write-Host "Removing existing temp dir: $tempDir"
    Remove-Item $tempDir -Recurse -Force
}
New-Item -ItemType Directory -Path $tempDir | Out-Null

Write-Host "Extracting $ExportZip to $tempDir ..."
Expand-Archive -Path $zipPath -DestinationPath $tempDir -Force

# Ensure target folders exist
$dirsToEnsure = @(
    "docs",
    "docs\reference",
    "examples",
    "bundles",
    "channel_maps"
)
foreach ($d in $dirsToEnsure) {
    $full = Join-Path $repoRoot $d
    if (-not (Test-Path $full)) {
        New-Item -ItemType Directory -Path $full | Out-Null
        Write-Host "Created directory: $d"
    }
}

# Map from export name -> repo path
# NOTE: we intentionally skip 2015_tahoe_t43_trans_tables.zip to avoid duplicating the YAML.
$mapping = @{
    "NA_Trans__SUPER_HANDOFF__LATEST.zip" = "bundles\NA_Trans__SUPER_HANDOFF__LATEST.zip";
    "AllChannels_PID_Name_Map__bundle_2025-11-04.zip" = "channel_maps\AllChannels_PID_Name_Map__bundle_2025-11-04.zip";
    "General Information.pdf" = "docs\reference\General Information.pdf";
    "transtables.zip" = "bundles\transtables.zip";
    "Metrics for Evaluating Automatic Transmission Shift Quality and Harshness.pdf" = "docs\reference\Metrics for Evaluating Automatic Transmission Shift Quality and Harshness.pdf";
    "Engine.pdf" = "docs\reference\Engine.pdf";
    "2015_tahoe_t43_trans_tables.yaml" = "examples\2015_tahoe_t43_trans_tables.yaml";
    "Transmission.pdf" = "docs\reference\Transmission.pdf";
    "Driveline-Axle.pdf" = "docs\reference\Driveline-Axle.pdf";
    "PIPELINE_README__NA_trans.md" = "docs\PIPELINE_README__NA_trans.md";
    "SlipTables__RPMxTORQUE__9x9__HowTo_and_Tools.zip" = "bundles\SlipTables__RPMxTORQUE__9x9__HowTo_and_Tools.zip";
    "NA_Tuning_Project_HandOff__2025-11-04_0722.zip" = "bundles\NA_Tuning_Project_HandOff__2025-11-04_0722.zip";
    "trans_service.txt" = "docs\trans_service.txt";
    "transcripts.txt" = "docs\transcripts.txt";
    # SKIP: "2015_tahoe_t43_trans_tables.zip" (duplicate of the YAML)
    "EC³ Torque Converter Clutch Strategy for a 6L80 in a Heavy SUV.pdf"  = "docs\reference\ECA3 Torque Converter Clutch Strategy for a 6L80 in a Heavy SUV.pdf";
    "ECA3 Torque Converter Clutch Strategy for a 6L80 in a Heavy SUV.pdf" = "docs\reference\ECA3 Torque Converter Clutch Strategy for a 6L80 in a Heavy SUV.pdf";
    "NA_Trans_Project_Memories_Export__2025-11-19.txt" = "docs\NA_Trans_Project_Memories_Export__2025-11-19.txt";
    "NA_Trans_Project_README__Export_Notes.txt" = "docs\NA_Trans_Project_README__Export_Notes.txt";
}

$imported = @()
$skippedExisting = @()
$missing = @()

foreach ($entry in $mapping.GetEnumerator()) {
    $srcRel = $entry.Key
    $dstRel = $entry.Value

    $srcPath = Join-Path $tempDir $srcRel
    $dstPath = Join-Path $repoRoot $dstRel
    $dstDir  = Split-Path $dstPath -Parent

    if (-not (Test-Path $srcPath)) {
        Write-Host "WARNING: Source not found in export: $srcRel" -ForegroundColor Yellow
        $missing += $srcRel
        continue
    }

    if (-not (Test-Path $dstDir)) {
        New-Item -ItemType Directory -Path $dstDir | Out-Null
    }

    if (Test-Path $dstPath) {
        Write-Host "Skipping (already exists): $dstRel" -ForegroundColor DarkYellow
        $skippedExisting += $dstRel
        continue
    }

    Copy-Item $srcPath $dstPath
    Write-Host "Imported: $srcRel -> $dstRel"
    $imported += $dstRel
}

# Clean up temp directory
Write-Host "Cleaning up temp dir $tempDir ..."
Remove-Item $tempDir -Recurse -Force

Write-Host ""
Write-Host ">>> Import summary:" -ForegroundColor Cyan
Write-Host "Imported files:"
$imported | ForEach-Object { Write-Host "  $_" }

if ($skippedExisting.Count -gt 0) {
    Write-Host ""
    Write-Host "Skipped (already existed in repo):" -ForegroundColor DarkYellow
    $skippedExisting | ForEach-Object { Write-Host "  $_" }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "Missing in export (not copied):" -ForegroundColor Yellow
    $missing | ForEach-Object { Write-Host "  $_" }
}

Write-Host ""
Write-Host ">>> Done. Next suggested steps (you can run manually):" -ForegroundColor Green
Write-Host "  git status"
Write-Host "  git add ."
Write-Host "  git commit -m 'Import NA Trans project export files'"
Write-Host "  git push"
