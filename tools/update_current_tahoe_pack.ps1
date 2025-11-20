param(
    [string]$BundlesDir = ".\bundles",
    [string]$Prefix     = "Tahoe_6L80_Pack__"
)

$bundlesPath = Resolve-Path $BundlesDir -ErrorAction Stop

Write-Host "[INFO] Looking for Tahoe packs in $bundlesPath with prefix '$Prefix'..."

$packs = Get-ChildItem -Path $bundlesPath -Filter "${Prefix}*.zip" -File | Sort-Object LastWriteTime -Descending

if (-not $packs -or $packs.Count -eq 0) {
    Write-Host "[ERROR] No matching packs found (pattern: ${Prefix}*.zip)."
    exit 1
}

$chosen = $packs[0]
Write-Host "[INFO] Latest pack found: $($chosen.Name)"

$currentFile = Join-Path $bundlesPath "CURRENT_TAHOE_PACK.txt"

$content = @()
$content += "# Current Tahoe 6L80 table pack"
$content += "# Updated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
$content += "# Directory: $($bundlesPath)"
$content += ""
$content += $chosen.Name

Set-Content -Path $currentFile -Value $content -Encoding UTF8

Write-Host "[OK] Updated CURRENT_TAHOE_PACK.txt -> $($chosen.Name)"
