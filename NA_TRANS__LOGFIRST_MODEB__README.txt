NA TRANS: LOG-FIRST MODE-B PIPELINE
===================================

This bundle contains the scripts and helpers needed to run the
NA TRANS / LOG-FIRST / MODE-B pipeline starting from:

- Raw CSV logs in:      .\newlogs\*.csv
- Baseline SHIFT/TCC in .\newlogs\baseline_current\

The pipeline:
  1) Cleans prior outputs under newlogs (keeps raw CSV + baseline_current).
  2) Runs CLEAN_FULL on all .\newlogs\*.csv.
  3) Canonicalizes gears to gear_actual__canon / gear_cmd__canon.
  4) Builds RAWEDGES SHIFT from gear_actual__canon (strict headers).
  5) Builds TCC APPLY/RELEASE from engine–turbine slip (50/120 rpm).
  6) Runs passes: STOPGO, KICKDOWN, CORNER (core+chassis+combined), LAT, INTENT.
  7) Blends RAWEDGES + TCC + passes over baseline_current (MODE-B logic).
  8) Assembles a flash pack and ZIP under:
       .\newlogs\output\03_flash_pack\

How to run
----------

1) Drop your new raw logs (.csv) into:

       C:\tuning\logs\newlogs

   Keep/refresh the baseline tables in:

       C:\tuning\logs\newlogs\baseline_current

2) Open a PowerShell prompt at:

       C:\tuning\logs

3) Paste and run the commands from:

       NA_TRANS__LOGFIRST_MODEB__RUN_CMD.txt

   That file contains a single PowerShell block that:
     - Cleans newlogs\cleaned, newlogs\output, newlogs\_staging_Review.
     - Runs CLEAN_FULL and all pipeline steps.
     - Writes the final flash pack ZIP under:
         newlogs\output\03_flash_pack\NA_TRANS__LOGFIRST_MODEB__FLASH_PACK__{timestamp}.zip

Notes
-----

- The RAWEDGES builder uses:
    time_s, speed_mph__canon, throttle_pct__canon, gear_actual__canon, brake
  with strict/no-fallback header checks.

- Brake is considered "ON" only when brake pressure > 150 kPa; edges with
  heavier braking are skipped.

- SHIFT/TCC blending enforces:
    - Downshift mph <= upshift mph - 1.0
    - TCC Release >= Apply + 1.1 mph
    - Preserves 317/318 sentinels from the baseline tables.

