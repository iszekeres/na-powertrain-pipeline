============================================================
NA_Trans Weekly Epoch Logging & Pipeline Policy
============================================================

Goal
----
Define a simple, repeatable pattern for using the NA_Trans LOG-FIRST / MODE-B pipeline with *short daily logs*, while keeping the tune stable and the data clean.

We treat each **week** as an "epoch":

- Run the same transmission tables for the whole week.
- Collect logs all week.
- At the end of the week, run the full pipeline over that week’s logs.
- Flash the new tables for the next week.
- Repeat.

This avoids "chasing noise" from day to day while still benefiting from daily logging.

----------------------------------------------------------------
1. Concepts (vocabulary)
----------------------------------------------------------------

- Epoch:
  A block of time (typically 1 week) where the flashed transmission tables remain fixed.

- Training logs:
  The set of logs we actually use as inputs to the NA_Trans LOG-FIRST pipeline for a given epoch.

- Daily logs:
  Any individual log file (even short logs, like 20 minutes) recorded during the epoch.

- Full pipeline:
  The NA_Trans LOG-FIRST / MODE-B / SMOOTH / XGEAR run described in:
  - NA_TRANS__LOGFIRST_MODEB__README.txt
  - PIPELINE_README__NA_trans.md
  using the scripts in this repo:
    - clean_full_only_wrapper__PB_MP__QUIET.py
    - trans_clean_analyze__SAFE_REBUILD.py
    - na_trans_step2_gear_canon.py
    - na_trans_step3_rawedges_shift.py
    - na_trans_step4_tcc_from_slip_50_120.py
    - na_trans_step5_build_baseline_logfirst.py
    - na_trans_step5_run_passes.ps1
    - na_trans_step6_blend_modeb.py
    - table_audit_and_fix.py
    - na_trans_smooth_tables.py
    - tsv_escape_guard.py
    - force_1dp__STRICT_SHIFT.py
    - force_1dp__STRICT_TCC.py
    - na_trans_crossgear_fix.py
    - tcc_pack_force_1dp.py
    - na_trans_scan_harsh_shifts.py

----------------------------------------------------------------
2. Weekly epoch loop (high level)
----------------------------------------------------------------

We run a simple cycle:

1) Start of week (Epoch N):
   - Flash the latest SHIFT/TCC tables produced by the pipeline.
   - These tables stay active (unchanged) for the entire epoch.

2) During the week:
   - Drive normally.
   - Take multiple short logs (e.g., 20–40 minutes) as daily usage allows.
   - Save all raw logs into the "epoch folder" under the data root
     (example convention, not enforced by scripts):

       C:\tuning\na-trans-data\newlogs\EPOCH_YYYY-MM-DD\raw\*.csv

   - Optionally run *diagnostic-only* tools (like harsh-shift scan) on recent logs to catch obvious problems. These diagnostics do NOT automatically change the tables.

3) End of week (Epoch N):
   - Treat all logs from this epoch as the **training logs** for the next update.
   - Run the full NA_Trans LOG-FIRST pipeline over this epoch’s logs.
   - The pipeline builds updated SHIFT/TCC tables (BLENDED_LOGOVERBASE, post-smooth/XGEAR) and produces a FLASH PACK zip.

4) Start of next week (Epoch N+1):
   - Flash the new tables from the FLASH PACK generated at the end of Epoch N.
   - Repeat the process (collect logs, run full pipeline at end of the week, etc.).

----------------------------------------------------------------
3. Daily logging policy
----------------------------------------------------------------

Daily behavior during an epoch:

- It is fine to:
  - Record one or more short logs per day (e.g., commute, errands).
  - Clean newly collected logs to CLEAN_FULL format.
  - Run *diagnostic* scripts like:
    - na_trans_scan_harsh_shifts.py
    - (optionally) selected passes in "read-only" mode, just to look at deltas.

- It is NOT recommended to:
  - Rebuild and flash new SHIFT/TCC tables *every day* using only that day’s log.
  - Mix many different tunes into a single training run.

Reason: Daily-only logs are often too short for good coverage (few hits per cell), and constant table changes make it harder to separate driver behavior from tune behavior. Weekly epochs let the statistics stabilize.

----------------------------------------------------------------
4. How old logs are used
----------------------------------------------------------------

Policy for old logs:

- Primary training data:
  - For "Epoch N+1" tuning, we primarily use logs recorded while "Epoch N" tables are active.
  - Logs from older tunes (Epoch N-1, N-2, ...) may be stored, but we do not blindly blend them into the current training run.

- Old logs still have value for:
  - Baseline comparisons:
    - Comparing how a shift behaved several tunes ago vs. now.
  - Regression hunting:
    - If something feels worse, we can compare previous epochs’ tables and behavior.
  - Coverage/usage analysis:
    - Seeing which cells never get used in real driving.

- Recommendation:
  - Keep old raw logs and CLEAN_FULL logs in dated subfolders.
  - Use them as reference datasets rather than as core training data once the tune has changed significantly.

----------------------------------------------------------------
5. Suggested folder layout (data side, not in Git)
----------------------------------------------------------------

The code repo (this repo) is *code-only*. Data is stored outside Git, e.g.:

  C:\tuning\na-trans-data\newlogs\
    EPOCH_2025-11-10\
      raw\               # Raw logs for this epoch
      cleaned\           # CLEAN_FULL files
      output\            # Tables, passes, flash packs for this epoch

The code (scripts) still typically see a legacy junction:

  C:\tuning\logs\newlogs   -->   C:\tuning\na-trans-data\newlogs

You can create an epoch folder and point the scripts at it as needed, but this README is about the *process*, not the exact on-disk layout.

----------------------------------------------------------------
6. Practical rules of thumb
----------------------------------------------------------------

- Keep tables stable for about a week at a time.
- Log as much as you like during the week; more coverage is always better.
- At the end of the week, run the full LOG-FIRST / MODE-B pipeline over that week’s logs.
- Flash the new tables and start a new epoch.
- Use older epochs’ logs for comparison/debug, not as primary training data, once the tune has moved on.

This policy balances:
- The need for **stable statistics** (enough hits per cell).
- The desire to **log daily** and catch problems early.
- The need to avoid **chasing noise** by changing tables too frequently.
============================================================

