# NA Trans Pipeline (Log → CLEAN_FULL Stage)

Repo layout: c:/tuning/processing is now the git root; keep everything (scripts, refs, raw/processed logs, docs) there so future GPT sessions can pick up cleanly.

Logging requirements: Always marshal channels from channel_map_master__NA_trans (MPH units, final drive 3.08); required cols include speed_mph, throttle/pedal %, gear_actual/gear_cmd, engine/Turbine/Input/Output RPM, brake pressure, temps (ECT/TFT). Keep both duplicated “Trans Current Gear” inputs so the canonicalizer can choose the right signal.

Cleaning flow:
- Copy raw CSV batch into logs_raw\<session>\.
- Run clean_log_NA.py to produce __clean_full__ (or __trans_focus__clean_FULL__) in logs_processed\<session>\cleaned\; this step validates headers, builds time_s, brake flag, gear_*__canon, turbine/output RPMs, slip/states, etc., and must fail hard if a required column is missing.
- Feed those FULL files into the trans_clean_analyze script (the rebuilt SAFE version) to generate __trans_focus__shift_events__, __trans_focus__mapping__, and __trans_focus__summary__ inside logs_processed\<session>\output\00_cleaner\.

Quality checks use the summary/mapping outputs: steady-gear mismatch <1%, ABS ratio error p95 <2–3% when locked, healthy TCC lock/slip behavior, no dragging at high temps, KR p95 <1° cruise, LTFT/STFT ±5%, VBatt/fuel stability. Flag sessions for mechanical/logging fixes if any check fails.

Bookkeeping: save every FULL clean, all trans_focus files, and the summary text/plots. Log metadata (date, session label, log types, mode, anomalies) inside docs/LOG_SESSIONS__INDEX.txt. Use the CLEAN_FULL outputs for all downstream passes (shift tables, TCC builds, overlay/audit scripts).

Next steps: after QC, run the downstream scripts under scripts/pipeline/ and scripts/tcc/ (e.g., shift_table_builder_Throttle17.py, tcc_table_builder_Throttle17__FIX.py, overlay_polish_v3.py, table_audit_and_fix.py) using the FULL cleaned data. This doc covers only the log → CLEAN_FULL stage; layer the shift/TCC/pass runs on top once the cleans are stable.
