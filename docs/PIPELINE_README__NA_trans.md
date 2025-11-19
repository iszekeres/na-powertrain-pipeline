# End-to-end CSV pipeline (NA engine + Transmission)

This package defines exactly **what we need** to take any raw CSV, **clean it**, **map channels**, and **produce analysis-ready data** every time.

## 0) File organization (recommended)
```
/01_Planning
/02_Baseline
/03_Parts & Invoices
/04_Tuning Sessions/2025-11-04_Hwy/Run_01
/05_Maps & Calibrations
/06_Logs/Trans_Review
/07_Dyno & Performance
/08_Wiring & CAD
/09_Photos & Video
/10_Docs & Notes
/ZZ_Archive
```
File names: `<Area>__YYYY-MM-DD__Car__Detail__vNN.ext`

## 1) Capture the right channels
Use the master YAML: **channel_map_master__NA_trans__2025-11-04_0623.yaml**  
Make sure **both** “Trans Current Gear” columns are present (logger will duplicate one as `.1`).

## 2) Cleaners to run
- **Engine/Fueling/General:** `clean_log_NA.py`  → outputs `__clean_full__` and `__clean_core__`
- **Transmission focus:** `trans_clean_analyze.py` → outputs `__trans_focus__clean__`, `__shift_events__`, mapping, summary

> If you don't have these yet, say the word and I'll regenerate both scripts for you.

## 3) Standardized columns (always present after cleaning)
- time_s, rpm, vehicle_speed, throttle_pos_pct, accel_pedal_pct
- map_hires_kpa/map_kpa, baro_kpa, vacuum_kpa, vacuum_inHg
- lambda_meas, lambda_cmd, afr_cmd, afr_measured, lambda_error
- stft_b1_pct/2, ltft_b1_pct/2
- spark_deg, knock_retard_deg, vbatt_v, fuel_rail_kpa, iat_c, ect_c
- trans_gear_requested, trans_gear_actual, trans_ratio
- turbine_rpm, input_rpm, output_rpm, trans_slip_rpm
- tcc_slip_rpm, tcc_desired_slip_rpm, tcc_line_pressure, PCS1–5, fill_pressure_cmd, oncoming_clutch
- trans_fluid_temp_c, brake_pressure, time_of_latest_shift

## 4) Derived channels (done by the cleaners)
- `vacuum_kpa`, `vacuum_inHg`
- `afr_measured` (ethanol-aware), `lambda_error`
- `tcc_locked` (|slip| < 50 rpm)
- `implied_ratio` (rpm / output_rpm when locked)
- `ratio_error` vs `trans_ratio`

## 5) Quality checks (what “good” looks like)
- **Steady gear mismatch** < **1%**
- **Abs ratio error p95** < **2–3%** (locked)
- **TCC locked fraction** high on cruise; slip stable under load
- **KR p95** < **1°**; trims cruise median within **±5%**
- **VBatt σ** < **0.2 V**; **Fuel rail** droop < **5%** at WOT

## 6) Outputs you should see
From `clean_log_NA.py`:
- `...__clean_full__NA__v01.csv` (all original + standardized + derived)
- `...__clean_core__NA__v01.csv` (analysis subset)
- `clean_summary__<base>.txt` and quick plots

From `trans_clean_analyze.py`:
- `...__trans_focus__clean__<ts>.csv`
- `...__trans_focus__shift_events__<ts>.csv`
- `...__trans_focus__mapping__<ts>.csv`
- `...__trans_focus__summary__<ts>.txt`

## 7) One-liner to run both (PowerShell)
See `run_pipeline_template__2025-11-04_0623.ps1` for a copy/paste helper.

## 8) RWHP & Tire diameter
- Tire diameter uses **Output RPM + Vehicle Speed + Final Drive** (from YAML).
- RWHP from a clean, locked pull (we can add a script once you pick the segment).

---

If you change tires, diff, or add sensors, update the YAML and re-run. Keep the YAML and summaries under `/10_Docs & Notes` and the cleaned CSVs under `/06_Logs`.
