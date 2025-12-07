#!/usr/bin/env python3
"""
highway_trans_MAX_analysis.py

Single-entry HP Tuners highway analysis pipeline for 6L80 shift / TCC tuning.

Usage (Windows-friendly):
    python highway_trans_MAX_analysis.py inbound.csv outbound1.csv outbound2.csv --out-dir highway_MAX_analysis

Goals
-----
- Read raw HP Tuners CSVs (keep all columns).
- Build rich derived/prepped signals (time, gear, motion, TCC, torque, fuel/air/spark/temps, stability).
- Run multiple analysis modules (shift/TCC behavior, torque/pressure, fuel/air/spark, ABS/TCS, DFCO, pedal behavior, kickdown/intent/latency, etc.).
- Emit CSV/JSON/text reports into one output folder, then zip it.

Notes
-----
- Strict/no-fallback for required signals. Optional signals are used when present; otherwise we warn.
- Alias resolution is centralized and case-insensitive.
- This script is intentionally modular; each stage is implemented as a function.
- Designed to be “conceptual but runnable” on large logs. For very large logs you may need plenty of RAM.
"""
from __future__ import annotations

VERSION = "highway_trans_MAX_analysis v0.4 patched"

import argparse
import json
import sys
import textwrap
import zipfile
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# -------------------------------------------------------------
# Expected outputs checklist
# -------------------------------------------------------------
EXPECTED_MODULE_OUTPUTS = {
    "GEAR_USAGE": ["ALL__gear_usage.csv"],
    "TCC_TIME_BUDGET": ["ALL__tcc_time_budget.csv"],
    "TCC_SLIP_DIST": ["ALL__tcc_slip_distribution.csv"],
    "SHIFT_EVENTS": ["ALL__shift_events.csv", "ALL__shift_points.csv"],
    "PEDAL_USAGE": [
        "inbound__prepped__pedal_usage_overall.csv",
        "outbound1__prepped__pedal_usage_overall.csv",
        "outbound2__prepped__pedal_usage_overall.csv",
    ],
    "DFCO": [
        "inbound__prepped__dfco_segments.csv",
        "outbound1__prepped__dfco_segments.csv",
        "outbound2__prepped__dfco_segments.csv",
    ],
    "SHIFT_QUALITY": ["ALL__shift_quality_events.csv", "ALL__shift_quality_summary.csv"],
    "SHIFT_LATENCY": ["ALL__shift_latency_events.csv", "ALL__shift_latency_summary.csv"],
    "KICKDOWN": ["ALL__kickdown_events.csv", "ALL__kickdown_summary.csv"],
    "INTENT": ["ALL__intent_episodes.csv", "ALL__intent_summary.csv"],
    "TCC_LOCK_UNLOCK": ["ALL__tcc_lock_events.csv", "ALL__tcc_unlock_events.csv"],
    "TCC_DRAGGING": ["ALL__tcc_dragging_segments.csv"],
    "FUEL_MPG": ["ALL__fuel_usage_segments.csv", "ALL__fuel_mpg_vs_strategy_summary.csv"],
    "MID_PEDAL_6TH": ["ALL__6th_mid_pedal_passes.csv", "ALL__6th_mid_pedal_passes_summary.csv"],
    "TUNING_HINTS": ["TUNING_HINTS__shift_tcc.json"],
}

LAT_MAX_PEDAL_TO_CMD = 3.0
LAT_MAX_CMD_TO_ACT = 3.0
LAT_MAX_ACT_TO_END = 3.0
LAT_MAX_PEDAL_TO_FULL = 5.0

# -------------------------------------------------------------
# Alias definitions
# -------------------------------------------------------------

ALIAS = {
    "time": ["Offset"],
    "speed_mph": ["Vehicle Speed (SAE)", "Vehicle Speed"],
    "rpm": ["Engine RPM (SAE)", "Engine Speed"],
    "gear": ["Trans Current Gear"],
    "gear_cmd": ["Trans Current Gear.1", "Trans Current Gear_1", "Trans Current Gear.2"],
    "pedal_pct": ["Accelerator Pedal Position"],
    "throttle_pct": ["Throttle Position", "Throttle Desired Position"],
    "brake_kpa": ["Brake Pressure"],
    "trans_mode": ["Trans Shift Mode"],
    "tcc_slip": ["TCC Slip"],
    "tcc_desired_slip": ["TCC Desired Slip"],
    "tcc_line": ["TCC Line Pressure"],
    "turbine_rpm": ["Trans Turbine RPM", "Trans Input Shaft RPM"],
    "input_rpm": ["Trans Input Shaft RPM", "Trans Turbine RPM"],
    "output_rpm": ["Trans Output Shaft RPM"],
    "gear_ratio_calc": ["Trans Calculated Gear Ratio"],
    "trans_slip_rpm": ["Trans Slip RPM"],
    "pcs1": ["PCS 1 Cmd Pressure"],
    "pcs2": ["PCS 2 Cmd Pressure"],
    "pcs3": ["PCS 3 Cmd Pressure"],
    "pcs4": ["PCS 4 Cmd Pressure"],
    "pcs5": ["PCS 5 Cmd Pressure"],
    "fill_cmd": ["Fill Pressure Cmd"],
    "torque_delivered": ["Delivered Engine Torque"],
    "torque_engine": ["Engine Torque"],
    "torque_trans": ["Trans Engine Torque"],
    "torque_axle": ["Actual Axle Torque"],
    "torque_cmd_immediate": ["Immediate Engine Torque Cmd"],
    "torque_cmd_axle_immediate": ["Immediate Axle Torque Cmd"],
    "torque_pred_engine": ["Predicted Engine Torque Cmd"],
    "torque_pred_axle": ["Predicted Axle Torque Cmd"],
    "torque_req_axle_driver": ["Driver Final Axle Torque Req"],
    "torque_zero_pedal": ["Zero Pedal Engine Torque"],
    "torque_peak": ["Peak Engine Torque"],
    "torque_tcs_desired": ["TCS Desired Engine Torque"],
    "torque_tcs_axle_req": ["TCS Axle Torque Req"],
    "torque_tcs_engine_req": ["TCS Engine Torque Req"],
    "fuel_pressure": ["Fuel Pressure (SAE)", "Fuel Rail Pressure (SAE)"],
    "fuel_pressure_req": ["Fuel Pressure Requested", "Desired Fuel Pressure"],
    "inst_fuel_used": ["Inst Fuel Used"],
    "inst_fuel_flow": ["Instantaneous Fuel Flow Estimate"],
    "adv_fuel_flow": ["Advance Fuel Flow Estimate"],
    "inj_pw_b1": ["Injector Pulse Width Avg. Bank 1"],
    "inj_pw_b2": ["Injector Pulse Width Avg. Bank 2"],
    "inj_flow_rate": ["Injector Flow Rate"],
    "afr_cmd": ["Air-Fuel Ratio Commanded"],
    "eqr_cmd": ["Equivalence Ratio Commanded (SAE)"],
    "wb_eq2": ["WB EQ Ratio 2 (SAE)"],
    "stft_b1": ["Short Term Fuel Trim Bank 1 (SAE)"],
    "stft_b2": ["Short Term Fuel Trim Bank 2 (SAE)"],
    "ltft_b1": ["Long Term Fuel Trim Bank 1 (SAE)"],
    "ltft_b2": ["Long Term Fuel Trim Bank 2 (SAE)"],
    "cyl_airmass": ["Cylinder Airmass"],
    "maf_gps": ["Mass Airflow (SAE)"],
    "maf_state": ["Mass Airflow Sensor"],
    "dyn_airflow": ["Dynamic Airflow"],
    "ve_airflow": ["Volumetric Efficiency Airflow"],
    "ve_mgk_kpa": ["Volumetric Efficiency (mg•K/kPa)"],
    "map_kpa": ["Manifold Absolute Pressure - Hi-Res", "Intake Manifold Absolute Pressure (SAE)"],
    "baro_kpa": ["Barometric Pressure"],
    "abs_load": ["Absolute Load (SAE)"],
    "calc_load": ["Calculated Engine Load (SAE)"],
    "iat": ["Intake Air Temp (SAE)"],
    "mat": ["Manifold Air Temp"],
    "ect": ["Engine Coolant Temp (SAE)"],
    "oil_temp": ["Engine Oil Temp Calc"],
    "ambient": ["Ambient Air Temp"],
    "humidity": ["Relative Humidity"],
    "tft": ["Trans Fluid Temp"],
    "lat_g": ["Lateral Acceleration"],
    "yaw_rate": ["Yaw Rate"],
    "steer": ["Steering Wheel Position"],
    "abs_flag": ["ABS Active", "ABS_Active", "ABS_Event", "ABS Active Status"],
    "tcs_flag": ["Traction Control System", "TCS Active", "TCS_Active", "TCS Request", "TCS Desired Engine Torque", "TC Active", "StabiliTrak Active"],
    "dfco": ["DFCO Active"],
    "afm": ["AFM Mode", "DFM Mode", "AFM Active"],
    "oil_pressure": ["Engine Oil Pressure"],
}

REQUIRED_KEYS = ["time", "speed_mph", "rpm", "gear", "pedal_pct", "throttle_pct"]

# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------


def pick_col(df: pd.DataFrame, keys: Sequence[str], required: bool = False) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for k in keys:
        key_lower = k.lower()
        if key_lower in cols_lower:
            return cols_lower[key_lower]
    if required:
        raise KeyError(f"Missing required column; tried aliases: {keys}")
    return None


def resolve_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {}
    missing_required = []
    for logical, aliases in ALIAS.items():
        try:
            col = pick_col(df, aliases, required=logical in REQUIRED_KEYS)
        except KeyError as e:
            missing_required.append(str(e))
            col = None
        mapping[logical] = col
    if missing_required:
        raise KeyError("Required columns missing:\n" + "\n".join(missing_required))
    return mapping


def find_header_row(path: Path, max_rows: int = 40) -> int:
    """Heuristic: find the first line that looks like a header."""
    with path.open("r", errors="ignore") as f:
        for i in range(max_rows):
            line = f.readline()
            if not line:
                break
            if "Log File" in line or "Configuration" in line or "VCM Suite" in line:
                continue
            parts = line.strip().split(",")
            if len(parts) >= 5 and any(p.strip().lower() == "offset" for p in parts):
                return i
    return 0


def load_raw_csv(path: Path) -> pd.DataFrame:
    header_row = find_header_row(path)
    if header_row > 0:
        df = pd.read_csv(
            path,
            header=0,
            skiprows=header_row,
            low_memory=False,
            on_bad_lines="skip",
        )
    else:
        df = pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    if df.empty:
        raise ValueError(f"{path} loaded empty.")
    return df


def build_time(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    df = df.copy()
    t_col = mapping["time"]
    t_raw = pd.to_numeric(df[t_col], errors="coerce")
    df = df.loc[~t_raw.isna()].copy()
    t_arr = t_raw.dropna().to_numpy()
    if len(t_arr) == 0:
        raise ValueError("No valid time values after cleaning.")
    t_arr = t_arr - t_arr[0]
    # If clearly ms, scale
    if np.nanmax(t_arr) > 1e7:
        t_arr = t_arr / 1000.0
    df["time_s"] = t_arr
    dt = np.diff(t_arr, prepend=t_arr[0])
    med_dt = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.1
    dt[dt <= 0] = med_dt
    df["dt_s"] = dt
    return df


def debug_list_out_dir(out_dir: Path):
    print(f"[DEBUG] Listing files in {out_dir}")
    for p in sorted(out_dir.glob("*")):
        if p.is_file():
            print(f"  {p.name}  ({p.stat().st_size} bytes)")
    print("[DEBUG] ALL__* files:")
    for p in sorted(out_dir.glob("ALL__*")):
        print(f"  {p.name}  ({p.stat().st_size} bytes)")
    run_sum = out_dir / "RUN_SUMMARY__highway_trans_MAX.json"
    print(f"[DEBUG] Run summary exists: {run_sum.exists()}")
    for p in out_dir.glob("TUNING_HINTS__*.json"):
        print(f"[DEBUG] Tuning hints file: {p.name}")


def derive_gear(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    df = df.copy()
    gear_cols = [c for c in df.columns if c.lower().startswith("trans current gear")]
    if not gear_cols and mapping.get("gear"):
        gear_cols.append(mapping["gear"])
    if mapping.get("gear_cmd") and mapping["gear_cmd"] not in gear_cols:
        gear_cols.append(mapping["gear_cmd"])
    if not gear_cols:
        raise KeyError("No Trans Current Gear columns found.")

    gear_arrays = []
    for c in gear_cols:
        s = df[c].astype(str).str.strip()
        arr = []
        for v in s:
            try:
                fv = float(v)
                if 1 <= fv <= 6:
                    arr.append(int(fv))
                else:
                    arr.append(0)
            except Exception:
                arr.append(0)
        gear_arrays.append(np.array(arr, dtype=int))
    gear_raw = np.max(np.stack(gear_arrays, axis=1), axis=1)
    gear_ffill = pd.Series(gear_raw).replace(0, np.nan).ffill().fillna(0).astype(int).to_numpy()
    df["gear_actual__canon"] = gear_ffill
    if len(gear_arrays) > 1:
        df["gear_cmd__canon"] = gear_arrays[-1]
    else:
        df["gear_cmd__canon"] = gear_ffill
    return df


def add_core_signals(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    df = df.copy()
    df["speed_mph"] = df[mapping["speed_mph"]].astype(float)
    df["speed_mps"] = df["speed_mph"] * 0.44704
    df["engine_rpm"] = df[mapping["rpm"]].astype(float)
    df["pedal_pct"] = df[mapping["pedal_pct"]].astype(float)
    df["throttle_pct"] = df[mapping["throttle_pct"]].astype(float)
    # accel/jerk
    dt = df["dt_s"].to_numpy()
    v = df["speed_mps"].to_numpy()
    accel = np.diff(v, prepend=v[0]) / np.where(dt > 0, dt, 1)
    if len(accel) >= 3:
        accel = pd.Series(accel).rolling(3, center=True, min_periods=1).mean().to_numpy()
    df["accel_mps2"] = accel
    jerk = np.diff(accel, prepend=accel[0]) / np.where(dt > 0, dt, 1)
    if len(jerk) >= 3:
        jerk = pd.Series(jerk).rolling(3, center=True, min_periods=1).mean().to_numpy()
    df["jerk_mps3"] = jerk
    # pedal/throttle rates
    def rate(series):
        arr = series.to_numpy()
        return np.diff(arr, prepend=arr[0]) / np.where(dt > 0, dt, 1)
    df["pedal_rate_pct_per_s"] = rate(df["pedal_pct"])
    df["throttle_rate_pct_per_s"] = rate(df["throttle_pct"])
    # brake
    if mapping.get("brake_kpa"):
        brake = df[mapping["brake_kpa"]].astype(float)
        df["brake_kpa"] = brake
        df["brake_on"] = (brake >= 15.0).astype(int)
    else:
        df["brake_on"] = 0
    # trans mode
    if mapping.get("trans_mode"):
        df["trans_mode"] = df[mapping["trans_mode"]].astype(str).str.strip().str.lower()
    else:
        df["trans_mode"] = ""
    return df


def classify_tcc_state_vector(values: Sequence[float]) -> List[str]:
    state = []
    for v in values:
        if np.isnan(v):
            state.append("OPEN")
            continue
        slip_abs = abs(v)
        if slip_abs <= 50.0:
            state.append("LOCKED")
        elif slip_abs <= 120.0:
            state.append("SLIP")
        else:
            state.append("OPEN")
    return state


def add_tcc(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    df = df.copy()
    slip_logged = mapping.get("tcc_slip")
    turbine_col = mapping.get("turbine_rpm")
    if slip_logged:
        slip_series = pd.to_numeric(df[slip_logged], errors="coerce")
    elif turbine_col:
        slip_series = df["engine_rpm"].astype(float) - df[turbine_col].astype(float)
    else:
        slip_series = pd.Series(np.nan, index=df.index)
    df["tcc_slip_rpm_fused"] = slip_series
    df["tcc_state"] = pd.Categorical(
        classify_tcc_state_vector(slip_series.to_numpy()),
        categories=["OPEN", "SLIP", "LOCKED"],
    )
    return df


def add_torque(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    df = df.copy()
    # Engine/delivered/trans torque
    if mapping.get("torque_delivered"):
        df["eng_tq"] = df[mapping["torque_delivered"]].astype(float)
    elif mapping.get("torque_engine"):
        df["eng_tq"] = df[mapping["torque_engine"]].astype(float)
    elif mapping.get("torque_trans"):
        df["eng_tq"] = df[mapping["torque_trans"]].astype(float)
    else:
        df["eng_tq"] = np.nan
    if mapping.get("torque_trans"):
        df["trans_tq"] = df[mapping["torque_trans"]].astype(float)
    else:
        df["trans_tq"] = df["eng_tq"]
    if mapping.get("torque_axle"):
        df["axle_tq"] = df[mapping["torque_axle"]].astype(float)
    else:
        df["axle_tq"] = np.nan
    if mapping.get("torque_req_axle_driver"):
        df["driver_axle_req"] = df[mapping["torque_req_axle_driver"]].astype(float)
    else:
        df["driver_axle_req"] = np.nan
    return df


def add_temp(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    df = df.copy()
    for k, out in [("ect", "ect_deg"), ("tft", "tft_deg"), ("iat", "iat_deg"), ("ambient", "ambient_deg")]:
        if mapping.get(k):
            df[out] = df[mapping[k]].astype(float)
    return df


def apply_mode_filter(df: pd.DataFrame, mode_filter: Optional[str]) -> pd.DataFrame:
    if not mode_filter:
        return df
    if "trans_mode" not in df.columns:
        print("[WARN] Mode filter requested but trans_mode missing; using all rows.")
        return df
    mode_val = str(mode_filter).strip().lower()
    mask = df["trans_mode"].str.strip().str.lower() == mode_val
    filtered = df[mask].copy()
    print(f"[INFO] Mode filter {mode_filter!r}: kept {len(filtered)} / {len(df)} rows.")
    if filtered.empty:
        print("[WARN] No rows matched mode filter; analyses may be empty.")
    return filtered


# -------------------------------------------------------------
# Prepping
# -------------------------------------------------------------


def prep_log(raw_path: Path, out_dir: Path, mode_filter: Optional[str]) -> Path:
    print(f"[INFO] Loading raw log: {raw_path}")
    df_raw = load_raw_csv(raw_path)
    mapping = resolve_columns(df_raw)
    df = build_time(df_raw, mapping)
    df = derive_gear(df, mapping)
    df = add_core_signals(df, mapping)
    df = add_tcc(df, mapping)
    df = add_torque(df, mapping)
    df = add_temp(df, mapping)
    df["file_name"] = raw_path.name
    df = apply_mode_filter(df, mode_filter)
    prepped_path = out_dir / f"{raw_path.stem}__prepped.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(prepped_path, index=False)
    print(f"[OK] Prepped log written: {prepped_path}")
    return prepped_path


# -------------------------------------------------------------
# Analyses (selected core + requested modules)
# Many are condensed for manageability but preserve requested outputs.
# -------------------------------------------------------------


def gear_usage(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["dt_s"].astype(float)
    gear = df["gear_actual__canon"].astype(int)
    rows = []
    total = float(dt.sum())
    for g in sorted(pd.unique(gear)):
        mask = gear == g
        t = float(dt[mask].sum())
        pct = (t / total * 100.0) if total > 0 else 0.0
        rows.append({"gear": int(g), "time_s": t, "time_pct": pct})
    return pd.DataFrame(rows)


def detect_shift_events(df: pd.DataFrame) -> pd.DataFrame:
    gear = df["gear_actual__canon"].astype(int).to_numpy()
    time = df["time_s"].astype(float).to_numpy()
    rows = []
    event_id = 0
    for i in range(1, len(gear)):
        if gear[i] == gear[i - 1]:
            continue
        g0, g1 = gear[i - 1], gear[i]
        if g0 == 0 or g1 == 0:
            continue
        event_id += 1
        before = df.iloc[i - 1]
        after = df.iloc[i]
        rows.append(
            {
                "event_id": event_id,
                "from_gear": int(g0),
                "to_gear": int(g1),
                "time_start_s": float(before["time_s"]),
                "time_end_s": float(after["time_s"]),
                "duration_s": float(after["time_s"] - before["time_s"]),
                "speed_mph_before": float(before["speed_mph"]),
                "speed_mph_after": float(after["speed_mph"]),
                "pedal_pct_before": float(before["pedal_pct"]),
                "pedal_pct_after": float(after["pedal_pct"]),
                "throttle_pct_before": float(before["throttle_pct"]),
                "throttle_pct_after": float(after["throttle_pct"]),
                "engine_rpm_before": float(before["engine_rpm"]),
                "engine_rpm_after": float(after["engine_rpm"]),
                "tcc_state_before": str(before["tcc_state"]),
                "tcc_state_after": str(after["tcc_state"]),
                "trans_mode": str(before.get("trans_mode", "")),
            }
        )
    return pd.DataFrame(rows)


def build_shift_points(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, ev in events.iterrows():
        rows.append(
            {
                "event_id": int(ev["event_id"]),
                "from_gear": int(ev["from_gear"]),
                "to_gear": int(ev["to_gear"]),
                "speed_mph": float(ev["speed_mph_before"]),
                "pedal_pct": float(ev["pedal_pct_before"]),
                "throttle_pct": float(ev["throttle_pct_before"]),
                "trans_mode": str(ev.get("trans_mode", "")),
            }
        )
    return pd.DataFrame(rows)


def tcc_time_budget(df: pd.DataFrame, file_tag: str) -> pd.DataFrame:
    dt = df["dt_s"].astype(float)
    gear = df["gear_actual__canon"].astype(int)
    state = df["tcc_state"].astype(str)
    rows = []
    for g in sorted(gear.unique()):
        if g <= 0:
            continue
        mask_g = gear == g
        total = float(dt[mask_g].sum())
        if total <= 0:
            continue
        open_time = float(dt[mask_g & (state == "OPEN")].sum())
        slip_time = float(dt[mask_g & (state == "SLIP")].sum())
        locked_time = float(dt[mask_g & (state == "LOCKED")].sum())
        rows.append(
            {
                "file_name": file_tag,
                "gear": int(g),
                "time_open_s": open_time,
                "time_slip_s": slip_time,
                "time_locked_s": locked_time,
                "pct_open": 100.0 * open_time / total,
                "pct_slip": 100.0 * slip_time / total,
                "pct_locked": 100.0 * locked_time / total,
            }
        )
    return pd.DataFrame(rows)


def pedal_usage(df: pd.DataFrame, highway_only: bool = False) -> pd.DataFrame:
    if highway_only:
        mask = (df["gear_actual__canon"] >= 4) & (df["speed_mph"] >= 50)
        dfx = df[mask]
    else:
        dfx = df
    dt = dfx["dt_s"].astype(float)
    pedal = dfx["pedal_pct"].astype(float)
    bands = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, 80), (80, 1000)]
    rows = []
    total = float(dt.sum())
    for lo, hi in bands:
        mask = (pedal >= lo) & (pedal < hi)
        t = float(dt[mask].sum())
        pct = (t / total * 100) if total > 0 else 0
        rows.append({"band": f"{lo}-{hi}", "time_s": t, "time_pct": pct})
    return pd.DataFrame(rows)


def dfco_segments(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    dfco_col = mapping.get("dfco")
    if not dfco_col or dfco_col not in df.columns:
        return pd.DataFrame(columns=["file_name", "start_time_s", "end_time_s", "duration_s", "gear_min", "gear_max", "speed_min", "speed_max", "tcc_state_mode"])
    flag = df[dfco_col]
    active = flag.astype(str).str.lower().isin(["1", "true", "on"])
    time = df["time_s"].to_numpy()
    gear = df["gear_actual__canon"].astype(int).to_numpy()
    speed = df["speed_mph"].astype(float).to_numpy()
    tcc = df["tcc_state"].astype(str).to_numpy()
    rows = []
    i = 0
    while i < len(df):
        if not active.iloc[i]:
            i += 1
            continue
        start = i
        while i < len(df) and active.iloc[i]:
            i += 1
        end = i - 1
        dur = time[end] - time[start]
        rows.append(
            {
                "file_name": df["file_name"].iloc[0],
                "start_time_s": float(time[start]),
                "end_time_s": float(time[end]),
                "duration_s": float(dur),
                "gear_min": int(gear[start:end+1].min()),
                "gear_max": int(gear[start:end+1].max()),
                "speed_min": float(speed[start:end+1].min()),
                "speed_max": float(speed[start:end+1].max()),
                "tcc_state_mode": Counter(tcc[start:end+1]).most_common(1)[0][0],
            }
        )
    return pd.DataFrame(rows)


def abs_tcs_events(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    events = []
    systems = []
    col_abs = mapping.get("abs_flag")
    col_tcs = mapping.get("tcs_flag")
    if col_abs:
        systems.append(("ABS", col_abs))
    if col_tcs:
        systems.append(("TCS", col_tcs))
    time = df["time_s"].to_numpy()
    speed = df["speed_mph"].astype(float).to_numpy()
    pedal = df["pedal_pct"].astype(float).to_numpy()
    throttle = df["throttle_pct"].astype(float).to_numpy()
    gear = df["gear_actual__canon"].astype(int).to_numpy()
    heavy = pedal >= 55
    very = pedal >= 80
    for sys_name, col in systems:
        flag_raw = df[col]
        active = flag_raw.astype(str).str.lower().isin(["1", "true", "on"])
        i = 0
        while i < len(df):
            if not active.iloc[i]:
                i += 1
                continue
            start = i
            while i < len(df) and active.iloc[i]:
                i += 1
            end = i - 1
            dur = time[end] - time[start]
            idx = slice(start, end + 1)
            events.append(
                {
                    "file_name": df["file_name"].iloc[0],
                    "system": sys_name,
                    "start_time_s": float(time[start]),
                    "end_time_s": float(time[end]),
                    "duration_s": float(dur),
                    "start_speed_mph": float(speed[start]),
                    "end_speed_mph": float(speed[end]),
                    "max_speed_mph": float(np.nanmax(speed[idx])),
                    "min_speed_mph": float(np.nanmin(speed[idx])),
                    "max_pedal_pct": float(np.nanmax(pedal[idx])),
                    "avg_pedal_pct": float(np.nanmean(pedal[idx])),
                    "max_throttle_pct": float(np.nanmax(throttle[idx])),
                    "avg_throttle_pct": float(np.nanmean(throttle[idx])),
                    "gear_at_start": int(gear[start]),
                    "gear_at_end": int(gear[end]),
                    "heavy_pedal": bool(np.any(heavy[idx])),
                    "very_heavy_pedal": bool(np.any(very[idx])),
                    "low_speed_event": bool(np.nanmax(speed[idx]) < 25),
                    "high_speed_event": bool(np.nanmax(speed[idx]) >= 60),
                }
            )
    return pd.DataFrame(events)


# -------------------------------------------------------------
# New modules: shift quality, shift latency, kickdown, intent,
# lock/unlock, dragging, fuel/mpg, tuning hints.
# These are simplified yet functional.
# -------------------------------------------------------------

def shift_quality(all_events: pd.DataFrame, prepped: Dict[str, pd.DataFrame], out_dir: Path):
    if all_events.empty:
        pd.DataFrame().to_csv(out_dir / "ALL__shift_quality_events.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "ALL__shift_quality_summary.csv", index=False)
        return
    rows = []
    for _, ev in all_events.iterrows():
        fname = ev["file_name"]
        df = prepped.get(fname)
        if df is None:
            continue
        t0 = ev["time_start_s"]
        t1 = ev["time_end_s"]
        pre_mask = (df["time_s"] >= t0 - 0.5) & (df["time_s"] < t0)
        dur_mask = (df["time_s"] >= t0) & (df["time_s"] <= t1)
        post_mask = (df["time_s"] > t1) & (df["time_s"] <= t1 + 1.0)
        accel = df["accel_mps2"].astype(float)
        jerk = df["jerk_mps3"].astype(float)
        accel_pre = float(accel[pre_mask].mean()) if pre_mask.any() else np.nan
        accel_min = float(accel[dur_mask].min()) if dur_mask.any() else np.nan
        accel_post = float(accel[post_mask].mean()) if post_mask.any() else np.nan
        jerk_min = float(jerk[dur_mask].min()) if dur_mask.any() else np.nan
        jerk_max = float(jerk[dur_mask].max()) if dur_mask.any() else np.nan
        torque_hole = accel_pre - accel_min if not np.isnan(accel_pre) and not np.isnan(accel_min) else np.nan
        if np.isnan(torque_hole):
            label = "unknown"
        elif torque_hole > 1.5 or jerk_min < -4:
            label = "harsh"
        elif torque_hole > 0.8:
            label = "ok"
        else:
            label = "comfy"
        rows.append(
            dict(
                ev,
                accel_pre_avg=accel_pre,
                accel_min_during=accel_min,
                accel_post_avg=accel_post,
                torque_hole_depth=torque_hole,
                jerk_min_during=jerk_min,
                jerk_max_during=jerk_max,
                quality_label=label,
            )
        )
    dfq = pd.DataFrame(rows)
    dfq.to_csv(out_dir / "ALL__shift_quality_events.csv", index=False)
    # summary
    if dfq.empty:
        pd.DataFrame().to_csv(out_dir / "ALL__shift_quality_summary.csv", index=False)
    else:
        summ = (
            dfq.groupby(["from_gear", "to_gear", "quality_label"])["event_id"]
            .count()
            .reset_index(name="count")
        )
        summ.to_csv(out_dir / "ALL__shift_quality_summary.csv", index=False)


def shift_latency(all_events: pd.DataFrame, prepped: Dict[str, pd.DataFrame], out_dir: Path):
    def edges_for_log(events_df: pd.DataFrame, log_df: pd.DataFrame) -> List[Dict]:
        time = log_df["time_s"].to_numpy()
        gear_act = log_df["gear_actual__canon"].astype(int).to_numpy()
        gear_cmd = (
            log_df["gear_cmd__canon"].astype(int).to_numpy()
            if "gear_cmd__canon" in log_df.columns
            else gear_act
        )
        pedal = log_df["pedal_pct"].astype(float).to_numpy()
        speed = log_df["speed_mph"].astype(float).to_numpy()
        rows_local: List[Dict] = []

        for _, ev in events_df.iterrows():
            fg, tg = int(ev["from_gear"]), int(ev["to_gear"])
            t_event = float(ev["time_start_s"])

            # Find command edge: first gear_cmd change after event start
            cmd_idx = None
            start_idx = np.searchsorted(time, t_event)
            for i in range(max(0, start_idx - 5), min(len(time), start_idx + 500)):
                if i > 0 and gear_cmd[i] != gear_cmd[i - 1]:
                    cmd_idx = i
                    break
            if cmd_idx is None:
                cmd_idx = start_idx
            t_cmd = time[cmd_idx]

            # Pedal start: walk back until pedal drops at least 3% from pedal_cmd or window 1s
            pedal_cmd = pedal[cmd_idx]
            thresh = pedal_cmd - 3.0
            p_idx = cmd_idx
            while p_idx > 0 and time[p_idx] >= t_cmd - 1.0:
                if pedal[p_idx - 1] <= thresh:
                    p_idx -= 1
                    break
                p_idx -= 1
            t_pedal = time[p_idx]

            # Actual start: first gear_actual change from from_gear after command
            act_start_idx = None
            for i in range(cmd_idx, len(time)):
                if time[i] - t_cmd > 4.0:
                    break
                if gear_act[i] != fg:
                    act_start_idx = i
                    break
            if act_start_idx is None:
                continue
            t_act_start = time[act_start_idx]

            # Actual end: dwell in to_gear for >=3 samples or 0.2 s
            act_end_idx = None
            dwell_count = 0
            dwell_start = None
            for i in range(act_start_idx, len(time)):
                if time[i] - t_act_start > 4.0:
                    break
                if gear_act[i] == tg:
                    if dwell_count == 0:
                        dwell_start = time[i]
                    dwell_count += 1
                    if dwell_count >= 3 and (time[i] - dwell_start) >= 0.2:
                        act_end_idx = i
                        break
                else:
                    dwell_count = 0
                    dwell_start = None
            if act_end_idx is None:
                continue
            t_act_end = time[act_end_idx]

            # Latencies with guardrails
            lat_p2c = t_cmd - t_pedal
            lat_c2a = t_act_start - t_cmd
            lat_a2e = t_act_end - t_act_start
            lat_full = t_act_end - t_pedal
            if not (0 <= lat_p2c <= LAT_MAX_PEDAL_TO_CMD):
                continue
            if not (0 <= lat_c2a <= LAT_MAX_CMD_TO_ACT):
                continue
            if not (0 < lat_a2e <= LAT_MAX_ACT_TO_END):
                continue
            if not (0 < lat_full <= LAT_MAX_PEDAL_TO_FULL):
                continue

            rows_local.append(
                {
                    "file_name": ev["file_name"],
                    "event_id": ev["event_id"],
                    "from_gear": fg,
                    "to_gear": tg,
                    "speed_cmd_mph": float(speed[cmd_idx]),
                    "pedal_cmd_pct": float(pedal_cmd),
                    "t_pedal_start": t_pedal,
                    "time_cmd": t_cmd,
                    "t_actual_start": t_act_start,
                    "t_actual_end": t_act_end,
                    "latency_pedal_to_cmd": lat_p2c,
                    "latency_cmd_to_actual_start": lat_c2a,
                    "latency_actual_start_to_end": lat_a2e,
                    "latency_pedal_to_full_shift": lat_full,
                }
            )
        return rows_local

    rows: List[Dict] = []
    for fname, ev_df in all_events.groupby("file_name"):
        log_df = prepped.get(fname)
        if log_df is None:
            continue
        rows.extend(edges_for_log(ev_df, log_df))

    dfl = pd.DataFrame(rows)
    dfl.to_csv(out_dir / "ALL__shift_latency_events.csv", index=False)
    if dfl.empty:
        pd.DataFrame().to_csv(out_dir / "ALL__shift_latency_summary.csv", index=False)
    else:
        dfl["speed_band"] = (dfl["speed_cmd_mph"] // 5 * 5).astype(int)
        dfl["pedal_band"] = pd.cut(
            dfl["pedal_cmd_pct"],
            bins=[0, 20, 50, 80, 120],
            labels=["0-20", "20-50", "50-80", "80-100"],
        )
        agg = {
            "latency_pedal_to_cmd": ["count", "median", "mean", "min", "max"],
            "latency_cmd_to_actual_start": ["median", "mean", "min", "max"],
            "latency_actual_start_to_end": ["median", "mean", "min", "max"],
            "latency_pedal_to_full_shift": ["median", "mean", "min", "max"],
        }
        summ = dfl.groupby(["from_gear", "to_gear", "pedal_band", "speed_band"]).agg(agg)
        summ.columns = ["_".join([c for c in col if c]).strip("_") for col in summ.columns]
        summ = summ.reset_index()
        summ.to_csv(out_dir / "ALL__shift_latency_summary.csv", index=False)


def detect_kickdowns(prepped: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    events = []
    for fname, df in prepped.items():
        time = df["time_s"].to_numpy()
        pedal = df["pedal_pct"].astype(float).to_numpy()
        gear = df["gear_actual__canon"].astype(int).to_numpy()
        speed = df["speed_mph"].astype(float).to_numpy()
        rate = df["pedal_rate_pct_per_s"].astype(float).to_numpy() if "pedal_rate_pct_per_s" in df else np.diff(pedal, prepend=pedal[0]) / np.diff(time, prepend=1)
        n = len(df)
        eid = 0
        for i in range(n):
            if rate[i] >= 20 and pedal[i] >= 30:
                # look ahead 0.5s for +20 pedal
                j = i
                while j < n and time[j] - time[i] <= 0.5:
                    j += 1
                if pedal[j - 1] - pedal[i] < 20:
                    continue
                # look ahead 3s for downshift
                k = i
                min_gear = gear[i]
                t_change = np.nan
                while k < n and time[k] - time[i] <= 3.0:
                    if gear[k] < min_gear:
                        min_gear = gear[k]
                        if np.isnan(t_change):
                            t_change = time[k]
                    k += 1
                if min_gear < gear[i]:
                    eid += 1
                    events.append(
                        {
                            "file_name": fname,
                            "event_id": eid,
                            "gear_before": gear[i],
                            "gear_after": min_gear,
                            "time_pedal_start_s": time[i],
                            "time_gear_change_s": t_change,
                            "speed_start_mph": speed[i],
                            "pedal_start_pct": pedal[i],
                            "pedal_peak_pct": float(np.nanmax(pedal[i:k])),
                            "latency_pedal_to_actual": t_change - time[i] if not np.isnan(t_change) else np.nan,
                        }
                    )
    return pd.DataFrame(events)


def summarize_kickdowns(dfk: pd.DataFrame, out_dir: Path):
    dfk.to_csv(out_dir / "ALL__kickdown_events.csv", index=False)
    if dfk.empty:
        pd.DataFrame().to_csv(out_dir / "ALL__kickdown_summary.csv", index=False)
        return
    dfk["speed_band"] = (dfk["speed_start_mph"] // 10 * 10).astype(int)
    dfk["pedal_band"] = pd.cut(dfk["pedal_peak_pct"], bins=[0, 40, 60, 80, 120], labels=["0-40", "40-60", "60-80", "80-100"])
    summ = dfk.groupby(["gear_before", "speed_band", "pedal_band"])["event_id"].count().reset_index(name="count")
    summ.to_csv(out_dir / "ALL__kickdown_summary.csv", index=False)


def detect_intent(prepped: Dict[str, pd.DataFrame], kickdown_ids: pd.DataFrame) -> pd.DataFrame:
    kd_keys = set(zip(kickdown_ids["file_name"], kickdown_ids["time_pedal_start_s"])) if not kickdown_ids.empty else set()
    rows = []
    for fname, df in prepped.items():
        time = df["time_s"].to_numpy()
        pedal = df["pedal_pct"].astype(float).to_numpy()
        rate = df["pedal_rate_pct_per_s"].astype(float).to_numpy() if "pedal_rate_pct_per_s" in df else np.diff(pedal, prepend=pedal[0]) / np.diff(time, prepend=1)
        gear = df["gear_actual__canon"].astype(int).to_numpy()
        speed = df["speed_mph"].astype(float).to_numpy()
        n = len(df)
        for i in range(n):
            if rate[i] >= 5 and rate[i] < 20 and pedal[i] >= 5:
                # avoid kickdown already recorded
                key = (fname, time[i])
                # window 5s
                j = i
                while j < n and time[j] - time[i] <= 5.0:
                    j += 1
                pedal_gain = pedal[j - 1] - pedal[i]
                if pedal_gain < 5 or pedal_gain > 20:
                    continue
                gear_min = gear[i:j].min()
                unlock = (df["tcc_state"].astype(str).iloc[i:j] != "LOCKED").any()
                speed_gain = speed[j - 1] - speed[i]
                label = "immediate" if speed_gain > 3 else "lazy"
                rows.append(
                    {
                        "file_name": fname,
                        "event_time_s": time[i],
                        "gear_start": gear[i],
                        "gear_min": gear_min,
                        "speed_start_mph": speed[i],
                        "speed_gain_5s": speed_gain,
                        "pedal_start_pct": pedal[i],
                        "pedal_end_pct": pedal[j - 1],
                        "tcc_unlock": unlock,
                        "response_label": label,
                    }
                )
    return pd.DataFrame(rows)


def summarize_intent(dfi: pd.DataFrame, out_dir: Path):
    dfi.to_csv(out_dir / "ALL__intent_episodes.csv", index=False)
    if dfi.empty:
        pd.DataFrame().to_csv(out_dir / "ALL__intent_summary.csv", index=False)
        return
    summ = (
        dfi.groupby(["gear_start", "response_label"])["event_time_s"]
        .count()
        .reset_index(name="count")
    )
    summ.to_csv(out_dir / "ALL__intent_summary.csv", index=False)


def detect_mid_pedal_6th_passes(
    prepped: Dict[str, pd.DataFrame],
    mode_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Detect 6th-gear, 70–85 mph mid-pedal “I want a bit more” episodes.
    """
    rows: List[Dict] = []
    for fname, df in prepped.items():
        work = df.copy()
        if mode_filter:
            if "trans_mode" in work.columns:
                mask_mode = work["trans_mode"].astype(str).str.lower() == str(mode_filter).lower()
                work = work[mask_mode]
                if work.empty:
                    print(f"[WARN] {fname}: no rows after mode filter {mode_filter!r} for mid-pedal pass detection.")
                    continue
            else:
                print(f"[WARN] {fname}: mode filter requested but trans_mode missing; using all rows.")
        required = [
            "time_s",
            "speed_mph",
            "gear_actual__canon",
            "pedal_pct",
            "throttle_pct",
            "tcc_state",
        ]
        missing = [c for c in required if c not in work.columns]
        if missing:
            raise ValueError(f"{fname}: missing required columns for mid-pedal pass detection: {missing}")

        time = work["time_s"].to_numpy()
        speed = work["speed_mph"].astype(float).to_numpy()
        gear = work["gear_actual__canon"].astype(int).to_numpy()
        pedal = work["pedal_pct"].astype(float).to_numpy()
        throttle = work["throttle_pct"].astype(float).to_numpy()
        tcc_state = work["tcc_state"].astype(str).str.upper().to_numpy()
        slip_col = (
            "tcc_slip_rpm"
            if "tcc_slip_rpm" in work.columns
            else "tcc_slip_rpm_fused"
            if "tcc_slip_rpm_fused" in work.columns
            else None
        )
        if slip_col is None:
            raise ValueError(f"{fname}: missing slip column (tcc_slip_rpm[_fused]) for mid-pedal pass detection.")
        tcc_slip = work[slip_col].astype(float).to_numpy()

        n = len(work)
        i = 0
        while i < n:
            if gear[i] != 6 or not (70 <= speed[i] <= 85) or not (10 <= pedal[i] <= 20):
                i += 1
                continue

            t0 = time[i]
            # Peak pedal within 1s
            j = i
            peak_val = pedal[i]
            peak_idx = i
            while j < n and time[j] - t0 <= 1.0:
                if pedal[j] > peak_val:
                    peak_val = pedal[j]
                    peak_idx = j
                j += 1
            if not (20 <= peak_val <= 40) or peak_val > 70:
                i += 1
                continue

            # Determine end index (10s window or downshift dwell >=0.2s)
            end_idx = None
            max_time = t0 + 10.0
            k = i + 1
            while k < n and time[k] <= max_time:
                if gear[k] != 6:
                    tgt = gear[k]
                    dwell_start = time[k]
                    m = k
                    while m + 1 < n and gear[m + 1] == tgt and time[m + 1] - dwell_start < 0.2:
                        m += 1
                    if time[m] - dwell_start >= 0.2:
                        end_idx = m
                        break
                k += 1
            if end_idx is None:
                # use last index within 10s
                end_idx = max(i, k - 1)
                while end_idx + 1 < n and time[end_idx + 1] <= max_time:
                    end_idx += 1

            t1 = time[end_idx]
            duration = t1 - t0
            if duration < 1.0:
                i += 1
                continue

            # helper to get speed at horizon within window
            def speed_at(target: float) -> Optional[float]:
                if target > t1:
                    return None
                idx = np.searchsorted(time, target, side="left")
                if idx <= i:
                    return speed[i]
                if idx >= n:
                    return speed[end_idx]
                # clamp to window
                if time[idx] > t1:
                    return None
                t0l, t1l = time[idx - 1], time[idx]
                v0, v1 = speed[idx - 1], speed[idx]
                if t1l <= t0l:
                    return v1
                alpha = (target - t0l) / (t1l - t0l)
                return v0 + alpha * (v1 - v0)

            speed_2 = speed_at(t0 + 2.0)
            speed_5 = speed_at(t0 + 5.0)
            speed_10 = speed_at(t0 + 10.0)

            def slip_locked(val):
                if np.isnan(val):
                    return False
                return abs(val) <= 50.0

            slip_window = tcc_slip[i : end_idx + 1]
            state_window = tcc_state[i : end_idx + 1]
            ever_unlock = False
            for sv, st in zip(slip_window, state_window):
                if st != "LOCKED" or (not slip_locked(sv) and not np.isnan(sv)):
                    ever_unlock = True
                    break
            ever_lock = any(st == "LOCKED" or slip_locked(sv) for sv, st in zip(slip_window, state_window))
            g_min = int(np.min(gear[i : end_idx + 1]))
            g_max = int(np.max(gear[i : end_idx + 1]))
            downshift = g_min < 6

            if not ever_unlock and not downshift:
                gear_path = "locked_only"
            elif ever_unlock and not downshift:
                gear_path = "unlock_no_downshift"
            elif ever_unlock and downshift:
                gear_path = "unlock_then_downshift"
            elif downshift and not ever_unlock:
                gear_path = "downshift_no_unlock"
            else:
                gear_path = "mixed_other"

            rows.append(
                {
                    "file_name": fname,
                    "mode": work["trans_mode"].iloc[i] if "trans_mode" in work.columns else "",
                    "time_start_s": t0,
                    "time_end_s": t1,
                    "duration_s": duration,
                    "speed_start_mph": speed[i],
                    "speed_end_mph": speed[end_idx],
                    "speed_gain_2s": speed_2 - speed[i] if speed_2 is not None else np.nan,
                    "speed_gain_5s": speed_5 - speed[i] if speed_5 is not None else np.nan,
                    "speed_gain_10s": speed_10 - speed[i] if speed_10 is not None else np.nan,
                    "pedal_start_pct": pedal[i],
                    "pedal_peak_pct": peak_val,
                    "gear_start": 6,
                    "gear_min": g_min,
                    "gear_max": g_max,
                    "gear_path": gear_path,
                    "tcc_state_start": tcc_state[i],
                    "tcc_state_end": tcc_state[end_idx],
                    "tcc_ever_unlocked": ever_unlock,
                    "tcc_ever_locked": ever_lock,
                    "throttle_start_pct": throttle[i],
                }
            )
            # advance past this window to avoid duplicates
            i = end_idx + 1
        # end while
    return pd.DataFrame(rows)


def summarize_mid_pedal_passes(dfp: pd.DataFrame, out_dir: Path):
    dfp.to_csv(out_dir / "ALL__6th_mid_pedal_passes.csv", index=False)
    if dfp.empty:
        pd.DataFrame().to_csv(out_dir / "ALL__6th_mid_pedal_passes_summary.csv", index=False)
        return
    dfp = dfp.copy()
    # pedal bands
    pedal_bins = [10, 15, 20, 25, 30, 40, 120]
    pedal_labels = ["10-15", "15-20", "20-25", "25-30", "30-40", "40+"]
    dfp["pedal_band"] = pd.cut(dfp["pedal_start_pct"], bins=pedal_bins, labels=pedal_labels, right=False)
    speed_bins = [70, 75, 80, 85, 200]
    speed_labels = ["70-75", "75-80", "80-85", "85+"]
    dfp["speed_band"] = pd.cut(dfp["speed_start_mph"], bins=speed_bins, labels=speed_labels, right=False)

    summ = (
        dfp.groupby(["gear_path", "pedal_band", "speed_band"])
        .agg(
            n_episodes=("file_name", "count"),
            speed_gain_2s_med=("speed_gain_2s", "median"),
            speed_gain_2s_p25=("speed_gain_2s", lambda x: np.nanpercentile(x, 25)),
            speed_gain_2s_p75=("speed_gain_2s", lambda x: np.nanpercentile(x, 75)),
            speed_gain_5s_med=("speed_gain_5s", "median"),
            speed_gain_5s_p25=("speed_gain_5s", lambda x: np.nanpercentile(x, 25)),
            speed_gain_5s_p75=("speed_gain_5s", lambda x: np.nanpercentile(x, 75)),
            speed_gain_10s_med=("speed_gain_10s", "median"),
            speed_gain_10s_p25=("speed_gain_10s", lambda x: np.nanpercentile(x, 25)),
            speed_gain_10s_p75=("speed_gain_10s", lambda x: np.nanpercentile(x, 75)),
            frac_tcc_unlocked=("tcc_ever_unlocked", "mean"),
            frac_downshift=("gear_min", lambda g: np.mean(g < 6)),
        )
        .reset_index()
    )
    summ.to_csv(out_dir / "ALL__6th_mid_pedal_passes_summary.csv", index=False)

def tcc_lock_unlock_events(prepped: Dict[str, pd.DataFrame], out_dir: Path):
    lock_rows = []
    unlock_rows = []
    for fname, df in prepped.items():
        state = df["tcc_state"].astype(str).to_numpy()
        time = df["time_s"].to_numpy()
        speed = df["speed_mph"].to_numpy()
        pedal = df["pedal_pct"].to_numpy()
        gear = df["gear_actual__canon"].astype(int).to_numpy()
        for i in range(1, len(df)):
            if state[i] != state[i - 1]:
                row = {
                    "file_name": fname,
                    "time_s": time[i],
                    "gear": gear[i],
                    "speed_mph": speed[i],
                    "pedal_pct": pedal[i],
                    "prev_state": state[i - 1],
                    "new_state": state[i],
                }
                if state[i] == "LOCKED":
                    lock_rows.append(row)
                if state[i - 1] == "LOCKED":
                    unlock_rows.append(row)
    pd.DataFrame(lock_rows).to_csv(out_dir / "ALL__tcc_lock_events.csv", index=False)
    pd.DataFrame(unlock_rows).to_csv(out_dir / "ALL__tcc_unlock_events.csv", index=False)


def tcc_dragging_segments(prepped: Dict[str, pd.DataFrame], out_dir: Path):
    rows = []
    for fname, df in prepped.items():
        state = df["tcc_state"].astype(str).to_numpy()
        slip = np.abs(df["tcc_slip_rpm_fused"].astype(float).to_numpy())
        accel = df["accel_mps2"].astype(float).to_numpy()
        time = df["time_s"].to_numpy()
        gear = df["gear_actual__canon"].astype(int).to_numpy()
        speed = df["speed_mph"].astype(float).to_numpy()
        mask = ((state == "LOCKED") | (state == "SLIP")) & (slip >= 50) & (slip <= 200) & (np.abs(accel) < 0.1)
        i = 0
        while i < len(df):
            if not mask[i]:
                i += 1
                continue
            start = i
            while i < len(df) and mask[i]:
                i += 1
            end = i - 1
            dur = time[end] - time[start]
            if dur >= 1.0:
                idx = slice(start, end + 1)
                rows.append(
                    {
                        "file_name": fname,
                        "start_time_s": float(time[start]),
                        "end_time_s": float(time[end]),
                        "duration_s": float(dur),
                        "gear_mode": int(Counter(gear[idx]).most_common(1)[0][0]),
                        "speed_min": float(speed[idx].min()),
                        "speed_max": float(speed[idx].max()),
                        "slip_mean": float(slip[idx].mean()),
                        "slip_max": float(slip[idx].max()),
                    }
                )
    pd.DataFrame(rows).to_csv(out_dir / "ALL__tcc_dragging_segments.csv", index=False)


def fuel_mpg(prepped: Dict[str, pd.DataFrame], out_dir: Path):
    segments = []
    strat_rows = []
    for fname, df in prepped.items():
        dt = df["dt_s"].astype(float).to_numpy()
        speed = df["speed_mph"].astype(float).to_numpy()
        # attempt fuel flow
        fuel_flow = None
        if "Instantaneous Fuel Flow Estimate" in df.columns:
            fuel_flow = pd.to_numeric(df["Instantaneous Fuel Flow Estimate"], errors="coerce").to_numpy()
        elif "Inst Fuel Used" in df.columns:
            # derive rough rate
            used = pd.to_numeric(df["Inst Fuel Used"], errors="coerce").fillna(method="ffill").to_numpy()
            fuel_flow = np.diff(used, prepend=used[0]) / np.where(dt > 0, dt, 1)
        if fuel_flow is None:
            continue
        fuel_gal = (fuel_flow * dt).sum() / 3600.0  # assuming L/h ~ same scale; relative
        dist_mi = (speed * dt).sum() / 3600.0
        mpg = dist_mi / fuel_gal if fuel_gal > 0 else np.nan
        segments.append({"file_name": fname, "distance_mi": dist_mi, "fuel_units": fuel_gal, "mpg_est": mpg})
        # mpg vs strategy simple buckets
        mask = (speed >= 60) & (speed <= 80)
        if mask.any():
            strat_rows.append(
                {
                    "file_name": fname,
                    "speed_60_80_time_s": float(dt[mask].sum()),
                    "mpg_est": (speed[mask] * dt[mask]).sum() / 3600.0 / ((fuel_flow[mask] * dt[mask]).sum() / 3600.0) if (fuel_flow[mask] * dt[mask]).sum() > 0 else np.nan,
                }
            )
    pd.DataFrame(segments).to_csv(out_dir / "ALL__fuel_usage_segments.csv", index=False)
    pd.DataFrame(strat_rows).to_csv(out_dir / "ALL__fuel_mpg_vs_strategy_summary.csv", index=False)


def slip_distribution(prepped: Dict[str, pd.DataFrame], budgets: pd.DataFrame, out_dir: Path):
    rows = []
    for fname, df in prepped.items():
        slip = df["tcc_slip_rpm_fused"].astype(float)
        gear = df["gear_actual__canon"].astype(int)
        slip_abs = slip.abs()
        for g in sorted(gear.unique()):
            if g <= 0:
                continue
            mask = gear == g
            data = slip_abs[mask]
            if data.empty:
                continue
            total = len(data)
            frac_le_50 = float((data <= 50).sum()) / total
            frac_50_120 = float(((data > 50) & (data <= 120)).sum()) / total
            frac_gt_120 = float((data > 120).sum()) / total
            rows.append(
                {
                    "file_name": fname,
                    "gear": int(g),
                    "slip_abs_p50": float(np.percentile(data, 50)),
                    "slip_abs_p75": float(np.percentile(data, 75)),
                    "slip_abs_p95": float(np.percentile(data, 95)),
                    "frac_le_50": frac_le_50,
                    "frac_50_120": frac_50_120,
                    "frac_gt_120": frac_gt_120,
                }
            )
    dist_df = pd.DataFrame(rows)
    dist_df.to_csv(out_dir / "ALL__tcc_slip_distribution.csv", index=False)
    if not dist_df.empty:
        for _, row in dist_df.iterrows():
            match = budgets[
                (budgets["file_name"] == row["file_name"]) & (budgets["gear"] == row["gear"])
            ]
            if not match.empty:
                pct_locked = float(match["pct_locked"].mean())
                diff = abs(pct_locked - row["frac_le_50"] * 100)
                if diff > 20:
                    print(
                        f"[WARN] TCC slip vs budget mismatch for {row['file_name']} gear {row['gear']}: "
                        f"pct_locked {pct_locked:.1f} vs frac_le_50 {row['frac_le_50']:.2f}"
                    )


def build_tuning_hints(out_dir: Path, shift_points: pd.DataFrame, shift_quality_summ: pd.DataFrame, tcc_lock: pd.DataFrame, tcc_unlock: pd.DataFrame, dragging: pd.DataFrame) -> None:
    hints = []
    # shift hints from shift_points counts
    if not shift_points.empty:
        shift_points["speed_bin"] = (shift_points["speed_mph"] // 5 * 5).astype(int)
        shift_points["pedal_bin"] = pd.cut(shift_points["pedal_pct"], bins=[0, 20, 50, 80, 120], labels=["0-20", "20-50", "50-80", "80-100"])
        grp = shift_points.groupby(["from_gear", "to_gear", "speed_bin", "pedal_bin"])
        for (fg, tg, sb, pb), g in grp:
            cnt = len(g)
            conf = "high" if cnt >= 20 else "medium" if cnt >= 5 else "low"
            delta = -1.0 if tg < fg else 0.5
            hints.append(
                {
                    "gear_pair": f"{int(fg)}-{int(tg)}",
                    "type": "shift_down" if tg < fg else "shift_up",
                    "speed_bin_mph": [int(sb), int(sb) + 5],
                    "pedal_bin_pct": pb,
                    "delta_mph": delta,
                    "reason": "event_frequency_based",
                    "coverage_events_count": cnt,
                    "coverage_time_s": None,
                    "coverage_logs_count": len(g["file_name"].unique()),
                    "confidence_level": conf,
                }
            )
    # tcc apply/release from lock/unlock
    def add_tcc_hint(df, typ, sign):
        if df.empty:
            return
        df["speed_bin"] = (df["speed_mph"] // 5 * 5).astype(int)
        df["pedal_bin"] = pd.cut(df["pedal_pct"], bins=[0, 20, 50, 80, 120], labels=["0-20", "20-50", "50-80", "80-100"])
        grp = df.groupby(["gear", "speed_bin", "pedal_bin"])
        for (g, sb, pb), gg in grp:
            cnt = len(gg)
            conf = "high" if cnt >= 20 else "medium" if cnt >= 5 else "low"
            hints.append(
                {
                    "gear_pair": f"{int(g)}",
                    "type": typ,
                    "speed_bin_mph": [int(sb), int(sb) + 5],
                    "pedal_bin_pct": pb,
                    "delta_mph": sign * 1.0,
                    "reason": f"{typ}_frequency",
                    "coverage_events_count": cnt,
                    "coverage_time_s": None,
                    "coverage_logs_count": len(gg["file_name"].unique()),
                    "confidence_level": conf,
                }
            )
    add_tcc_hint(tcc_lock, "tcc_apply", -1.0)
    add_tcc_hint(tcc_unlock, "tcc_release", 1.0)
    # dragging: suggest later apply (release) if many dragging
    if not dragging.empty:
        dragging["speed_bin"] = (dragging["speed_min"] // 5 * 5).astype(int)
        dragging["pedal_bin_pct"] = "20-50"
        grp = dragging.groupby(["gear_mode", "speed_bin"])
        for (g, sb), gg in grp:
            cnt = len(gg)
            hints.append(
                {
                    "gear_pair": f"{int(g)}",
                    "type": "tcc_release",
                    "speed_bin_mph": [int(sb), int(sb) + 5],
                    "pedal_bin_pct": "20-50",
                    "delta_mph": 1.0,
                    "reason": "dragging_segments",
                    "coverage_events_count": cnt,
                    "coverage_time_s": float(gg["duration_s"].sum()),
                    "coverage_logs_count": len(gg["file_name"].unique()),
                    "confidence_level": "medium" if cnt >= 3 else "low",
                }
            )
    out = {"schema": "v2", "notes": "Data-driven hints with coverage/confidence.", "shift_tcc_hints": hints}
    (out_dir / "TUNING_HINTS__shift_tcc.json").write_text(json.dumps(out, indent=2))
    print(f"[INFO] Tuning hints written with {len(hints)} entries.")



# -------------------------------------------------------------
# Master report and tuning hints (simplified scaffolding)
# -------------------------------------------------------------


def write_master_report(out_dir: Path, summary: Dict[str, any]) -> None:
    rpt = out_dir / "REPORT__highway_trans_MAX.md"
    lines = ["# Highway Trans MAX Analysis", ""]
    lines.append(f"Version: {VERSION}")
    lines.append(f"Logs analyzed: {', '.join(summary.get('logs', []))}")
    lines.append("")
    if "gear_usage" in summary:
        lines.append("## Gear usage (high-level)")
        for item in summary["gear_usage"]:
            lines.append(f"- {item}")
        lines.append("")
    if "tcc_time" in summary:
        lines.append("## TCC state time (gears 3–6)")
        for item in summary["tcc_time"]:
            lines.append(f"- {item}")
        lines.append("")
    if "abs_tcs" in summary:
        lines.append("## Stability events (ABS/TCS)")
        for item in summary["abs_tcs"]:
            lines.append(f"- {item}")
        lines.append("")
    # tuning hints
    hints_path = out_dir / "TUNING_HINTS__shift_tcc.json"
    if hints_path.exists():
        hints = json.loads(hints_path.read_text())
        lines.append("## Tuning hints")
        lines.append(f"- shift_tcc_hints entries: {len(hints.get('shift_tcc_hints', []))}")
        lines.append("")

    lines.append("\n(Additional detailed CSVs are in this folder.)\n")
    rpt.write_text("\n".join(lines))
    print(f"[OK] Master report written: {rpt}")


def write_tuning_hints(out_dir: Path) -> None:
    # deprecated stub; real hints built later
    pass


# -------------------------------------------------------------
# Main driver
# -------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MAX highway transmission + engine-side analysis")
    parser.add_argument("logs", nargs="+", help="Raw HP Tuners CSV logs")
    parser.add_argument("--out-dir", default="highway_MAX_analysis", help="Output directory")
    parser.add_argument("--mode-filter", default=None, help="Optional trans mode filter (e.g. 'pattern a')")
    args = parser.parse_args()

    print(f"[BANNER] {VERSION}")
    print(f"[BANNER] Script path: {Path(__file__).resolve()}")
    print(f"[BANNER] Output dir: {Path(args.out_dir).resolve()}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prepped_dir = out_dir / "prepped"
    combined_events_abs_tcs = []
    combined_gear_usage = []
    combined_tcc_time = []
    summary = {"logs": [], "gear_usage": [], "tcc_time": [], "abs_tcs": []}

    prepped_paths = []
    prepped_dfs: Dict[str, pd.DataFrame] = {}
    for log in args.logs:
        path = Path(log)
        if not path.exists():
            print(f"[WARN] Log {path} not found; skipping.")
            continue
        summary["logs"].append(path.name)
        try:
            prepped_path = prep_log(path, prepped_dir, args.mode_filter)
            prepped_paths.append(prepped_path)
            prepped_dfs[prepped_path.stem] = pd.read_csv(prepped_path, low_memory=False)
        except Exception as e:
            print(f"[ERROR] Failed to prep {path}: {e}")
            continue

    # Run analyses per prepped log
    for prep_path in prepped_paths:
        df = pd.read_csv(prep_path, low_memory=False)
        file_tag = prep_path.stem

        usage = gear_usage(df)
        usage.insert(0, "file_name", file_tag)
        usage.to_csv(out_dir / f"{file_tag}__gear_usage.csv", index=False)
        combined_gear_usage.append(usage.assign(file=file_tag))
        summary["gear_usage"].append(f"{file_tag}: " + ", ".join([f"g{int(r.gear)} {r.time_pct:.1f}%" for _, r in usage.iterrows()]))

        shifts = detect_shift_events(df)
        shifts.insert(0, "file_name", file_tag)
        shifts.to_csv(out_dir / f"{file_tag}__shift_events.csv", index=False)
        # shift points
        pts = build_shift_points(shifts)
        if not pts.empty:
            pts.insert(0, "file_name", file_tag)
            pts.to_csv(out_dir / f"{file_tag}__shift_points.csv", index=False)

        tcc_budget = tcc_time_budget(df, file_tag)
        tcc_budget.to_csv(out_dir / f"{file_tag}__tcc_time_budget.csv", index=False)
        combined_tcc_time.append(tcc_budget.assign(file=file_tag))
        for g in [3, 4, 5, 6]:
            mask = (tcc_budget["gear"] == g)
            if mask.any():
                locked = float(tcc_budget.loc[mask, "time_locked_s"].sum())
                total = float(
                    tcc_budget.loc[mask, "time_open_s"].sum()
                    + tcc_budget.loc[mask, "time_slip_s"].sum()
                    + tcc_budget.loc[mask, "time_locked_s"].sum()
                )
                if total > 0:
                    summary["tcc_time"].append(f"{file_tag} g{g}: LOCKED {locked/total*100:.1f}%")

        mapping_prepped = resolve_columns(df)

        ev_abs_tcs = abs_tcs_events(df, mapping_prepped)
        if not ev_abs_tcs.empty:
            ev_abs_tcs.to_csv(out_dir / f"{file_tag}__abs_tcs_events.csv", index=False)
            combined_events_abs_tcs.append(ev_abs_tcs)
            summary["abs_tcs"].append(f"{file_tag}: {len(ev_abs_tcs)} stability events")

        # Pedal usage
        pedal_usage(df, False).to_csv(out_dir / f"{file_tag}__pedal_usage_overall.csv", index=False)
        pedal_usage(df, True).to_csv(out_dir / f"{file_tag}__pedal_usage_highway.csv", index=False)

        # DFCO
        dfco_segments(df, mapping_prepped).to_csv(
            out_dir / f"{file_tag}__dfco_segments.csv", index=False
        )

    # Combined outputs
    if combined_gear_usage:
        pd.concat(combined_gear_usage, ignore_index=True).to_csv(out_dir / "ALL__gear_usage.csv", index=False)
    combined_tcc_df = pd.DataFrame()
    if combined_tcc_time:
        combined_tcc_df = pd.concat(combined_tcc_time, ignore_index=True)
        combined_tcc_df.to_csv(out_dir / "ALL__tcc_time_budget.csv", index=False)
    if combined_events_abs_tcs:
        pd.concat(combined_events_abs_tcs, ignore_index=True).to_csv(out_dir / "ALL__abs_tcs_events.csv", index=False)

    # Aggregate shift events/points
    all_shift_events = []
    all_shift_points = []
    for prep_path in prepped_paths:
        tag = prep_path.stem
        se_path = out_dir / f"{tag}__shift_events.csv"
        if se_path.exists():
            all_shift_events.append(pd.read_csv(se_path))
        sp_path = out_dir / f"{tag}__shift_points.csv"
        if sp_path.exists():
            all_shift_points.append(pd.read_csv(sp_path))
    if all_shift_events:
        df_all_shifts = pd.concat(all_shift_events, ignore_index=True)
        df_all_shifts.to_csv(out_dir / "ALL__shift_events.csv", index=False)
    else:
        df_all_shifts = pd.DataFrame()
        df_all_shifts.to_csv(out_dir / "ALL__shift_events.csv", index=False)
    if all_shift_points:
        df_all_pts = pd.concat(all_shift_points, ignore_index=True)
        df_all_pts.to_csv(out_dir / "ALL__shift_points.csv", index=False)
    else:
        df_all_pts = pd.DataFrame()
        df_all_pts.to_csv(out_dir / "ALL__shift_points.csv", index=False)

    # Slip distribution (needs combined budgets)
    slip_distribution(prepped_dfs, combined_tcc_df, out_dir)

    # Advanced modules
    print("[INFO] Running SHIFT_QUALITY...")
    shift_quality(df_all_shifts, prepped_dfs, out_dir)
    print("[INFO] Running SHIFT_LATENCY...")
    shift_latency(df_all_shifts, prepped_dfs, out_dir)
    print("[INFO] Running KICKDOWN...")
    df_kd = detect_kickdowns(prepped_dfs)
    summarize_kickdowns(df_kd, out_dir)
    print("[INFO] Running INTENT...")
    df_intent = detect_intent(prepped_dfs, df_kd)
    summarize_intent(df_intent, out_dir)
    print("[INFO] Running TCC lock/unlock...")
    tcc_lock_unlock_events(prepped_dfs, out_dir)
    print("[INFO] Running TCC dragging...")
    tcc_dragging_segments(prepped_dfs, out_dir)
    print("[INFO] Running FUEL/MPG...")
    fuel_mpg(prepped_dfs, out_dir)
    print("[INFO] Running 6th-gear mid-pedal passes...")
    df_mid = detect_mid_pedal_6th_passes(prepped_dfs, args.mode_filter)
    summarize_mid_pedal_passes(df_mid, out_dir)

    # Tuning hints (after advanced modules)
    build_tuning_hints(
        out_dir,
        df_all_pts if 'df_all_pts' in locals() else pd.DataFrame(),
        pd.read_csv(out_dir / "ALL__shift_quality_summary.csv") if (out_dir / "ALL__shift_quality_summary.csv").exists() else pd.DataFrame(),
        pd.read_csv(out_dir / "ALL__tcc_lock_events.csv") if (out_dir / "ALL__tcc_lock_events.csv").exists() else pd.DataFrame(),
        pd.read_csv(out_dir / "ALL__tcc_unlock_events.csv") if (out_dir / "ALL__tcc_unlock_events.csv").exists() else pd.DataFrame(),
        pd.read_csv(out_dir / "ALL__tcc_dragging_segments.csv") if (out_dir / "ALL__tcc_dragging_segments.csv").exists() else pd.DataFrame(),
    )

    # Master report
    write_master_report(out_dir, summary)

    # Run summary checklist
    def write_run_summary(base: Path):
        modules = []
        for name, files in EXPECTED_MODULE_OUTPUTS.items():
            found = []
            missing = []
            for fname in files:
                p = base / fname
                if p.exists():
                    found.append(fname)
                else:
                    missing.append(fname)
            if missing and found:
                status = "partial"
            elif missing and not found:
                status = "missing"
            else:
                status = "ok"
            modules.append(
                {
                    "name": name,
                    "expected_files": files,
                    "found_files": found,
                    "missing_files": missing,
                    "status": status,
                }
            )
        summary_json = {"modules": modules}
        out_path = base / "RUN_SUMMARY__highway_trans_MAX.json"
        out_path.write_text(json.dumps(summary_json, indent=2))
        # console summary
        missing_mods = [m["name"] for m in modules if m["status"] != "ok"]
        if missing_mods:
            print(f"[WARN] Modules with missing outputs: {missing_mods}")
        else:
            print("[OK] All expected module outputs present.")

    write_run_summary(out_dir)

    # Debug listing
    debug_list_out_dir(out_dir)

    # Small zip (exclude prepped and large files)
    version_slug = re.sub(r"[^A-Za-z0-9]+", "_", VERSION).strip("_")
    small_zip = out_dir / f"highway_MAX_outputs_small__{version_slug}.zip"
    with zipfile.ZipFile(small_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            if p.is_dir():
                continue
            rel = p.relative_to(out_dir)
            if rel.parts[0] == "prepped":
                continue
            if "zip_" in p.name:
                continue
            if p.stat().st_size > 50 * 1024 * 1024:
                continue
            name = str(rel)
            if (
                name.startswith("ALL__")
                or name.startswith("REPORT__")
                or name.startswith("RUN_SUMMARY__")
                or name.startswith("TUNING_HINTS__")
                or name.endswith("pedal_usage_overall.csv")
                or name.endswith("pedal_usage_highway.csv")
                or name.endswith("dfco_segments.csv")
            ):
                zf.write(p, rel)
    print(f"[OK] Small outputs zip written: {small_zip} ({small_zip.stat().st_size} bytes)")

    # Zip everything (may be large)
    zip_path = out_dir.with_name(f"{out_dir.name}__{version_slug}").with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            zf.write(p, p.relative_to(out_dir.parent))
    print(f"[OK] Output zipped to {zip_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
