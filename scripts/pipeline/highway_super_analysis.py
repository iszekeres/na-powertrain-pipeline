#!/usr/bin/env python3
"""
HIGHWAY SUPER ANALYSIS SCRIPT
=============================
Goal:
    A SINGLE Python script that can be dropped next to prepped logs and
    torque-surface / MAX-pack outputs to generate a detailed highway analysis
    for 4–5–6 + TCC.

Usage examples:
    python highway_super_analysis.py --all --schedule-name comfort_v2
    python highway_super_analysis.py --run-torque-deficit
    python highway_super_analysis.py --run-virtual-sim --schedule-name comfort_v2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FileConfig:
    """
    Expected input filenames / patterns (relative to script directory).
    """

    clean_log_pattern: str = "*CLEAN_FULL*with_brake_time*.csv"
    intent_episodes: str = "ALL__intent_episodes.csv"
    tcc_dragging_segments: str = "ALL__tcc_dragging_segments.csv"
    gear_usage: str = "ALL__gear_usage.csv"
    shift_events: str = "ALL__shift_events.csv"
    shift_quality_events: str = "ALL__shift_quality_events.csv"
    tcc_time_budget: str = "ALL__tcc_time_budget.csv"
    tcc_slip_distribution: str = "ALL__tcc_slip_distribution.csv"
    torque_surface_by_gear: str = "torque_surface__by_gear.csv"
    torque_air_spark_speedspace: str = "torque_air_spark_surface__SPEEDSPACE.csv"
    torque_gain_downshift_map: str = "torque_gain__downshift_map.csv"
    hybrid_winner_map: str = "hybrid_torque_winner__SPEEDSPACE.csv"
    schedule_shift_json_pattern: str = "schedule__{schedule_name}__updown_4_5_6.json"
    schedule_tcc_json_pattern: str = "schedule__{schedule_name}__tcc_5_6.json"


@dataclass
class ScheduleConfig:
    tps_axis: List[float]
    up_4_5: List[float]
    down_5_4: List[float]
    up_5_6: List[float]
    down_6_5: List[float]
    tcc_apply_5: List[float]
    tcc_release_5: List[float]
    tcc_apply_6: List[float]
    tcc_release_6: List[float]


def make_output_dir(base_dir: Path, label: str = "highway_super_analysis") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / f"{label}__{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[WARN] Missing expected CSV: {path}", file=sys.stderr)
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}", file=sys.stderr)
        return None


def load_clean_logs(base_dir: Path, pattern: str) -> pd.DataFrame:
    """
    Glob and load all CLEAN_FULL logs. If none found, fall back to prepped logs.
    Add a 'log_id' column with the source filename stem.
    """
    files = sorted(base_dir.glob(pattern))
    if not files:
        prepped_dir = base_dir / "newlogs" / "highway_MAX_analysis" / "prepped"
        if prepped_dir.exists():
            files = sorted(prepped_dir.glob("*__prepped.csv"))
        if not files:
            files = sorted(base_dir.rglob("*__prepped.csv"))
        if not files:
            raise FileNotFoundError(f"No clean/prepped logs found (pattern {pattern}).")
        print(f"[WARN] Falling back to prepped logs (found {len(files)} file(s)).")
    else:
        print(f"[INFO] Found {len(files)} clean log(s).")
    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df["log_id"] = f.stem
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def bin_series(values: pd.Series, bin_edges: List[float]) -> pd.Series:
    labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges) - 1)]
    return pd.cut(values, bins=bin_edges, labels=labels, right=False, include_lowest=True)


def _surface_cols(df_surface: pd.DataFrame) -> Tuple[str, str, str, str]:
    cols = df_surface.columns
    gear_col = "gear" if "gear" in cols else cols[0]
    speed_col = None
    for cand in ["speed_center_mph", "speed_center", "speed_mph", "speed"]:
        if cand in cols:
            speed_col = cand
            break
    pedal_col = None
    for cand in ["pedal_center_pct", "pedal_center", "pedal_pct", "tps_pct"]:
        if cand in cols:
            pedal_col = cand
            break
    torque_col = None
    for cand in ["axle_torque_mean", "physics_engine_torque_median", "eng_torque_mean", "physics_wheel_torque_median"]:
        if cand in cols:
            torque_col = cand
            break
    if speed_col is None or pedal_col is None or torque_col is None:
        raise KeyError(f"Surface missing required columns (speed/pedal/torque). Columns: {list(cols)}")
    return gear_col, speed_col, pedal_col, torque_col


def nearest_bin_lookup(
    df_surface: pd.DataFrame,
    gear: int,
    speed_mph: float,
    pedal_pct: float,
) -> Optional[float]:
    if df_surface is None or df_surface.empty:
        return None
    gear_col, speed_col, pedal_col, torque_col = _surface_cols(df_surface)
    sub = df_surface[df_surface[gear_col] == gear]
    if sub.empty:
        return None
    d_speed = sub[speed_col].astype(float) - float(speed_mph)
    d_pedal = sub[pedal_col].astype(float) - float(pedal_pct)
    dist2 = d_speed * d_speed + d_pedal * d_pedal
    idx_min = dist2.idxmin()
    val = sub.loc[idx_min, torque_col]
    if pd.isna(val):
        return None
    return float(val)


def lookup_torque_vectorized(
    df_surface: pd.DataFrame, gear_arr: np.ndarray, speed_arr: np.ndarray, pedal_arr: np.ndarray
) -> np.ndarray:
    """
    Fast approximate nearest lookup using per-gear vectorized rounding to nearest speed/pedal center.
    """
    gear_col, speed_col, pedal_col, torque_col = _surface_cols(df_surface)
    out = np.full(len(gear_arr), np.nan, dtype=float)
    for g in np.unique(gear_arr.astype(int)):
        mask = gear_arr == g
        if not mask.any():
            continue
        sub = df_surface[df_surface[gear_col] == g]
        if sub.empty:
            continue
        speed_centers = sub[speed_col].to_numpy(dtype=float)
        pedal_centers = sub[pedal_col].to_numpy(dtype=float)
        torques_map = sub.set_index([speed_col, pedal_col])[torque_col]

        speeds = speed_arr[mask].astype(float)
        pedals = pedal_arr[mask].astype(float)

        idx_s = np.abs(speeds[:, None] - speed_centers[None, :]).argmin(axis=1)
        idx_p = np.abs(pedals[:, None] - pedal_centers[None, :]).argmin(axis=1)
        nearest_speed = speed_centers[idx_s]
        nearest_pedal = pedal_centers[idx_p]
        keys = pd.MultiIndex.from_arrays([nearest_speed, nearest_pedal])
        out_vals = torques_map.reindex(keys, fill_value=np.nan).to_numpy()
        out[mask] = out_vals
    return out


def normalize_tcc_drag_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ALL__tcc_dragging_segments.csv into expected columns:
      gear, speed_mph_median, pedal_pct_median, slip_mean_rpm, duration_s, mode
    Uses flexible column detection and computes medians/durations if needed.
    Logs mapping summary and drops unusable rows.
    """
    original_rows = len(df)
    mapping_used: Dict[str, str] = {}

    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    # Gear
    gear_col = pick(
        ["gear", "gear_mode", "gear_actual", "Trans Current Gear", "Transmission Current Gear"]
    )
    if gear_col:
        mapping_used["gear"] = gear_col
        gear = pd.to_numeric(df[gear_col], errors="coerce").astype("Int64")
    else:
        gear = pd.Series([], dtype="Int64")

    # Speed median
    speed_col = pick(
        [
            "speed_mph_median",
            "speed_mid",
            "speed_mph_mid",
            "vehicle_speed_mid",
            "speed_mean",
            "speed_mph_mean",
        ]
    )
    if speed_col:
        mapping_used["speed_mph_median"] = speed_col
        speed = pd.to_numeric(df[speed_col], errors="coerce").astype(float)
    else:
        start_col = pick(["speed_start_mph", "speed_mph_start"])
        end_col = pick(["speed_end_mph", "speed_mph_end"])
        if start_col and end_col:
            mapping_used["speed_mph_median"] = f"{start_col}+{end_col}"
            speed = 0.5 * (
                pd.to_numeric(df[start_col], errors="coerce").astype(float)
                + pd.to_numeric(df[end_col], errors="coerce").astype(float)
            )
        else:
            speed = pd.Series([], dtype=float)

    # Pedal median
    pedal_col = pick(["pedal_pct_median", "pedal_mean_pct", "pedal_center_pct"])
    if pedal_col is None:
        pedal_col = pick(["pedal_pct", "Accelerator Pedal Position"])
    if pedal_col:
        mapping_used["pedal_pct_median"] = pedal_col
        pedal = pd.to_numeric(df[pedal_col], errors="coerce").astype(float)
    else:
        pedal = pd.Series([], dtype=float)

    # Slip mean rpm
    slip_col = pick(["slip_mean_rpm", "slip_mean", "mean_tcc_slip_raw", "TCC Slip", "tcc_slip"])
    if slip_col:
        mapping_used["slip_mean_rpm"] = slip_col
        slip = pd.to_numeric(df[slip_col], errors="coerce").astype(float)
    else:
        slip = pd.Series([], dtype=float)

    # Duration
    duration_col = pick(["duration_s", "segment_duration_s", "dur_s"])
    if duration_col:
        mapping_used["duration_s"] = duration_col
        duration = pd.to_numeric(df[duration_col], errors="coerce").astype(float)
    else:
        start_t = pick(["t_start_s", "start_time_s", "t_start"])
        end_t = pick(["t_end_s", "end_time_s", "t_end"])
        if start_t and end_t:
            mapping_used["duration_s"] = f"{start_t}+{end_t}"
            duration = (
                pd.to_numeric(df[end_t], errors="coerce").astype(float)
                - pd.to_numeric(df[start_t], errors="coerce").astype(float)
            )
        else:
            duration = pd.Series([], dtype=float)

    duration = duration.clip(lower=0)

    # Mode
    mode_col = pick(["mode", "tcc_state_modes", "tcc_state"])
    if mode_col:
        mapping_used["mode"] = mode_col
        mode = df[mode_col].astype(str)
    else:
        mode = pd.Series(["unknown"] * len(df))

    norm = pd.DataFrame(
        {
            "gear": gear,
            "speed_mph_median": speed,
            "pedal_pct_median": pedal,
            "slip_mean_rpm": slip,
            "duration_s": duration,
            "mode": mode,
        }
    )

    norm = norm[
        (norm["gear"].between(1, 6))
        & norm["speed_mph_median"].notna()
        & norm["pedal_pct_median"].notna()
        & norm["slip_mean_rpm"].notna()
        & norm["duration_s"].notna()
        & (norm["duration_s"] > 0)
    ].copy()

    norm["gear"] = norm["gear"].astype(int)
    norm["speed_mph_median"] = norm["speed_mph_median"].astype(float)
    norm["pedal_pct_median"] = norm["pedal_pct_median"].astype(float)
    norm["slip_mean_rpm"] = norm["slip_mean_rpm"].astype(float)
    norm["duration_s"] = norm["duration_s"].astype(float)
    norm["mode"] = norm["mode"].astype(str)

    print(
        f"[INFO] Normalized TCC dragging segments: {original_rows} -> {len(norm)} rows; "
        f"mapped columns: {mapping_used}"
    )
    return norm


def load_schedule_config(base_dir: Path, files: FileConfig, schedule_name: str) -> Optional[ScheduleConfig]:
    shift_path = base_dir / files.schedule_shift_json_pattern.format(schedule_name=schedule_name)
    tcc_path = base_dir / files.schedule_tcc_json_pattern.format(schedule_name=schedule_name)

    if not shift_path.exists() or not tcc_path.exists():
        print(f"[WARN] Missing schedule JSON(s) for '{schedule_name}'. "
              f"Expected: {shift_path.name} and {tcc_path.name}", file=sys.stderr)
        return None

    # Use utf-8-sig to tolerate BOM from Windows editors
    with open(shift_path, "r", encoding="utf-8-sig") as f:
        shift_cfg = json.load(f)
    with open(tcc_path, "r", encoding="utf-8-sig") as f:
        tcc_cfg = json.load(f)

    tps_axis = shift_cfg["tps_axis"]
    if tps_axis != tcc_cfg.get("tps_axis", tps_axis):
        print("[WARN] TPS axes differ between shift and TCC configs; using shift axis", file=sys.stderr)

    sched = ScheduleConfig(
        tps_axis=tps_axis,
        up_4_5=shift_cfg["up_4_5"],
        down_5_4=shift_cfg["down_5_4"],
        up_5_6=shift_cfg["up_5_6"],
        down_6_5=shift_cfg["down_6_5"],
        tcc_apply_5=tcc_cfg["tcc_apply_5"],
        tcc_release_5=tcc_cfg["tcc_release_5"],
        tcc_apply_6=tcc_cfg["tcc_apply_6"],
        tcc_release_6=tcc_cfg["tcc_release_6"],
    )
    return sched


def tps_to_index(tps_axis: List[float], pedal_pct: float) -> int:
    for i in range(len(tps_axis) - 1):
        if tps_axis[i] <= pedal_pct < tps_axis[i + 1]:
            return i
    return len(tps_axis) - 1


def unify_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    cols = set(df.columns)
    for cand in ["time_s", "Time", "Time (s)", "Offset"]:
        if cand in cols:
            col_map[cand] = "time_s"
            break
    for cand in ["vehicle_speed_mph", "speed_mph", "Vehicle Speed (SAE)", "Vehicle Speed"]:
        if cand in cols:
            col_map[cand] = "vehicle_speed_mph"
            break
    for cand in ["gear_actual__canon", "gear_actual", "Trans Current Gear", "Transmission Current Gear"]:
        if cand in cols:
            col_map[cand] = "gear_actual__canon"
            break
    for cand in ["pedal_pct", "Accelerator Pedal Position", "Accelerator Pedal Position %", "APP %"]:
        if cand in cols:
            col_map[cand] = "pedal_pct"
            break
    for cand in ["mode", "trans_mode", "Trans Shift Mode"]:
        if cand in cols:
            col_map[cand] = "mode"
            break
    df = df.rename(columns=col_map)
    if "mode" not in df.columns:
        df["mode"] = "UNKNOWN"
    required = ["time_s", "vehicle_speed_mph", "gear_actual__canon", "pedal_pct", "mode"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Clean/prepped logs missing required columns after normalization: {missing}")
    return df


def run_virtual_schedule_sim(
    base_dir: Path,
    out_dir: Path,
    files: FileConfig,
    schedule: ScheduleConfig,
    df_surface_speedspace: pd.DataFrame,
    df_clean: pd.DataFrame,
):
    print("[INFO] Running virtual schedule simulation...")

    cols_needed = ["time_s", "vehicle_speed_mph", "gear_actual__canon", "pedal_pct", "mode", "log_id"]
    missing_cols = [c for c in cols_needed if c not in df_clean.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in clean logs: {missing_cols}")

    df = df_clean[cols_needed].copy()
    df.sort_values(["log_id", "time_s"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.dropna(subset=["gear_actual__canon"])

    df["gear_virtual"] = df["gear_actual__canon"].astype(int)
    df["tcc_virtual"] = np.where(df["gear_virtual"].isin([4, 5, 6]), "LOCKED", "OPEN")

    tps_axis = schedule.tps_axis
    n_rows = len(df)
    gear_virtual = df["gear_virtual"].to_numpy()
    tcc_virtual = df["tcc_virtual"].to_numpy()
    speed = df["vehicle_speed_mph"].to_numpy()
    pedal = df["pedal_pct"].to_numpy()

    for i in range(1, n_rows):
        if df.at[i, "log_id"] != df.at[i - 1, "log_id"]:
            gear_virtual[i] = df.at[i, "gear_actual__canon"]
            tcc_virtual[i] = "LOCKED" if gear_virtual[i] in (4, 5, 6) else "OPEN"
            continue

        gear_virtual[i] = gear_virtual[i - 1]
        tcc_virtual[i] = tcc_virtual[i - 1]

        g = int(gear_virtual[i])
        v = float(speed[i])
        p = float(pedal[i])
        k = tps_to_index(tps_axis, p)

        if g == 5:
            if tcc_virtual[i] == "LOCKED" and v < schedule.tcc_release_5[k]:
                tcc_virtual[i] = "OPEN"
            elif tcc_virtual[i] != "LOCKED" and v > schedule.tcc_apply_5[k]:
                tcc_virtual[i] = "LOCKED"
        elif g == 6:
            if tcc_virtual[i] == "LOCKED" and v < schedule.tcc_release_6[k]:
                tcc_virtual[i] = "OPEN"
            elif tcc_virtual[i] != "LOCKED" and v > schedule.tcc_apply_6[k]:
                tcc_virtual[i] = "LOCKED"
        else:
            tcc_virtual[i] = "OPEN"

        if g == 6:
            if v < schedule.down_6_5[k]:
                g = 5
        elif g == 5:
            if v > schedule.up_5_6[k]:
                g = 6
            if v < schedule.down_5_4[k]:
                g = 4
        elif g == 4:
            if v > schedule.up_4_5[k]:
                g = 5

        gear_virtual[i] = g

    df["gear_virtual"] = gear_virtual
    df["tcc_virtual"] = tcc_virtual

    df["T_axle_actual"] = lookup_torque_vectorized(
        df_surface_speedspace,
        df["gear_actual__canon"].to_numpy(),
        df["vehicle_speed_mph"].to_numpy(),
        df["pedal_pct"].to_numpy(),
    )
    df["T_axle_virtual"] = lookup_torque_vectorized(
        df_surface_speedspace,
        df["gear_virtual"].to_numpy(),
        df["vehicle_speed_mph"].to_numpy(),
        df["pedal_pct"].to_numpy(),
    )
    df["delta_T_axle"] = df["T_axle_virtual"] - df["T_axle_actual"]

    out_timeseries = out_dir / "virtual_schedule__timeseries.csv"
    df.to_csv(out_timeseries, index=False)

    speed_bins = list(range(45, 95, 5))
    pedal_bins = [0, 5, 10, 20, 30, 40, 60, 100]

    df["speed_bin_mph"] = bin_series(df["vehicle_speed_mph"], speed_bins)
    df["pedal_bin_pct"] = bin_series(df["pedal_pct"], pedal_bins)

    grp = df.groupby(["mode", "speed_bin_mph", "pedal_bin_pct"], dropna=False, observed=False)
    summary = grp.agg(
        time_s=("time_s", lambda s: s.iloc[-1] - s.iloc[0] if len(s) > 1 else 0.0),
        mean_T_axle_actual=("T_axle_actual", "mean"),
        mean_T_axle_virtual=("T_axle_virtual", "mean"),
        mean_delta_T_axle=("delta_T_axle", "mean"),
        p95_delta_T_axle=("delta_T_axle", lambda x: np.nanpercentile(x.dropna(), 95) if len(x.dropna()) else np.nan),
        dominant_gear_virtual=("gear_virtual", lambda x: x.value_counts().index[0] if len(x) else np.nan),
    ).reset_index()

    total_time = summary["time_s"].sum()
    summary["time_pct"] = summary["time_s"] / total_time if total_time > 0 else 0.0

    out_bin = out_dir / "virtual_schedule__bin_summary.csv"
    summary.to_csv(out_bin, index=False)

    intent_path = base_dir / files.intent_episodes
    if intent_path.exists():
        df_intent = pd.read_csv(intent_path)
        rows = []
        for _, ep in df_intent.iterrows():
            log_id = ep.get("log_id", ep.get("file_tag", None))
            if log_id is None:
                continue
            t_start = ep.get("t_start_s", ep.get("start_time_s"))
            t_end = ep.get("t_end_s", ep.get("end_time_s"))
            if pd.isna(t_start) or pd.isna(t_end):
                continue
            sub = df[(df["log_id"] == log_id) & (df["time_s"] >= t_start) & (df["time_s"] <= t_end)]
            if sub.empty:
                continue
            start_speed = sub["vehicle_speed_mph"].iloc[0]
            target5 = start_speed + 5.0
            target10 = start_speed + 10.0
            t_base = sub["time_s"].iloc[0]
            t_reach5 = sub[sub["vehicle_speed_mph"] >= target5]
            t_reach10 = sub[sub["vehicle_speed_mph"] >= target10]
            time_to_dv5_actual = (t_reach5["time_s"].iloc[0] - t_base) if not t_reach5.empty else np.nan
            time_to_dv10_actual = (t_reach10["time_s"].iloc[0] - t_base) if not t_reach10.empty else np.nan
            mean_T_actual = sub["T_axle_actual"].mean()
            mean_T_virtual = sub["T_axle_virtual"].mean()
            delta_T_mean = mean_T_virtual - mean_T_actual
            if mean_T_virtual and mean_T_actual and mean_T_virtual > 0 and mean_T_actual > 0:
                factor = mean_T_actual / mean_T_virtual
                time_to_dv5_virtual = time_to_dv5_actual * factor if not pd.isna(time_to_dv5_actual) else np.nan
                time_to_dv10_virtual = time_to_dv10_actual * factor if not pd.isna(time_to_dv10_actual) else np.nan
            else:
                time_to_dv5_virtual = np.nan
                time_to_dv10_virtual = np.nan
            rows.append(
                {
                    "episode_id": ep.get("episode_id", len(rows)),
                    "log_id": log_id,
                    "mode": sub["mode"].iloc[0],
                    "response_label": ep.get("response_label", ""),
                    "intent_strength": ep.get("intent_strength", ""),
                    "t_start_s": t_start,
                    "t_end_s": t_end,
                    "time_to_dv5_actual": time_to_dv5_actual,
                    "time_to_dv10_actual": time_to_dv10_actual,
                    "time_to_dv5_virtual_est": time_to_dv5_virtual,
                    "time_to_dv10_virtual_est": time_to_dv10_virtual,
                    "mean_T_axle_actual": mean_T_actual,
                    "mean_T_axle_virtual": mean_T_virtual,
                    "delta_T_mean": delta_T_mean,
                    "gear_start_actual": int(sub["gear_actual__canon"].iloc[0]),
                    "gear_start_virtual": int(sub["gear_virtual"].iloc[0]),
                    "pct_time_geardown_virtual": (sub["gear_virtual"] < sub["gear_actual__canon"]).mean(),
                    "pct_time_tcc_locked_virtual": (sub["tcc_virtual"] == "LOCKED").mean(),
                }
            )
        df_intent_out = pd.DataFrame(rows)
        out_intent = out_dir / "virtual_schedule__intent_summary.csv"
        df_intent_out.to_csv(out_intent, index=False)


def run_torque_deficit_integral(
    base_dir: Path,
    out_dir: Path,
    df_clean: pd.DataFrame,
    df_surface_speedspace: pd.DataFrame,
):
    print("[INFO] Running torque deficit integral (pain map)...")

    cols_needed = ["time_s", "vehicle_speed_mph", "gear_actual__canon", "pedal_pct", "mode", "log_id"]
    missing_cols = [c for c in cols_needed if c not in df_clean.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in clean logs: {missing_cols}")

    df = df_clean[cols_needed].copy()
    df.sort_values(["log_id", "time_s"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    speed_bins = list(range(45, 95, 5))
    pedal_bins = [0, 5, 10, 20, 30, 40, 60, 100]
    df["speed_bin_mph"] = bin_series(df["vehicle_speed_mph"], speed_bins)
    df["pedal_bin_pct"] = bin_series(df["pedal_pct"], pedal_bins)

    df["dt"] = 0.0
    for _, sub in df.groupby("log_id"):
        idx = sub.index
        dt = sub["time_s"].diff().fillna(0.0).clip(lower=0.0, upper=1.0)
        df.loc[idx, "dt"] = dt

    gear_arr = df["gear_actual__canon"].to_numpy()
    speed_arr = df["vehicle_speed_mph"].to_numpy()
    pedal_arr = df["pedal_pct"].to_numpy()
    df["T_actual"] = lookup_torque_vectorized(df_surface_speedspace, gear_arr, speed_arr, pedal_arr)

    best_stack = []
    for g_test in (4, 5, 6):
        best_stack.append(lookup_torque_vectorized(df_surface_speedspace, np.full_like(gear_arr, g_test), speed_arr, pedal_arr))
    df["T_best"] = np.nanmax(np.vstack(best_stack), axis=0)
    df["delta_T"] = (df["T_best"] - df["T_actual"]).clip(lower=0.0)
    df["delta_T_dt"] = df["delta_T"] * df["dt"]

    grp_cols = ["mode", "speed_bin_mph", "pedal_bin_pct"]
    grp = df.groupby(grp_cols, dropna=False, observed=False)
    agg = grp.agg(
        time_s=("dt", "sum"),
        torque_deficit_integral=("delta_T_dt", "sum"),
        samples=("dt", "count"),
    ).reset_index()

    total_time = agg["time_s"].sum()
    agg["time_pct"] = agg["time_s"] / total_time if total_time > 0 else 0.0
    agg["mean_delta_T"] = agg["torque_deficit_integral"] / agg["time_s"].replace(0, np.nan)

    out_path = out_dir / "highway_torque_deficit_integral__by_bin.csv"
    agg.to_csv(out_path, index=False)


def run_intent_frustration_map(
    base_dir: Path,
    out_dir: Path,
    files: FileConfig,
    df_clean: pd.DataFrame,
    df_surface_speedspace: pd.DataFrame,
):
    intent_path = base_dir / files.intent_episodes
    df_intent = load_csv_if_exists(intent_path)
    if df_intent is None:
        print("[WARN] No intent episodes file; skipping intent frustration map.")
        return

    print("[INFO] Running intent × frustration analysis...")

    df_clean_sorted = df_clean.sort_values(["log_id", "time_s"]).reset_index(drop=True)

    episodes_rows = []

    for idx, ep in df_intent.iterrows():
        log_id = ep.get("log_id", ep.get("file_tag", None))
        if log_id is None:
            continue

        t_start = ep.get("t_start_s", ep.get("start_time_s"))
        t_end = ep.get("t_end_s", ep.get("end_time_s"))
        if pd.isna(t_start) or pd.isna(t_end):
            continue

        sub = df_clean_sorted[
            (df_clean_sorted["log_id"] == log_id)
            & (df_clean_sorted["time_s"] >= t_start)
            & (df_clean_sorted["time_s"] <= t_end)
        ].copy()

        if sub.empty:
            continue

        response_label = ep.get("response_label", "unknown")
        mode = sub["mode"].iloc[0]
        speed_start = sub["vehicle_speed_mph"].iloc[0]
        speed_end = sub["vehicle_speed_mph"].iloc[-1]
        pedal_start = sub["pedal_pct"].iloc[0]
        pedal_peak = sub["pedal_pct"].max()
        time_to_peak = sub.loc[sub["pedal_pct"].idxmax(), "time_s"] - t_start

        delta_pedal = pedal_peak - pedal_start
        pedal_rate = delta_pedal / time_to_peak if time_to_peak > 0 else 0.0

        if delta_pedal < 5 and pedal_rate < 5:
            intent_strength = "LIGHT"
        elif delta_pedal < 15 and pedal_rate < 15:
            intent_strength = "MED"
        else:
            intent_strength = "AGGRESSIVE"

        t_limit_3 = t_start + 3.0
        t_limit_5 = t_start + 5.0
        sub_3 = sub[sub["time_s"] <= t_limit_3]
        sub_5 = sub[sub["time_s"] <= t_limit_5]
        dv3 = (sub_3["vehicle_speed_mph"].iloc[-1] - speed_start) if len(sub_3) else 0.0
        dv5 = (sub_5["vehicle_speed_mph"].iloc[-1] - speed_start) if len(sub_5) else 0.0

        sub_peak = sub.loc[[sub["pedal_pct"].idxmax()]]
        v_peak = float(sub_peak["vehicle_speed_mph"].iloc[0])
        p_peak = float(sub_peak["pedal_pct"].iloc[0])
        g_start = int(sub["gear_actual__canon"].iloc[0])

        T_actual_peak = nearest_bin_lookup(df_surface_speedspace, gear=g_start, speed_mph=v_peak, pedal_pct=p_peak)
        T_candidates = []
        best_gear = g_start
        for g_test in (4, 5, 6):
            val = nearest_bin_lookup(df_surface_speedspace, gear=g_test, speed_mph=v_peak, pedal_pct=p_peak)
            if val is not None:
                T_candidates.append((g_test, val))
        if T_candidates:
            best_gear, T_best_peak = max(T_candidates, key=lambda x: x[1])
        else:
            T_best_peak = T_actual_peak

        if T_actual_peak is not None and T_actual_peak > 0 and T_best_peak is not None:
            torque_gap_pct = (T_best_peak - T_actual_peak) / T_actual_peak
        else:
            torque_gap_pct = np.nan

        frustrated = (
            str(response_label).lower() == "lazy"
            and intent_strength in ("MED", "AGGRESSIVE")
        )

        episodes_rows.append(
            {
                "episode_id": ep.get("episode_id", idx),
                "log_id": log_id,
                "mode": mode,
                "response_label": response_label,
                "intent_strength": intent_strength,
                "frustrated": frustrated,
                "t_start_s": t_start,
                "t_end_s": t_end,
                "speed_start_mph": speed_start,
                "speed_end_mph": speed_end,
                "pedal_start_pct": pedal_start,
                "pedal_peak_pct": pedal_peak,
                "delta_pedal": delta_pedal,
                "pedal_rate_pct_per_s": pedal_rate,
                "dv3_mph": dv3,
                "dv5_mph": dv5,
                "gear_start": g_start,
                "best_gear_at_peak": best_gear,
                "T_actual_peak": T_actual_peak,
                "T_best_peak": T_best_peak,
                "torque_gap_pct_at_peak": torque_gap_pct,
            }
        )

    df_ep = pd.DataFrame(episodes_rows)
    out_ep = out_dir / "highway_intent_frustration__episodes.csv"
    df_ep.to_csv(out_ep, index=False)

    if not df_ep.empty:
        df_ep["speed_band"] = pd.cut(
            df_ep["speed_start_mph"],
            bins=list(range(45, 95, 5)),
            right=False,
            include_lowest=True,
        )
        df_ep["pedal_band"] = pd.cut(
            df_ep["pedal_start_pct"],
            bins=[0, 5, 10, 20, 30, 40, 60, 100],
            right=False,
            include_lowest=True,
        )

        grp = df_ep.groupby(["mode", "intent_strength", "speed_band", "pedal_band"], dropna=False, observed=False)
        summary = grp.agg(
            n_episodes=("episode_id", "count"),
            n_frustrated=("frustrated", lambda x: int(x.sum())),
            frac_frustrated=("frustrated", "mean"),
            median_torque_gap_pct=("torque_gap_pct_at_peak", "median"),
            median_dv5_mph=("dv5_mph", "median"),
        ).reset_index()

        out_summary = out_dir / "highway_intent_frustration__summary.csv"
        summary.to_csv(out_summary, index=False)


def run_speed_pedal_occupancy(
    out_dir: Path,
    df_clean: pd.DataFrame,
):
    print("[INFO] Running speed × pedal occupancy...")

    cols_needed = ["time_s", "vehicle_speed_mph", "pedal_pct", "mode", "log_id"]
    missing_cols = [c for c in cols_needed if c not in df_clean.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in clean logs: {missing_cols}")

    df = df_clean[cols_needed].copy()
    df.sort_values(["log_id", "time_s"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["dt"] = 0.0
    for _, sub in df.groupby("log_id"):
        idx = sub.index
        dt = sub["time_s"].diff().fillna(0.0).clip(lower=0.0, upper=1.0)
        df.loc[idx, "dt"] = dt

    speed_bins = list(range(45, 95, 5))
    pedal_bins = [0, 5, 10, 20, 30, 40, 60, 100]
    df["speed_bin_mph"] = bin_series(df["vehicle_speed_mph"], speed_bins)
    df["pedal_bin_pct"] = bin_series(df["pedal_pct"], pedal_bins)

    grp = df.groupby(["mode", "speed_bin_mph", "pedal_bin_pct"], dropna=False, observed=False)
    agg = grp.agg(
        time_s=("dt", "sum"),
        samples=("dt", "count"),
    ).reset_index()

    total_time = agg["time_s"].sum()
    agg["time_pct"] = agg["time_s"] / total_time if total_time > 0 else 0.0

    out_path = out_dir / "highway_speed_pedal_occupancy.csv"
    agg.to_csv(out_path, index=False)


def run_tcc_slip_energy_map(
    base_dir: Path,
    out_dir: Path,
    files: FileConfig,
    df_surface_speedspace: pd.DataFrame,
):
    """
    TCC slip energy map:
      - Normalize dragging segments (flex column detection)
      - Estimate engine torque from torque surface
      - Compute slip_energy_raw = |slip_rpm| * duration_s * max(torque, 0)
      - Bin by gear / speed (5-mph) / pedal (5%) and aggregate counts/time/energy
    """
    seg_path = base_dir / files.tcc_dragging_segments
    df_seg_raw = load_csv_if_exists(seg_path)
    if df_seg_raw is None:
        print("[WARN] No tcc dragging segments; skipping slip energy map.")
        return

    print("[INFO] Running TCC slip energy map...")

    df_seg = normalize_tcc_drag_segments(df_seg_raw)
    if df_seg.empty:
        print("[WARN] No usable TCC dragging segments; skipping slip energy map.")
        return

    torque_est = lookup_torque_vectorized(
        df_surface_speedspace,
        df_seg["gear"].to_numpy(),
        df_seg["speed_mph_median"].to_numpy(),
        df_seg["pedal_pct_median"].to_numpy(),
    )
    df_seg["torque_est"] = torque_est
    df_seg["slip_energy_raw"] = np.where(
        np.isnan(df_seg["torque_est"]),
        np.nan,
        df_seg["slip_mean_rpm"].abs() * df_seg["duration_s"] * np.maximum(df_seg["torque_est"], 0),
    )

    df_seg["speed_bin"] = 5 * np.round(df_seg["speed_mph_median"] / 5.0)
    df_seg["pedal_bin"] = 5 * np.round(df_seg["pedal_pct_median"] / 5.0)

    grp = df_seg.groupby(["gear", "speed_bin", "pedal_bin"], dropna=False, observed=False)
    agg = grp.agg(
        episodes_count=("duration_s", "count"),
        total_drag_time_s=("duration_s", "sum"),
        total_slip_energy_raw=("slip_energy_raw", "sum"),
        mean_slip_rpm=("slip_mean_rpm", "mean"),
        mean_torque_est=("torque_est", "mean"),
    ).reset_index()

    agg = agg.rename(columns={"speed_bin": "speed_center_mph", "pedal_bin": "pedal_center_pct"})

    out_path = out_dir / "highway_tcc_slip_energy__by_bin.csv"
    agg.to_csv(out_path, index=False)
    print(f"[INFO] Wrote slip energy map: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Highway Super Analysis: virtual schedule sim + torque pain maps + intent/frustration, etc."
    )
    parser.add_argument(
        "--prepped-dir",
        type=str,
        help="Directory of prepped highway logs (overrides default glob).",
    )
    parser.add_argument(
        "--torque-surface",
        type=str,
        help="Override path to torque surface SPEEDSPACE CSV.",
    )
    parser.add_argument(
        "--intent-episodes",
        type=str,
        help="Override path to intent episodes CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        help="Override output directory (no timestamp suffix).",
    )
    parser.add_argument(
        "--run-virtual-sim",
        action="store_true",
        help="Run virtual schedule simulation for a given schedule_name.",
    )
    parser.add_argument(
        "--run-torque-deficit",
        action="store_true",
        help="Run torque deficit integral (pain map).",
    )
    parser.add_argument(
        "--run-intent-frustration",
        action="store_true",
        help="Run intent × frustration map.",
    )
    parser.add_argument(
        "--run-occupancy",
        action="store_true",
        help="Run speed × pedal occupancy map.",
    )
    parser.add_argument(
        "--run-tcc-heat",
        action="store_true",
        help="Run TCC slip energy map.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all analyses.",
    )
    parser.add_argument(
        "--schedule-name",
        type=str,
        default="candidate",
        help="Schedule name to look up schedule__<name>__*.json for virtual sim.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).parent.resolve()
    files = FileConfig()

    run_virtual = args.all or args.run_virtual_sim
    run_torque_def = args.all or args.run_torque_deficit
    run_intent = args.all or args.run_intent_frustration
    run_occupancy = args.all or args.run_occupancy
    run_tcc_heat = args.all or args.run_tcc_heat

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = make_output_dir(base_dir, label="highway_super_analysis")
    print(f"[INFO] Output directory: {out_dir}")

    if args.prepped_dir:
        prepped_base = Path(args.prepped_dir)
        df_clean_raw = load_clean_logs(prepped_base, "*.csv")
    else:
        df_clean_raw = load_clean_logs(base_dir, files.clean_log_pattern)
    df_clean = unify_clean_columns(df_clean_raw)
    torque_surface_path = Path(args.torque_surface) if args.torque_surface else (base_dir / files.torque_air_spark_speedspace)
    df_torque_speedspace = load_csv_if_exists(torque_surface_path)
    if df_torque_speedspace is None and not args.torque_surface:
        alt = base_dir / "newlogs" / "highway_torque_surface" / files.torque_air_spark_speedspace
        df_torque_speedspace = load_csv_if_exists(alt)
    if df_torque_speedspace is None:
        raise FileNotFoundError("Missing torque_air_spark_surface__SPEEDSPACE.csv; required for most analyses.")

    base_dir_intent = base_dir
    if args.intent_episodes:
        intent_path = Path(args.intent_episodes)
        files.intent_episodes = intent_path.name
        base_dir_intent = intent_path.parent

    if run_occupancy:
        run_speed_pedal_occupancy(out_dir, df_clean)

    if run_torque_def:
        run_torque_deficit_integral(base_dir, out_dir, df_clean, df_torque_speedspace)

    if run_intent:
        run_intent_frustration_map(base_dir_intent, out_dir, files, df_clean, df_torque_speedspace)

    if run_tcc_heat:
        run_tcc_slip_energy_map(base_dir, out_dir, files, df_torque_speedspace)

    if run_virtual:
        schedule = load_schedule_config(base_dir, files, args.schedule_name)
        if schedule is not None:
            run_virtual_schedule_sim(
                base_dir=base_dir,
                out_dir=out_dir,
                files=files,
                schedule=schedule,
                df_surface_speedspace=df_torque_speedspace,
                df_clean=df_clean,
            )
        else:
            print("[WARN] Skipping virtual schedule sim (no schedule config).")


if __name__ == "__main__":
    main()
