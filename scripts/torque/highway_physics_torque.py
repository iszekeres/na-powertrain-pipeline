#!/usr/bin/env python3
"""
Physics-derived torque surface builder for highway logs.

This module extends the ECM torque surface by computing wheel/engine torque
from measured acceleration, correcting for drivetrain efficiency and TCC
multiplication, then aligning with ECM-derived surfaces to build a hybrid
"truth" surface and downshift gain maps.

Usage:
    python highway_physics_torque.py \
        --prepped-dir newlogs/highway_MAX_analysis/prepped \
        --out-dir newlogs/highway_torque_surface \
        --merge-to-existing-surfaces
"""

from __future__ import annotations

import argparse
import math
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Vehicle mass: 5900 lb -> kg
MASS_KG = 5900.0 * 0.453592
TIRE_DIAMETER_IN = 32.5
TIRE_RADIUS_M = TIRE_DIAMETER_IN * 0.0254 / 2.0
FINAL_DRIVE = 3.08
GEAR_RATIOS = {1: 4.027, 2: 2.364, 3: 1.532, 4: 1.152, 5: 0.852, 6: 0.667}
DRIVETRAIN_EFF_CLAMP = (0.80, 0.92)
MPH_TO_MPS = 0.44704

PEDAL_BINS_PHYSICS = np.array([0, 5, 20, 50, 80, 100], dtype=float)
RPM_BINS = np.arange(1200, 5200 + 250, 250)

REQUIRED_CHANNELS: Dict[str, Sequence[str]] = {
    "time_s": ["time_s"],
    "speed_mph": ["speed_mph", "Vehicle Speed (SAE)", "Vehicle Speed"],
    "gear": ["gear_actual__canon", "gear_actual"],
    "engine_rpm": ["engine_rpm", "Engine RPM (SAE)", "Engine RPM"],
    "trans_input_rpm": ["trans_input_rpm", "Trans Input Shaft RPM", "Trans Turbine RPM", "Trans Input RPM"],
    "trans_output_rpm": ["trans_output_rpm", "Trans Output Shaft RPM"],
    "tcc_slip_rpm": ["tcc_slip_rpm", "tcc_slip_rpm_fused", "tcc_slip"],
    "tcc_state": ["tcc_state"],
    "delivered_torque": ["Delivered Engine Torque", "Delivered Torque"],
    "engine_torque": ["Engine Torque"],
    "axle_torque": ["Actual Axle Torque"],
    "cyl_airmass": ["Cylinder Airmass"],
    "airflow": ["Dynamic Airflow", "Mass Airflow (SAE)"],
    "map_kpa": ["Intake Manifold Absolute Pressure (SAE)", "Manifold Absolute Pressure - Hi-Res"],
    "ve_airflow": ["Volumetric Efficiency Airflow"],
    "spark_deg": ["Timing Advance (SAE)"],
    "knock_retard": ["Knock Retard"],
    "lambda_eq": ["WB EQ Ratio 2 (SAE)", "Equivalence Ratio Commanded (SAE)"],
    "brake_pressure": ["Brake Pressure"],
    "pedal_pct": ["pedal_pct", "Accelerator Pedal Position", "Accelerator Pedal Position %"],
    "throttle_pct": ["throttle_pct", "Throttle Position", "Throttle Position (SAE) %"],
    "dfco_flag": ["DFCO Active", "dfco_active"],
    "tcs_flag": ["Traction Control System", "TCS Active", "TCS Request", "TC Active", "StabiliTrak Active"],
    "abs_flag": ["ABS Active", "ABS Event", "ABS Commanded"],
}


def pick_column(df: pd.DataFrame, logical: str, candidates: Sequence[str]) -> Optional[str]:
    for cand in candidates:
        if cand in df.columns:
            return cand
    return None


def resolve_required(df: pd.DataFrame) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    missing: List[str] = []
    for logical, opts in REQUIRED_CHANNELS.items():
        col = pick_column(df, logical, opts)
        if col is None:
            if logical == "abs_flag":
                continue
            if logical == "tcs_flag" and pick_column(df, "abs_flag", REQUIRED_CHANNELS.get("abs_flag", [])):
                continue
            missing.append(logical)
        else:
            resolved[logical] = col
    if missing:
        raise RuntimeError(f"Missing required channels: {missing}")
    return resolved


def normalize_flag(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower().str.strip()
    num = pd.to_numeric(series, errors="coerce")
    active_num = num.fillna(0) > 0
    active_str = s.isin(["1", "true", "on", "yes", "active"])
    return active_num | active_str


def load_prepped(prepped_dir: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    csvs = sorted(prepped_dir.glob("*__prepped.csv"))
    if not csvs:
        raise RuntimeError(f"No prepped logs found under {prepped_dir}")
    frames: List[pd.DataFrame] = []
    resolved: Optional[Dict[str, str]] = None
    for path in csvs:
        df_raw = pd.read_csv(path, low_memory=False)
        mapping = resolve_required(df_raw)
        if resolved is None:
            resolved = mapping
        df = pd.DataFrame()
        df["file_name"] = path.stem
        df["time_s"] = pd.to_numeric(df_raw[mapping["time_s"]], errors="coerce")
        df["speed_mph"] = pd.to_numeric(df_raw[mapping["speed_mph"]], errors="coerce")
        df["gear"] = pd.to_numeric(df_raw[mapping["gear"]], errors="coerce")
        df["engine_rpm"] = pd.to_numeric(df_raw[mapping["engine_rpm"]], errors="coerce")
        df["trans_input_rpm"] = pd.to_numeric(df_raw[mapping["trans_input_rpm"]], errors="coerce")
        df["trans_output_rpm"] = pd.to_numeric(df_raw[mapping["trans_output_rpm"]], errors="coerce")
        df["tcc_slip_rpm"] = pd.to_numeric(df_raw[mapping["tcc_slip_rpm"]], errors="coerce")
        df["tcc_state"] = df_raw[mapping["tcc_state"]].astype(str).str.lower().str.strip()
        df["delivered_torque"] = pd.to_numeric(df_raw[mapping["delivered_torque"]], errors="coerce")
        df["engine_torque_ecm"] = pd.to_numeric(df_raw[mapping["engine_torque"]], errors="coerce")
        df["axle_torque_ecm"] = pd.to_numeric(df_raw[mapping["axle_torque"]], errors="coerce")
        df["cyl_airmass"] = pd.to_numeric(df_raw[mapping["cyl_airmass"]], errors="coerce")
        df["airflow"] = pd.to_numeric(df_raw[mapping["airflow"]], errors="coerce")
        df["map_kpa"] = pd.to_numeric(df_raw[mapping["map_kpa"]], errors="coerce")
        df["ve_airflow"] = pd.to_numeric(df_raw[mapping["ve_airflow"]], errors="coerce")
        df["spark_deg"] = pd.to_numeric(df_raw[mapping["spark_deg"]], errors="coerce")
        df["knock_retard"] = pd.to_numeric(df_raw[mapping["knock_retard"]], errors="coerce")
        df["lambda_eq"] = pd.to_numeric(df_raw[mapping["lambda_eq"]], errors="coerce")
        df["brake_pressure"] = pd.to_numeric(df_raw[mapping["brake_pressure"]], errors="coerce")
        df["pedal_pct"] = pd.to_numeric(df_raw[mapping["pedal_pct"]], errors="coerce")
        df["throttle_pct"] = pd.to_numeric(df_raw[mapping["throttle_pct"]], errors="coerce")
        df["dfco_active"] = normalize_flag(df_raw[mapping["dfco_flag"]])
        tcs_col = mapping.get("tcs_flag")
        abs_col = mapping.get("abs_flag")
        df["tcs_active"] = normalize_flag(df_raw[tcs_col]) if tcs_col else pd.Series(False, index=df.index)
        df["abs_active"] = normalize_flag(df_raw[abs_col]) if abs_col else pd.Series(False, index=df.index)
        frames.append(df)
    assert resolved is not None
    combined = pd.concat(frames, ignore_index=True)
    combined["speed_mps"] = combined["speed_mph"] * MPH_TO_MPS
    return combined, resolved


def compute_filtered_accel(time_s: pd.Series, speed_mps: pd.Series) -> pd.Series:
    accel_raw = np.gradient(speed_mps.to_numpy(), time_s.to_numpy())
    accel = pd.Series(accel_raw, index=time_s.index)
    accel = accel.rolling(window=5, center=True, min_periods=1).median()
    accel = accel.rolling(window=9, center=True, min_periods=1).mean()
    accel = accel.rolling(window=5, center=True, min_periods=1).mean()
    med = accel.median(skipna=True)
    mad = np.median(np.abs(accel - med))
    if mad > 0:
        spikes = (accel - med).abs() > (8 * mad)
        accel.loc[spikes] = np.nan
    accel[accel.abs() > 8.0] = np.nan
    return accel


@dataclass
class AccelSegments:
    mask: pd.Series
    segments: List[Tuple[int, int, float]]


def build_accel_segments(df: pd.DataFrame, accel: pd.Series) -> AccelSegments:
    valid = accel.notna()
    valid &= accel > 0
    valid &= df["speed_mph"] >= 10
    valid &= df["gear"].isin([3, 4, 5, 6])
    valid &= ~df["tcs_active"] & ~df["abs_active"]
    valid &= ~df["dfco_active"]
    valid &= (df["brake_pressure"].fillna(0) < 50)
    jerk = accel.diff().abs()
    valid &= (jerk < 4.0) | jerk.isna()

    keep = np.zeros(len(df), dtype=bool)
    segments: List[Tuple[int, int, float]] = []
    start_idx: Optional[int] = None
    time = df["time_s"].to_numpy()
    for i, ok in enumerate(valid.to_numpy()):
        if ok and start_idx is None:
            start_idx = i
        if (not ok or i == len(valid) - 1) and start_idx is not None:
            end_idx = i if ok and i == len(valid) - 1 else i - 1
            duration = float(time[end_idx] - time[start_idx])
            if duration >= 0.5:
                keep[start_idx : end_idx + 1] = True
                segments.append((start_idx, end_idx, duration))
            start_idx = None
    mask = pd.Series(keep, index=df.index)
    return AccelSegments(mask=mask, segments=segments)


def estimate_efficiency(df: pd.DataFrame) -> Dict[int, float]:
    ratios_by_gear: Dict[int, List[float]] = defaultdict(list)
    base = (
        (df["tcc_state"] == "locked")
        & df["physics_engine_raw"].notna()
        & df["delivered_torque"].notna()
        & (df["physics_engine_raw"] > 50)
        & (df["delivered_torque"] > 50)
        & (df["accel_mps2_clean"].between(0.05, 1.5))
        & (df["brake_pressure"].fillna(0) < 15)
        & ~df["tcs_active"]
        & ~df["abs_active"]
    )
    for gear in [3, 4, 5, 6]:
        mask = base & (df["gear"] == gear)
        if not mask.any():
            continue
        ratios = (df.loc[mask, "physics_engine_raw"] / df.loc[mask, "delivered_torque"]).replace([np.inf, -np.inf], np.nan)
        ratios = ratios[(ratios > 0.6) & (ratios < 1.2)].dropna()
        if ratios.empty:
            continue
        ratios_by_gear[gear] = ratios.tolist()
    all_ratios = [v for vals in ratios_by_gear.values() for v in vals]
    global_med = float(np.median(all_ratios)) if all_ratios else 0.86
    eff_map: Dict[int, float] = {}
    for gear in [3, 4, 5, 6]:
        if ratios_by_gear.get(gear):
            eff = float(np.median(ratios_by_gear[gear]))
        else:
            eff = global_med
        eff = float(np.clip(eff, DRIVETRAIN_EFF_CLAMP[0], DRIVETRAIN_EFF_CLAMP[1]))
        eff_map[gear] = eff
    return eff_map


def apply_tcc_correction(df: pd.DataFrame) -> None:
    turbine = df["trans_input_rpm"]
    eng = df["engine_rpm"]
    slip = eng - turbine
    mult = eng / turbine.replace(0, np.nan)
    mult = mult.clip(lower=1.0, upper=1.2)
    open_mask = df["tcc_state"] == "open"
    slip_mask = df["tcc_state"] == "slip"
    bad_open = open_mask & (slip.abs() > 200)
    mult.loc[bad_open] = np.nan
    active = (open_mask | slip_mask) & mult.notna()
    invalid = (open_mask | slip_mask) & mult.isna()
    df.loc[active, "physics_engine_torque"] = df.loc[active, "physics_engine_torque"] * mult[active]
    df.loc[active, "physics_wheel_torque"] = df.loc[active, "physics_wheel_torque"] * mult[active]
    df.loc[invalid, ["physics_engine_torque", "physics_wheel_torque"]] = np.nan
    df["tcc_multiplier"] = mult


def compute_physics_torque(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, float]]:
    accel = compute_filtered_accel(df["time_s"], df["speed_mps"])
    segments = build_accel_segments(df, accel)
    df["accel_mps2_clean"] = np.where(segments.mask, accel, np.nan)
    df.loc[df["accel_mps2_clean"] < 0.05, "accel_mps2_clean"] = np.nan
    df.loc[df["speed_mph"] < 10, "accel_mps2_clean"] = np.nan

    df["gear_ratio"] = df["gear"].map(GEAR_RATIOS)
    df["physics_wheel_torque"] = df["accel_mps2_clean"] * MASS_KG * TIRE_RADIUS_M
    df.loc[df["physics_wheel_torque"] <= 0, "physics_wheel_torque"] = np.nan
    df["physics_engine_raw"] = df["physics_wheel_torque"] / (df["gear_ratio"] * FINAL_DRIVE)

    eff_map = estimate_efficiency(df)
    df["drivetrain_eff"] = df["gear"].map(eff_map)
    df["physics_engine_torque"] = df["physics_engine_raw"] / df["drivetrain_eff"]
    df.loc[df["physics_engine_torque"] <= 0, "physics_engine_torque"] = np.nan

    apply_tcc_correction(df)

    df.loc[df["physics_engine_torque"] > 2000, "physics_engine_torque"] = np.nan
    df.loc[df["physics_wheel_torque"] > 3000, "physics_wheel_torque"] = np.nan

    df["physics_valid"] = df["physics_engine_torque"].notna()
    return df, eff_map


def agg_nanmedian(series: pd.Series) -> float:
    arr = series.dropna()
    return float(np.nanmedian(arr)) if not arr.empty else math.nan


def bin_pedal(series: pd.Series) -> pd.Series:
    return pd.cut(series, PEDAL_BINS_PHYSICS, include_lowest=True)


def build_physics_surface_rpm(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    df_use = df[df["physics_valid"] & df["gear"].isin([3, 4, 5, 6])].copy()
    df_use["rpm_bin"] = pd.cut(df_use["engine_rpm"], RPM_BINS)
    df_use["pedal_bin"] = bin_pedal(df_use["pedal_pct"])
    cols = [
        "gear",
        "rpm_center",
        "pedal_center",
        "n_samples",
        "physics_engine_torque_median",
        "physics_wheel_torque_median",
        "airflow_median",
        "cyl_airmass_median",
        "map_kpa_median",
        "spark_deg_median",
        "knock_retard_median",
    ]
    records: List[Dict[str, float]] = []
    for (gear, rpm_bin, pedal_bin), sub in df_use.groupby(["gear", "rpm_bin", "pedal_bin"]):
        if pd.isna(rpm_bin) or pd.isna(pedal_bin):
            continue
        n = len(sub)
        if n < 20:
            continue
        records.append(
            {
                "gear": int(gear),
                "rpm_center": rpm_bin.mid,
                "pedal_center": pedal_bin.mid,
                "n_samples": n,
                "physics_engine_torque_median": agg_nanmedian(sub["physics_engine_torque"]),
                "physics_wheel_torque_median": agg_nanmedian(sub["physics_wheel_torque"]),
                "airflow_median": agg_nanmedian(sub["airflow"]),
                "cyl_airmass_median": agg_nanmedian(sub["cyl_airmass"]),
                "map_kpa_median": agg_nanmedian(sub["map_kpa"]),
                "spark_deg_median": agg_nanmedian(sub["spark_deg"]),
                "knock_retard_median": agg_nanmedian(sub["knock_retard"]),
            }
        )
    surface = pd.DataFrame(records, columns=cols)
    surface.to_csv(out_dir / "physics_torque_surface__by_gear.csv", index=False)
    return surface


def build_physics_surface_speed(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    df_use = df[df["physics_valid"] & df["gear"].isin([3, 4, 5, 6])].copy()
    max_speed_val = df_use["speed_mph"].max() if not df_use.empty else 100
    max_speed = int(math.ceil(max_speed_val)) if not math.isnan(max_speed_val) else 100
    edges = np.arange(10, max_speed + 1, 1)
    if len(edges) < 2:
        edges = np.array([0, 1])
    df_use["speed_bin"] = pd.cut(df_use["speed_mph"], edges)
    df_use["pedal_bin"] = bin_pedal(df_use["pedal_pct"])
    cols = [
        "gear",
        "speed_center",
        "pedal_center",
        "n_samples",
        "physics_engine_torque_median",
        "physics_wheel_torque_median",
        "airflow_median",
        "cyl_airmass_median",
        "map_kpa_median",
        "spark_deg_median",
        "knock_retard_median",
    ]
    records: List[Dict[str, float]] = []
    for (gear, speed_bin, pedal_bin), sub in df_use.groupby(["gear", "speed_bin", "pedal_bin"]):
        if pd.isna(speed_bin) or pd.isna(pedal_bin):
            continue
        n = len(sub)
        if n < 20:
            continue
        records.append(
            {
                "gear": int(gear),
                "speed_center": speed_bin.mid,
                "pedal_center": pedal_bin.mid,
                "n_samples": n,
                "physics_engine_torque_median": agg_nanmedian(sub["physics_engine_torque"]),
                "physics_wheel_torque_median": agg_nanmedian(sub["physics_wheel_torque"]),
                "airflow_median": agg_nanmedian(sub["airflow"]),
                "cyl_airmass_median": agg_nanmedian(sub["cyl_airmass"]),
                "map_kpa_median": agg_nanmedian(sub["map_kpa"]),
                "spark_deg_median": agg_nanmedian(sub["spark_deg"]),
                "knock_retard_median": agg_nanmedian(sub["knock_retard"]),
            }
        )
    surface = pd.DataFrame(records, columns=cols)
    surface.to_csv(out_dir / "physics_torque_surface__SPEEDSPACE.csv", index=False)
    return surface


def load_ecm_surfaces(out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rpm_path = out_dir / "torque_air_spark_surface__by_gear.csv"
    speed_path = out_dir / "torque_air_spark_surface__SPEEDSPACE.csv"
    fallback_rpm = out_dir / "torque_surface__by_gear.csv"
    if not rpm_path.exists() and fallback_rpm.exists():
        rpm_path = fallback_rpm
    if not rpm_path.exists():
        raise RuntimeError("Missing ECM torque surface CSVs; run highway_torque_surface.py first.")
    if not speed_path.exists():
        raise RuntimeError("Missing ECM speed-space torque surface CSV.")
    rpm = pd.read_csv(rpm_path)
    speed = pd.read_csv(speed_path)
    return rpm, speed


def aggregate_to_physics_bins_rpm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rpm_bin_phys"] = pd.cut(df["rpm_center"], RPM_BINS)
    df["pedal_bin_phys"] = pd.cut(df["pedal_center"], PEDAL_BINS_PHYSICS, include_lowest=True)
    df = df.dropna(subset=["rpm_bin_phys", "pedal_bin_phys"])
    agg = (
        df.groupby(["gear", "rpm_bin_phys", "pedal_bin_phys"])
        .agg(
            {
                "eng_torque_mean": "median",
                "eng_torque_p50": "median",
                "axle_torque_mean": "median",
                "n_samples": "sum",
            }
        )
        .reset_index()
    )
    agg["rpm_center"] = agg["rpm_bin_phys"].apply(lambda x: x.mid)
    agg["pedal_center"] = agg["pedal_bin_phys"].apply(lambda x: x.mid)
    return agg


def aggregate_to_physics_bins_speed(df: pd.DataFrame, speed_edges: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["speed_bin_phys"] = pd.cut(df["speed_center"], speed_edges)
    df["pedal_bin_phys"] = pd.cut(df["pedal_center"], PEDAL_BINS_PHYSICS, include_lowest=True)
    df = df.dropna(subset=["speed_bin_phys", "pedal_bin_phys"])
    agg = (
        df.groupby(["gear", "speed_bin_phys", "pedal_bin_phys"])
        .agg(
            {
                "eng_torque_mean": "median",
                "eng_torque_p50": "median",
                "axle_torque_mean": "median",
                "n_samples": "sum",
            }
        )
        .reset_index()
    )
    agg["speed_center"] = agg["speed_bin_phys"].apply(lambda x: x.mid)
    agg["pedal_center"] = agg["pedal_bin_phys"].apply(lambda x: x.mid)
    return agg


def merge_surfaces_rpm(phys: pd.DataFrame, ecm: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if phys.empty:
        merged_empty = pd.DataFrame()
        merged_empty.to_csv(out_dir / "torque_surface_compare__RPMSPACE.csv", index=False)
        return merged_empty
    phys = phys.copy()
    phys["rpm_bin_phys"] = pd.cut(phys["rpm_center"], RPM_BINS)
    phys["pedal_bin_phys"] = pd.cut(phys["pedal_center"], PEDAL_BINS_PHYSICS, include_lowest=True)
    phys = phys.dropna(subset=["rpm_bin_phys", "pedal_bin_phys"])
    ecm_binned = aggregate_to_physics_bins_rpm(ecm)
    merged = pd.merge(
        phys,
        ecm_binned,
        on=["gear", "rpm_bin_phys", "pedal_bin_phys"],
        how="outer",
        suffixes=("_phys", "_ecm"),
    )
    merged["rpm_center"] = merged["rpm_bin_phys"].apply(lambda x: x.mid if pd.notna(x) else math.nan)
    merged["pedal_center"] = merged["pedal_bin_phys"].apply(lambda x: x.mid if pd.notna(x) else math.nan)
    merged["delta_torque_physics_vs_ecm"] = merged["physics_engine_torque_median"] - merged["eng_torque_mean"]
    merged["delta_torque_physics_vs_ecm_pct"] = merged["delta_torque_physics_vs_ecm"] / merged["eng_torque_mean"] * 100.0
    merged.to_csv(out_dir / "torque_surface_compare__RPMSPACE.csv", index=False)
    return merged


def merge_surfaces_speed(phys: pd.DataFrame, ecm: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if phys.empty:
        merged_empty = pd.DataFrame()
        merged_empty.to_csv(out_dir / "torque_surface_compare__SPEEDSPACE.csv", index=False)
        return merged_empty
    max_speed_list = [val for val in [phys["speed_center"].max(), ecm["speed_center"].max()] if not math.isnan(val)]
    max_speed = int(math.ceil(max(max_speed_list))) if max_speed_list else 100
    speed_edges = np.arange(10, max_speed + 1, 1)
    if len(speed_edges) < 2:
        speed_edges = np.array([0, 1])
    phys = phys.copy()
    phys["speed_bin_phys"] = pd.cut(phys["speed_center"], speed_edges)
    phys["pedal_bin_phys"] = pd.cut(phys["pedal_center"], PEDAL_BINS_PHYSICS, include_lowest=True)
    phys = phys.dropna(subset=["speed_bin_phys", "pedal_bin_phys"])
    ecm_binned = aggregate_to_physics_bins_speed(ecm, speed_edges)
    merged = pd.merge(
        phys,
        ecm_binned,
        on=["gear", "speed_bin_phys", "pedal_bin_phys"],
        how="outer",
        suffixes=("_phys", "_ecm"),
    )
    merged["speed_center"] = merged["speed_bin_phys"].apply(lambda x: x.mid if pd.notna(x) else math.nan)
    merged["pedal_center"] = merged["pedal_bin_phys"].apply(lambda x: x.mid if pd.notna(x) else math.nan)
    merged["delta_torque_physics_vs_ecm"] = merged["physics_engine_torque_median"] - merged["eng_torque_mean"]
    merged["delta_torque_physics_vs_ecm_pct"] = merged["delta_torque_physics_vs_ecm"] / merged["eng_torque_mean"] * 100.0
    merged.to_csv(out_dir / "torque_surface_compare__SPEEDSPACE.csv", index=False)
    return merged


def build_downshift_gain(phys_speed: pd.DataFrame, pairs: Sequence[Tuple[int, int]], out_path: Path) -> pd.DataFrame:
    if phys_speed.empty or "pedal_center" not in phys_speed.columns or "speed_center" not in phys_speed.columns:
        empty_cols = [
            "from_gear",
            "to_gear",
            "speed_center",
            "pedal_center",
            "physics_engine_torque_before",
            "physics_engine_torque_after",
            "delta_torque",
            "delta_torque_pct",
            "airflow_delta",
            "cyl_airmass_delta",
            "spark_delta",
            "knock_delta",
        ]
        out_df = pd.DataFrame(columns=empty_cols)
        out_df.to_csv(out_path, index=False)
        return out_df
    phys_speed = phys_speed.copy()
    phys_speed["pedal_bin_label"] = pd.cut(phys_speed["pedal_center"], PEDAL_BINS_PHYSICS, include_lowest=True)
    rows: List[Dict[str, float]] = []
    for from_gear, to_gear in pairs:
        high = phys_speed[phys_speed["gear"] == from_gear]
        low = phys_speed[phys_speed["gear"] == to_gear]
        merged = pd.merge(
            high,
            low,
            on=["speed_center", "pedal_bin_label"],
            suffixes=("_high", "_low"),
        )
        for _, r in merged.iterrows():
            before = r["physics_engine_torque_median_high"]
            after = r["physics_engine_torque_median_low"]
            if pd.isna(before) or pd.isna(after) or before <= 0:
                continue
            delta = after - before
            pct = delta / before * 100.0
            rows.append(
                {
                    "from_gear": from_gear,
                    "to_gear": to_gear,
                    "speed_center": r["speed_center"],
                    "pedal_center": r["pedal_bin_label"].mid if pd.notna(r["pedal_bin_label"]) else math.nan,
                    "physics_engine_torque_before": before,
                    "physics_engine_torque_after": after,
                    "delta_torque": delta,
                    "delta_torque_pct": pct,
                    "airflow_delta": r["airflow_median_low"] - r["airflow_median_high"],
                    "cyl_airmass_delta": r["cyl_airmass_median_low"] - r["cyl_airmass_median_high"],
                    "spark_delta": r["spark_deg_median_low"] - r["spark_deg_median_high"],
                    "knock_delta": r["knock_retard_median_low"] - r["knock_retard_median_high"],
                }
            )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    return out_df


def build_hybrid_surface(phys: pd.DataFrame, ecm: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    if phys.empty:
        out_df = pd.DataFrame()
        out_df.to_csv(out_path, index=False)
        return out_df
    phys = phys.copy()
    phys["rpm_bin_phys"] = pd.cut(phys["rpm_center"], RPM_BINS)
    phys["pedal_bin_phys"] = pd.cut(phys["pedal_center"], PEDAL_BINS_PHYSICS, include_lowest=True)
    phys = phys.dropna(subset=["rpm_bin_phys", "pedal_bin_phys"])
    ecm_binned = aggregate_to_physics_bins_rpm(ecm)
    merged = pd.merge(
        phys,
        ecm_binned,
        on=["gear", "rpm_bin_phys", "pedal_bin_phys"],
        how="outer",
        suffixes=("_phys", "_ecm"),
    )
    merged["rpm_center"] = merged["rpm_bin_phys"].apply(lambda x: x.mid if pd.notna(x) else math.nan)
    merged["pedal_center"] = merged["pedal_bin_phys"].apply(lambda x: x.mid if pd.notna(x) else math.nan)

    blended_vals = []
    sources = []
    for _, row in merged.iterrows():
        phys_val = row.get("physics_engine_torque_median", math.nan)
        ecm_val = row.get("eng_torque_mean", math.nan)
        n_phys = row.get("n_samples_phys", 0)
        if not math.isnan(phys_val) and n_phys >= 50:
            blended_vals.append(phys_val)
            sources.append("physics")
        elif not math.isnan(phys_val) and not math.isnan(ecm_val) and 20 <= n_phys < 50:
            blended_vals.append(0.7 * phys_val + 0.3 * ecm_val)
            sources.append("blend")
        elif math.isnan(phys_val) and not math.isnan(ecm_val):
            blended_vals.append(ecm_val)
            sources.append("ecm")
        elif not math.isnan(phys_val):
            blended_vals.append(phys_val)
            sources.append("physics_only")
        else:
            blended_vals.append(math.nan)
            sources.append("none")
    merged["hybrid_engine_torque"] = blended_vals
    merged["hybrid_source"] = sources
    merged.to_csv(out_path, index=False)
    return merged


def build_hybrid_speed_surface(phys_speed: pd.DataFrame, ecm_speed: pd.DataFrame) -> pd.DataFrame:
    if phys_speed.empty:
        return pd.DataFrame()
    phys = phys_speed.copy()
    max_speed_list = [val for val in [phys["speed_center"].max(), ecm_speed["speed_center"].max()] if not math.isnan(val)]
    max_speed = int(math.ceil(max(max_speed_list))) if max_speed_list else 100
    edges = np.arange(10, max_speed + 1, 1)
    if len(edges) < 2:
        edges = np.array([0, 1])
    phys["speed_bin_phys"] = pd.cut(phys["speed_center"], edges)
    phys["pedal_bin_phys"] = pd.cut(phys["pedal_center"], PEDAL_BINS_PHYSICS, include_lowest=True)
    phys = phys.dropna(subset=["speed_bin_phys", "pedal_bin_phys"])
    ecm_binned = aggregate_to_physics_bins_speed(ecm_speed, edges)
    merged = pd.merge(
        phys,
        ecm_binned,
        on=["gear", "speed_bin_phys", "pedal_bin_phys"],
        how="outer",
        suffixes=("_phys", "_ecm"),
    )
    merged["speed_center"] = merged["speed_bin_phys"].apply(lambda x: x.mid if pd.notna(x) else math.nan)
    merged["pedal_center"] = merged["pedal_bin_phys"].apply(lambda x: x.mid if pd.notna(x) else math.nan)
    hybrid_vals = []
    sources = []
    for _, row in merged.iterrows():
        phys_val = row.get("physics_engine_torque_median", math.nan)
        ecm_val = row.get("eng_torque_mean", math.nan)
        n_phys = row.get("n_samples_phys", 0)
        if not math.isnan(phys_val) and n_phys >= 50:
            hybrid_vals.append(phys_val)
            sources.append("physics")
        elif not math.isnan(phys_val) and not math.isnan(ecm_val) and 20 <= n_phys < 50:
            hybrid_vals.append(0.7 * phys_val + 0.3 * ecm_val)
            sources.append("blend")
        elif math.isnan(phys_val) and not math.isnan(ecm_val):
            hybrid_vals.append(ecm_val)
            sources.append("ecm")
        elif not math.isnan(phys_val):
            hybrid_vals.append(phys_val)
            sources.append("physics_only")
        else:
            hybrid_vals.append(math.nan)
            sources.append("none")
    merged["hybrid_engine_torque"] = hybrid_vals
    merged["hybrid_source"] = sources
    return merged


def summarize_efficiency(eff_map: Dict[int, float]) -> List[str]:
    lines = ["Drivetrain efficiency (clamped 0.80-0.92):"]
    for g in sorted(eff_map.keys()):
        lines.append(f"  Gear {g}: {eff_map[g]:.4f}")
    return lines


def summarize_downshift(down: pd.DataFrame, label: str) -> List[str]:
    lines = [f"Downshift {label} torque gains:"]
    if down.empty:
        lines.append("  No valid coverage.")
        return lines
    for low, high in zip(PEDAL_BINS_PHYSICS[:-1], PEDAL_BINS_PHYSICS[1:]):
        sub = down[(down["pedal_center"] >= low) & (down["pedal_center"] <= high)]
        if sub.empty:
            continue
        lines.append(
            f"  Pedal {low:.0f}-{high:.0f}%: median Δtq {sub['delta_torque'].median():.1f} lbft "
            f"({sub['delta_torque_pct'].median():.1f}%); speed span "
            f"{sub['speed_center'].min():.1f}-{sub['speed_center'].max():.1f} mph"
        )
    return lines


def summarize_boundaries(down: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    if down.empty:
        return ["No downshift boundary data."]
    spark = down["spark_delta"].dropna()
    knock = down["knock_delta"].dropna()
    airflow = down["airflow_delta"].dropna()
    cyl = down["cyl_airmass_delta"].dropna()
    if not spark.empty:
        lines.append(f"Spark change at boundaries: median {spark.median():.2f} deg (min {spark.min():.2f}, max {spark.max():.2f})")
    if not knock.empty:
        lines.append(f"Knock delta at boundaries: median {knock.median():.2f} deg (min {knock.min():.2f}, max {knock.max():.2f})")
    if not airflow.empty:
        lines.append(f"Airflow delta at boundaries: median {airflow.median():.3f} g/s (min {airflow.min():.3f}, max {airflow.max():.3f})")
    if not cyl.empty:
        lines.append(f"Airmass delta at boundaries: median {cyl.median():.4f} g (min {cyl.min():.4f}, max {cyl.max():.4f})")
    return lines


def shift_bias_suggestions(down: pd.DataFrame, pair: Tuple[int, int]) -> List[str]:
    lines = [f"Shift bias suggestions for {pair[0]}->{pair[1]} (mph where Δtq>40 lbft):"]
    if down.empty:
        lines.append("  No valid coverage.")
        return lines
    for low, high in zip(PEDAL_BINS_PHYSICS[:-1], PEDAL_BINS_PHYSICS[1:]):
        sub = down[(down["pedal_center"] >= low) & (down["pedal_center"] <= high) & (down["delta_torque"] > 40)]
        if sub.empty:
            lines.append(f"  Pedal {low:.0f}-{high:.0f}%: no gain >40 lbft")
            continue
        lines.append(f"  Pedal {low:.0f}-{high:.0f}%: target shift near {sub['speed_center'].median():.1f} mph")
    return lines


def summarize_gear6(df: pd.DataFrame, down65: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    mid_pedal = df[(df["gear"] == 6) & (df["pedal_pct"].between(20, 50)) & (df["speed_mph"] >= 65)]
    if mid_pedal.empty:
        lines.append("6th gear mid-pedal coverage: none in logs.")
        return lines
    lock_mask = (mid_pedal["tcc_state"] == "locked") | (mid_pedal["tcc_multiplier"].fillna(1.0) < 1.05)
    lock_rate = lock_mask.mean() * 100.0
    slip_rpm = (mid_pedal["engine_rpm"] - mid_pedal["trans_input_rpm"]).abs().dropna()
    slip95 = float(np.nanpercentile(slip_rpm, 95)) if not slip_rpm.empty else math.nan
    lines.append(f"6th gear mid-pedal lock ratio: {lock_rate:.1f}% (95th percentile slip {slip95:.1f} rpm)")
    if not down65.empty:
        med_gain = down65["delta_torque"].median()
        lines.append(f"6->5 torque gain median: {med_gain:.1f} lbft; 6th underutilized if consistently >80 lbft")
    return lines


def summarize_slip(df: pd.DataFrame) -> List[str]:
    slip_rows = df[(df["tcc_multiplier"].fillna(1.0) > 1.1) & (df["pedal_pct"] > 20) & df["physics_valid"]]
    if slip_rows.empty:
        return ["No high-slip torque multiplication detected."]
    pct = len(slip_rows) / max(len(df), 1) * 100.0
    slip_mag = (slip_rows["engine_rpm"] - slip_rows["trans_input_rpm"]).abs()
    return [f"High-slip rows (>1.1x): {pct:.2f}% of valid; median slip {slip_mag.median():.1f} rpm"]


def write_summary(path: Path, sections: List[List[str]]) -> None:
    lines: List[str] = []
    for block in sections:
        lines.extend(block)
        lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def make_zip(out_dir: Path) -> None:
    names = [
        "physics_torque_surface__by_gear.csv",
        "physics_torque_surface__SPEEDSPACE.csv",
        "torque_surface_compare__RPMSPACE.csv",
        "torque_surface_compare__SPEEDSPACE.csv",
        "physics_torque_gain__downshift_map.csv",
        "hybrid_torque_surface.csv",
        "hybrid_downshift_gain_map.csv",
        "PHYSICS_TORQUE__SUMMARY.txt",
        "PHYSICS_TORQUE__SUMMARY__RPMSPACE.txt",
        "PHYSICS_TORQUE__SUMMARY__SPEEDSPACE.txt",
        "HYBRID_TORQUE__SUMMARY.txt",
    ]
    zip_path = out_dir / "highway_torque_surface_outputs_physics.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            p = out_dir / name
            if p.exists():
                zf.write(p, p.name)
    print(f"[OK] Wrote zip {zip_path} ({zip_path.stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Physics-derived torque surface and comparisons")
    parser.add_argument("--prepped-dir", default="newlogs/highway_MAX_analysis/prepped", help="Folder with *_prepped.csv")
    parser.add_argument("--out-dir", default="newlogs/highway_torque_surface", help="Output directory")
    parser.add_argument(
        "--merge-to-existing-surfaces",
        action="store_true",
        help="Align and blend with existing ECM torque surfaces in the output directory",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading prepped logs from {args.prepped_dir} ...")
    df, _ = load_prepped(Path(args.prepped_dir))
    print(f"[INFO] Loaded {len(df):,} rows.")

    print("[INFO] Computing physics torque ...")
    df, eff_map = compute_physics_torque(df)

    print("[INFO] Building physics torque surfaces ...")
    surf_rpm = build_physics_surface_rpm(df, out_dir)
    surf_speed = build_physics_surface_speed(df, out_dir)

    summary_core = summarize_efficiency(eff_map)

    if args.merge_to_existing_surfaces:
        ecm_rpm, ecm_speed = load_ecm_surfaces(out_dir)
        print("[INFO] Aligning physics vs ECM surfaces ...")
        compare_rpm = merge_surfaces_rpm(surf_rpm, ecm_rpm, out_dir)
        compare_speed = merge_surfaces_speed(surf_speed, ecm_speed, out_dir)

        print("[INFO] Building downshift gain maps ...")
        downshift_pairs = [(6, 5), (5, 4)]
        physics_down = build_downshift_gain(surf_speed, downshift_pairs, out_dir / "physics_torque_gain__downshift_map.csv")
        hybrid_surface = build_hybrid_surface(surf_rpm, ecm_rpm, out_dir / "hybrid_torque_surface.csv")
        hybrid_speed = build_hybrid_speed_surface(surf_speed, ecm_speed)
        hybrid_down = build_downshift_gain(hybrid_speed, downshift_pairs, out_dir / "hybrid_downshift_gain_map.csv")

        down65 = physics_down[physics_down["from_gear"] == 6]
        down54 = physics_down[physics_down["from_gear"] == 5]

        print("[INFO] Writing summaries ...")
        write_summary(
            out_dir / "PHYSICS_TORQUE__SUMMARY__RPMSPACE.txt",
            [summary_core, ["RPM-space physics surface rows: " + str(len(surf_rpm))], summarize_downshift(physics_down, "speed bins"), summarize_boundaries(physics_down)],
        )
        write_summary(
            out_dir / "PHYSICS_TORQUE__SUMMARY__SPEEDSPACE.txt",
            [summary_core, ["Speed-space physics surface rows: " + str(len(surf_speed))], summarize_downshift(physics_down, "speed bins"), summarize_boundaries(physics_down)],
        )
        write_summary(
            out_dir / "PHYSICS_TORQUE__SUMMARY.txt",
            [
                summary_core,
                [
                    "Physics vs ECM RPM-space delta median: "
                    + (f"{compare_rpm['delta_torque_physics_vs_ecm'].median():.1f}" if not compare_rpm.empty else "n/a"),
                ],
                summarize_gear6(df, down65),
                shift_bias_suggestions(down65, (6, 5)),
                shift_bias_suggestions(down54, (5, 4)),
                summarize_slip(df),
                summarize_boundaries(physics_down),
            ],
        )
        write_summary(
            out_dir / "HYBRID_TORQUE__SUMMARY.txt",
            [
                ["Hybrid surface rows: " + str(len(hybrid_surface))],
                ["Hybrid downshift rows: " + str(len(hybrid_down))],
                summarize_downshift(hybrid_down, "hybrid"),
            ],
        )
    else:
        print("[WARN] --merge-to-existing-surfaces not set; skipping ECM alignment and hybrid outputs.")
        build_downshift_gain(surf_speed, [(6, 5), (5, 4)], out_dir / "physics_torque_gain__downshift_map.csv")
        write_summary(out_dir / "PHYSICS_TORQUE__SUMMARY.txt", [summary_core])
        (out_dir / "PHYSICS_TORQUE__SUMMARY__RPMSPACE.txt").write_text("\n".join(summary_core), encoding="utf-8")
        (out_dir / "PHYSICS_TORQUE__SUMMARY__SPEEDSPACE.txt").write_text("\n".join(summary_core), encoding="utf-8")

    make_zip(out_dir)
    print("[DONE] Physics torque analysis complete.")


if __name__ == "__main__":
    main()
