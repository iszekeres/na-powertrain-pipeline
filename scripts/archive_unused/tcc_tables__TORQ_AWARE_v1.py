#!/usr/bin/env python
"""
tcc_tables__TORQ_AWARE_v1.py

Torque-aware TCC APPLY/RELEASE tables aligned to torque-aware shift maps.
Outputs:
  newlogs/output/01_tables/tcc/TCC_APPLY__Throttle17__COMFORT_TORQAWARE.tsv
  newlogs/output/01_tables/tcc/TCC_RELEASE__Throttle17__COMFORT_TORQAWARE.tsv
  newlogs/output/01_tables/tcc/TCC_APPLY__Throttle17__PERF_TORQAWARE.tsv
  newlogs/output/01_tables/tcc/TCC_RELEASE__Throttle17__PERF_TORQAWARE.tsv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from torque_curve_loader import load_torque_curve, TorqueCurve

# Drivetrain constants
GEAR_RATIOS: Dict[int, float] = {1: 4.03, 2: 2.36, 3: 1.53, 4: 1.15, 5: 0.85, 6: 0.67}
FINAL_DRIVE = 3.08
TIRE_DIAMETER_IN = 32.5

TPS_AXIS: List[int] = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]
GEARS = [3, 4, 5, 6]

PLATEAU_LO = 1900.0
PLATEAU_HI = 2500.0
HUMP_LO = 2500.0
HUMP_HI = 2900.0
DIP_LO = 3000.0
DIP_HI = 3300.0

RPM_CAP_WOT = 6200.0

SENTINEL_APPLY = 318.0
SENTINEL_RELEASE = 317.0
SENTINEL_THRESHOLD = 300.0


def mph_from_rpm(rpm: float | np.ndarray, gear: int) -> float | np.ndarray:
    gr = GEAR_RATIOS[gear]
    return (np.asarray(rpm) * TIRE_DIAMETER_IN) / (gr * FINAL_DRIVE * 336.0)


def rpm_from_mph(mph: float | np.ndarray, gear: int) -> float | np.ndarray:
    gr = GEAR_RATIOS[gear]
    return (np.asarray(mph) * gr * FINAL_DRIVE * 336.0) / TIRE_DIAMETER_IN


def load_shift_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return df


def shift_value(up_df: pd.DataFrame, row_label: str, tps: int) -> float:
    row = up_df[up_df[up_df.columns[0]] == row_label]
    if row.empty:
        return np.nan
    return float(row.iloc[0][str(tps)])


def load_slip_curve(path: Path) -> Callable[[np.ndarray], np.ndarray]:
    df = pd.read_csv(path, sep="\t")
    if "rpm" not in df.columns:
        raise ValueError(f"Slip table missing 'rpm' column: {path}")
    rpm_axis = df["rpm"].to_numpy(dtype=float)
    slip_cols = [c for c in df.columns if c != "rpm"]
    col_rpms = [float(c) for c in slip_cols]
    slip_vals = df[slip_cols].to_numpy(dtype=float).mean(axis=0)

    def interp_slip(rpm_turbine: np.ndarray | float) -> np.ndarray:
        arr = np.asarray(rpm_turbine, dtype=float)
        arr_clamped = np.clip(arr, min(col_rpms), max(col_rpms))
        return np.interp(arr_clamped, col_rpms, slip_vals)

    return interp_slip


def torque_threshold(mode: str, tps: int) -> float:
    mode = mode.lower()
    if mode == "comfort":
        if tps <= 12:
            return 0.55
        if tps <= 25:
            return 0.60
        if tps <= 56:
            return 0.65
        if tps <= 75:
            return 0.70
        return 1.1  # lockout handled elsewhere
    # performance
    if tps <= 25:
        return 0.60
    if tps <= 50:
        return 0.68
    if tps <= 75:
        return 0.75
    return 0.80


def slip_threshold(mode: str, gear: int) -> float:
    if mode == "comfort":
        return 25.0 if gear in (3, 4) else 20.0
    return 20.0 if gear in (3, 4) else 18.0


def gap_for_release(mode: str, tps: int) -> float:
    if mode == "comfort":
        if tps <= 25:
            return 4.0
        if tps <= 56:
            return 5.0
        return 6.0
    if tps <= 25:
        return 3.0
    if tps <= 56:
        return 4.0
    return 5.0


def select_apply_mph(
    mode: str,
    gear: int,
    tps: int,
    mph_min: float,
    mph_max: float,
    curve: TorqueCurve,
    slip_interp: Callable[[np.ndarray], np.ndarray],
) -> float:
    # lockout rules
    if mode == "comfort" and tps >= 81:
        return SENTINEL_APPLY
    if mode == "performance" and tps >= 94:
        return SENTINEL_APPLY

    grid = np.linspace(mph_min, mph_max, 100)
    best_cost = float("inf")
    best_mph = np.nan
    torque_thresh = torque_threshold(mode, tps)
    slip_lim = slip_threshold(mode, gear)

    for mph in grid:
        rpm_lock = float(rpm_from_mph(mph, gear))
        if rpm_lock <= 0 or rpm_lock > RPM_CAP_WOT + 50:
            continue
        T_norm = curve.interp(rpm_lock) / curve.T_peak if curve.T_peak > 0 else 0.0
        if T_norm < torque_thresh:
            continue

        slip_val = float(slip_interp(rpm_lock))
        if slip_val > slip_lim:
            continue

        in_plateau = PLATEAU_LO <= rpm_lock <= PLATEAU_HI
        in_hump = HUMP_LO <= rpm_lock <= HUMP_HI
        in_dip = DIP_LO <= rpm_lock <= DIP_HI

        zone_pen = 0.0
        if in_dip:
            zone_pen = 800.0
        elif in_plateau or in_hump:
            zone_pen = -50.0
        else:
            zone_pen = 150.0 if mode == "comfort" else 80.0

        bias = 0.0
        if mode == "comfort":
            bias = 0.003 * rpm_lock
        else:
            bias = -0.002 * rpm_lock

        cost = zone_pen + bias
        if cost < best_cost:
            best_cost = cost
            best_mph = mph

    if np.isnan(best_mph):
        return SENTINEL_APPLY
    return round(float(best_mph), 1)


def enforce_monotonic(row: List[float]) -> List[float]:
    out: List[float] = []
    last = 0.0
    for val in row:
        if val >= SENTINEL_THRESHOLD or np.isnan(val):
            out.append(val)
            continue
        if val < last:
            val = last
        out.append(val)
        last = val
    return out


def build_tables(
    mode: str,
    up_shift_df: pd.DataFrame,
    down_shift_df: pd.DataFrame,
    slip_curves: Dict[int, Callable[[np.ndarray], np.ndarray]],
    curve: TorqueCurve,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    apply_rows = []
    release_rows = []

    for gear in GEARS:
        label_apply = f"{gear}rd Apply" if gear == 3 else f"{gear}th Apply"
        label_release = label_apply.replace("Apply", "Release")
        apply_vals: List[float] = []
        release_vals: List[float] = []

        for tps in TPS_AXIS:
            # band from shift tables
            # down mph row gear -> gear-1
            if gear > 1:
                down_label = f"{gear} -> {gear-1} Shift"
                down_mph = shift_value(down_shift_df, down_label, tps)
            else:
                down_mph = 0.0

            if gear < 6:
                up_label = f"{gear} -> {gear+1} Shift"
                up_mph = shift_value(up_shift_df, up_label, tps)
            else:
                up_mph = 150.0

            margin_down = 3.0 if mode == "comfort" else 2.0
            margin_up = 3.0 if mode == "comfort" else 2.0

            mph_min = down_mph + margin_down
            mph_max = up_mph - margin_up

            if mph_max <= mph_min:
                apply_vals.append(SENTINEL_APPLY if tps >= 81 else np.nan)
                release_vals.append(SENTINEL_RELEASE if tps >= 81 else np.nan)
                continue

            apply_mph = select_apply_mph(
                mode,
                gear,
                tps,
                mph_min,
                mph_max,
                curve,
                slip_curves[gear],
            )

            if apply_mph >= SENTINEL_THRESHOLD:
                apply_vals.append(SENTINEL_APPLY)
                release_vals.append(SENTINEL_RELEASE)
                continue

            gap = gap_for_release(mode, tps)
            rel_mph = max(0.0, apply_mph - gap)
            if rel_mph >= apply_mph:
                rel_mph = apply_mph - 0.1
            apply_vals.append(apply_mph)
            release_vals.append(round(float(rel_mph), 1))

        apply_vals = enforce_monotonic(apply_vals)
        release_vals = enforce_monotonic(release_vals)

        apply_rows.append({"mph": label_apply, **{str(t): apply_vals[i] for i, t in enumerate(TPS_AXIS)}})
        release_rows.append({"mph": label_release, **{str(t): release_vals[i] for i, t in enumerate(TPS_AXIS)}})

    cols = ["mph"] + [str(t) for t in TPS_AXIS]
    return pd.DataFrame(apply_rows, columns=cols), pd.DataFrame(release_rows, columns=cols)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")
    print(f"[INFO] Wrote {path}")


def main() -> None:
    curve = load_torque_curve(prefer_mech=True)
    print(f"[INFO] Loaded torque curve. Peak torque={curve.T_peak:.1f} at {curve.rpm_tq_peak:.0f} rpm")

    shift_dir = Path("newlogs") / "output" / "01_tables" / "shift"
    tcc_out_dir = Path("newlogs") / "output" / "01_tables" / "tcc"
    slip_dir = Path("newlogs") / "output" / "01_tables" / "tcc_slip"

    slip_curves = {
        g: load_slip_curve(slip_dir / f"TCC_SLIP_TABLE__GEAR{g}__EC3_FROM_LOGS.tsv")
        for g in GEARS
    }

    for mode in ("comfort", "performance"):
        suffix = "COMFORT_TORQAWARE" if mode == "comfort" else "PERF_TORQAWARE"
        up_path = shift_dir / f"SHIFT_TABLES__UP__Throttle17__{suffix}.tsv"
        down_path = shift_dir / f"SHIFT_TABLES__DOWN__Throttle17__{suffix}.tsv"

        if not up_path.exists() or not down_path.exists():
            raise FileNotFoundError(f"Missing shift tables for mode={mode}: {up_path} / {down_path}")

        up_df = load_shift_table(up_path)
        down_df = load_shift_table(down_path)

        apply_df, release_df = build_tables(mode, up_df, down_df, slip_curves, curve)

        write_table(apply_df, tcc_out_dir / f"TCC_APPLY__Throttle17__{suffix}.tsv")
        write_table(release_df, tcc_out_dir / f"TCC_RELEASE__Throttle17__{suffix}.tsv")


if __name__ == "__main__":
    main()
