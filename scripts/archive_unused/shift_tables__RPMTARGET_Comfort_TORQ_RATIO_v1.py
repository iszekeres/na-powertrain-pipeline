#!/usr/bin/env python
"""
shift_tables__RPMTARGET_Comfort_TORQ_RATIO_v1.py

Build a Comfort-only RPMTARGET schedule derived from the torque curve by
comparing wheel-torque ratios ("Strategy C").

Outputs:
  newlogs/output/01_tables/shift/SHIFT_TABLES__UP__Throttle17__COMFORT_RPMTARGET__TORQ_RATIO.tsv
  newlogs/output/01_tables/shift/SHIFT_TABLES__DOWN__Throttle17__COMFORT_RPMTARGET__TORQ_RATIO.tsv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Vehicle constants
GEAR_RATIOS = {
    1: 4.03,
    2: 2.36,
    3: 1.53,
    4: 1.15,
    5: 0.85,
    6: 0.67,
}
FINAL_DRIVE = 3.08
TIRE_DIAMETER_IN = 32.5
RPM_MAX = 6200.0
TPS_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]
GAP_MPH = 1.0

TORQUE_MECH_PATH = Path("newlogs/output/02_passes/TORQUE_MECH/TORQUE_CURVE__MECH_WOT.tsv")
ECU_TORQUE_PATH = Path("newlogs/output/02_passes/TORQUE/TORQUE_CURVE__GLOBAL_WOT.tsv")

RATIO_THRESHOLDS = [
    (0, 0.30),
    (6, 0.35),
    (12, 0.40),
    (19, 0.45),
    (25, 0.50),
    (31, 0.55),
    (37, 0.60),
    (44, 0.65),
    (50, 0.70),
    (56, 0.75),
    (62, 0.78),
    (69, 0.80),
    (75, 0.82),
    (87, 0.86),
    (100, 0.90),
]

DOWN_RPM_TARGETS = [
    (6, 2200.0),
    (12, 2400.0),
    (19, 2600.0),
    (25, 2800.0),
    (31, 3000.0),
    (37, 3200.0),
    (44, 3400.0),
    (50, 3500.0),
    (56, 3600.0),
    (62, 3700.0),
    (69, 3800.0),
    (75, 3900.0),
    (100, 4000.0),
]


def mph_from_rpm(rpm: float, gear: int) -> float:
    if gear not in GEAR_RATIOS:
        return 0.0
    return (rpm * TIRE_DIAMETER_IN) / (GEAR_RATIOS[gear] * FINAL_DRIVE * 336.0)


def rpm_from_mph(mph: float, gear: int) -> float:
    if gear not in GEAR_RATIOS:
        return 0.0
    return (mph * GEAR_RATIOS[gear] * FINAL_DRIVE * 336.0) / TIRE_DIAMETER_IN


def load_torque_curve() -> Tuple[np.ndarray, np.ndarray]:
    def _pick(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        cols = {c.lower(): c for c in df.columns}
        for candidate in candidates:
            key = candidate.lower()
            if key in cols:
                return cols[key]
        return None

    for path in (TORQUE_MECH_PATH, ECU_TORQUE_PATH):
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        rpm_col = "rpm_bin_center" if "rpm_bin_center" in df.columns else df.columns[0]
        col = _pick(df, ["torque_mech_roadload", "torque_mech_raw", "torque_ecu", "torque_abs", "torque"])
        if col is None:
            continue
        rpm_vals = df[rpm_col].to_numpy(dtype=float)
        torque_vals = df[col].to_numpy(dtype=float)
        valid = np.isfinite(rpm_vals) & np.isfinite(torque_vals)
        rpm_vals = rpm_vals[valid]
        torque_vals = torque_vals[valid]
        if len(rpm_vals) == 0:
            continue
        order = np.argsort(rpm_vals)
        rpm_vals = rpm_vals[order]
        torque_vals = torque_vals[order]
        print(f"[INFO] Loaded torque curve {path.name} using column '{col}'")
        return rpm_vals, torque_vals

    raise FileNotFoundError("No torque curve found (mechanical or ECU).")


def torque_interpolator(rpm_vals: np.ndarray, torque_vals: np.ndarray):
    def _interp(target_rpm: float) -> float:
        if target_rpm <= rpm_vals[0]:
            return torque_vals[0]
        if target_rpm >= rpm_vals[-1]:
            return torque_vals[-1]
        idx = np.searchsorted(rpm_vals, target_rpm) - 1
        idx = max(0, min(idx, len(rpm_vals) - 2))
        x0, x1 = rpm_vals[idx], rpm_vals[idx + 1]
        y0, y1 = torque_vals[idx], torque_vals[idx + 1]
        if x1 == x0:
            return y0
        frac = (target_rpm - x0) / (x1 - x0)
        return y0 + frac * (y1 - y0)

    return _interp


def ratio_min_for_tps(tps: int) -> float:
    for threshold, ratio in RATIO_THRESHOLDS:
        if tps <= threshold:
            return ratio
    return RATIO_THRESHOLDS[-1][1]


def build_up_table(torque_func) -> Dict[Tuple[int, int], float]:
    up_table: Dict[Tuple[int, int], float] = {}
    for gear in range(1, 6):
        gear_ratio = GEAR_RATIOS[gear]
        next_ratio = GEAR_RATIOS.get(gear + 1, gear_ratio)
        max_mph = mph_from_rpm(RPM_MAX, gear)
        mph_grid = np.linspace(5.0, max_mph, 300)
        for tps in TPS_AXIS:
            min_ratio = ratio_min_for_tps(tps)
            chosen = mph_grid[-1]
            for mph in mph_grid:
                rpm_before = rpm_from_mph(mph, gear)
                rpm_after = rpm_from_mph(mph, gear + 1)
                if rpm_before > RPM_MAX:
                    break
                tb = torque_func(rpm_before)
                ta = torque_func(rpm_after)
                tw_before = tb * gear_ratio * FINAL_DRIVE
                tw_after = ta * next_ratio * FINAL_DRIVE
                if tw_before <= 0:
                    continue
                ratio = tw_after / tw_before
                if ratio >= min_ratio:
                    chosen = mph
                    break
            key = (gear, tps)
            up_table[key] = round(chosen, 1)
    # enforce monotonicity per gear
    for gear in range(1, 6):
        last = 0.0
        for tps in TPS_AXIS:
            key = (gear, tps)
            val = up_table.get(key, 0.0)
            if val < last:
                up_table[key] = last
            else:
                last = val
    return up_table


def target_down_rpm(tps: int) -> float:
    for threshold, rpm in DOWN_RPM_TARGETS:
        if tps <= threshold:
            return rpm
    return DOWN_RPM_TARGETS[-1][1]


def build_down_table(up_table: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    down_table: Dict[Tuple[int, int], float] = {}
    for gear in range(2, 7):
        lower_gear = gear - 1
        for tps in TPS_AXIS:
            down_label = (gear, tps)
            target_rpm = target_down_rpm(tps)
            mph_hump = mph_from_rpm(target_rpm, lower_gear)
            up_mph = up_table.get((lower_gear, tps), mph_hump)
            max_down = max(0.0, up_mph - GAP_MPH)
            down_val = min(max_down, mph_hump)
            down_val = round(down_val, 1)
            down_table[down_label] = down_val
    # enforce monotonic vs TPS per gear pair
    for gear in range(2, 7):
        last = 0.0
        for tps in TPS_AXIS:
            key = (gear, tps)
            val = down_table.get(key, 0.0)
            if val < last:
                down_table[key] = last
            else:
                last = val
    return down_table


def build_df(up_table: Dict[Tuple[int, int], float]) -> pd.DataFrame:
    rows = []
    labels = {
        1: "1 -> 2 Shift",
        2: "2 -> 3 Shift",
        3: "3 -> 4 Shift",
        4: "4 -> 5 Shift",
        5: "5 -> 6 Shift",
    }
    for gear in range(1, 6):
        label = labels[gear]
        row = {"mph": label}
        for tps in TPS_AXIS:
            row[str(tps)] = up_table.get((gear, tps), 0.0)
        rows.append(row)
    return pd.DataFrame(rows, columns=["mph"] + [str(tps) for tps in TPS_AXIS])


def build_down_df(down_table: Dict[Tuple[int, int], float]) -> pd.DataFrame:
    rows = []
    labels = {
        2: "2 -> 1 Shift",
        3: "3 -> 2 Shift",
        4: "4 -> 3 Shift",
        5: "5 -> 4 Shift",
        6: "6 -> 5 Shift",
    }
    for gear in range(2, 7):
        label = labels[gear]
        row = {"mph": label}
        for tps in TPS_AXIS:
            row[str(tps)] = down_table.get((gear, tps), 0.0)
        rows.append(row)
    return pd.DataFrame(rows, columns=["mph"] + [str(tps) for tps in TPS_AXIS])


def main() -> None:
    rpm_vals, torque_vals = load_torque_curve()
    torque_func = torque_interpolator(rpm_vals, torque_vals)
    up = build_up_table(torque_func)
    down = build_down_table(up)
    up_df = build_df(up)
    down_df = build_down_df(down)
    out_dir = Path("newlogs/output/01_tables/shift")
    out_dir.mkdir(parents=True, exist_ok=True)
    up_path = out_dir / "SHIFT_TABLES__UP__Throttle17__COMFORT_RPMTARGET__TORQ_RATIO.tsv"
    down_path = out_dir / "SHIFT_TABLES__DOWN__Throttle17__COMFORT_RPMTARGET__TORQ_RATIO.tsv"
    up_df.to_csv(up_path, sep="\\t", index=False, encoding="utf-8")
    down_df.to_csv(down_path, sep="\\t", index=False, encoding="utf-8")
    print(f"[INFO] Wrote UP table to {up_path}")
    print(f"[INFO] Wrote DOWN table to {down_path}")


if __name__ == "__main__":
    main()
*** End Patch***/
