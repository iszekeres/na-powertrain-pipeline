#!/usr/bin/env python
"""
shift_tables__RPMTARGET_TORQ_AWARE_v1.py

Ground-up torque-aware RPMTARGET shift tables for Comfort and Performance.
Outputs:
  newlogs/output/01_tables/shift/SHIFT_TABLES__UP__Throttle17__COMFORT_TORQAWARE.tsv
  newlogs/output/01_tables/shift/SHIFT_TABLES__DOWN__Throttle17__COMFORT_TORQAWARE.tsv
  newlogs/output/01_tables/shift/SHIFT_TABLES__UP__Throttle17__PERF_TORQAWARE.tsv
  newlogs/output/01_tables/shift/SHIFT_TABLES__DOWN__Throttle17__PERF_TORQAWARE.tsv

Strategy:
  - Use the mechanical (preferred) or ECU torque curve.
  - UP shifts place the post-shift RPM in plateau/hump (1900–2900) and avoid
    the 3000–3300 dip with a cost function.
  - DOWN shifts land on a torque-friendly RPM target with hysteresis to the UP table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from torque_curve_loader import load_torque_curve, TorqueCurve

# Drivetrain constants
GEAR_RATIOS: Dict[int, float] = {1: 4.03, 2: 2.36, 3: 1.53, 4: 1.15, 5: 0.85, 6: 0.67}
FINAL_DRIVE = 3.08
TIRE_DIAMETER_IN = 32.5

TPS_AXIS: List[int] = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]

RPM_REDLINE = 6600.0
RPM_CAP_WOT = 6200.0
RPM_MIN_LANDING = 1500.0

# Zones
PLATEAU_LO = 1900.0
PLATEAU_HI = 2500.0
HUMP_LO = 2500.0
HUMP_HI = 2900.0
DIP_LO = 3000.0
DIP_HI = 3300.0

MPH_MIN_PER_GEAR = {1: 5.0, 2: 12.0, 3: 20.0, 4: 30.0, 5: 40.0}


def mph_from_rpm(rpm: float | np.ndarray, gear: int) -> float | np.ndarray:
    gr = GEAR_RATIOS[gear]
    return (np.asarray(rpm) * TIRE_DIAMETER_IN) / (gr * FINAL_DRIVE * 336.0)


def rpm_from_mph(mph: float | np.ndarray, gear: int) -> float | np.ndarray:
    gr = GEAR_RATIOS[gear]
    return (np.asarray(mph) * gr * FINAL_DRIVE * 336.0) / TIRE_DIAMETER_IN


# ---- Targets ----

def target_up_rpm(mode: str, tps: int, curve: TorqueCurve) -> float:
    mode = mode.lower()
    if mode == "comfort":
        if tps <= 12:
            return 2200.0
        if tps <= 25:
            return 2300.0
        if tps <= 37:
            return 2450.0
        if tps <= 56:
            return 2600.0
        if tps <= 75:
            return 2750.0
        return 2850.0  # keep under the dip

    # performance
    rp = curve.rpm_tq_peak
    rh = curve.rpm_hp_peak
    if tps <= 25:
        return 2400.0
    if tps <= 50:
        return 2600.0
    if tps <= 75:
        return 2800.0
    candidate = 0.5 * rp + 0.5 * rh
    if DIP_LO <= candidate <= DIP_HI:
        candidate = DIP_HI + 100.0
    return float(np.clip(candidate, 2600.0, RPM_CAP_WOT))


def target_down_rpm(mode: str, tps: int, curve: TorqueCurve) -> float:
    mode = mode.lower()
    if mode == "comfort":
        if tps <= 12:
            return 2200.0
        if tps <= 25:
            return 2350.0
        if tps <= 37:
            return 2500.0
        if tps <= 56:
            return 2650.0
        if tps <= 75:
            return 2800.0
        return 2850.0

    rp = curve.rpm_tq_peak
    if tps <= 25:
        return 2400.0
    if tps <= 50:
        return 2600.0
    if tps <= 75:
        return 2800.0
    candidate = rp
    if DIP_LO <= candidate <= DIP_HI:
        candidate = HUMP_HI
    return float(np.clip(candidate, 2600.0, RPM_CAP_WOT))


# ---- Cost helpers ----

def zone_penalty(rpm_after: float, mode: str) -> float:
    in_plateau_hump = (PLATEAU_LO <= rpm_after <= HUMP_HI)
    in_dip = (DIP_LO <= rpm_after <= DIP_HI)
    if in_dip:
        return 800.0
    if in_plateau_hump:
        return -50.0  # reward a bit
    # above hump
    return 150.0 if mode == "comfort" else 80.0


def build_up_table(mode: str, curve: TorqueCurve) -> Dict[Tuple[int, int], float]:
    up_map: Dict[Tuple[int, int], float] = {}
    for gear in range(1, 6):  # 1->2 ... 5->6
        mph_min = MPH_MIN_PER_GEAR.get(gear, 5.0)
        mph_max = float(mph_from_rpm(RPM_CAP_WOT, gear))
        grid = np.linspace(mph_min, mph_max, 250)
        for tps in TPS_AXIS:
            tgt_rpm = target_up_rpm(mode, tps, curve)
            best_mph = grid[-1]
            best_cost = float("inf")
            for mph in grid:
                rpm_before = rpm_from_mph(mph, gear)
                if rpm_before > RPM_CAP_WOT + 20:
                    break
                rpm_after = rpm_from_mph(mph, gear + 1)
                dist = abs(rpm_after - tgt_rpm)
                cost = dist
                cost += zone_penalty(rpm_after, mode)
                if rpm_after < PLATEAU_LO:
                    cost += 300.0
                if rpm_after > HUMP_HI:
                    cost += 150.0 if mode == "comfort" else 50.0
                if mode == "comfort":
                    cost += 0.005 * rpm_before  # dislikes revs
                else:
                    cost -= 0.003 * rpm_before  # likes revs a bit
                if cost < best_cost:
                    best_cost = cost
                    best_mph = mph
            up_map[(gear, tps)] = round(float(best_mph), 1)

        # enforce monotonic vs TPS
        prev = 0.0
        for tps in TPS_AXIS:
            val = up_map[(gear, tps)]
            if val < prev:
                val = prev
                up_map[(gear, tps)] = val
            prev = val
    return up_map


def build_down_table(mode: str, curve: TorqueCurve, up_map: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    down_map: Dict[Tuple[int, int], float] = {}
    down_gap = 1.0 if mode == "comfort" else 0.8
    for gear_hi in range(2, 7):  # 2->1 ... 6->5
        gear_lo = gear_hi - 1
        for tps in TPS_AXIS:
            tgt_rpm = target_down_rpm(mode, tps, curve)
            tgt_rpm = float(np.clip(tgt_rpm, RPM_MIN_LANDING, RPM_CAP_WOT))
            mph_hump = float(mph_from_rpm(tgt_rpm, gear_lo))
            up_mph = up_map[(gear_lo, tps)]  # lo -> hi
            max_down = max(0.0, up_mph - down_gap)
            down_mph = min(mph_hump, max_down)
            down_map[(gear_hi, tps)] = round(down_mph, 1)

        # monotonic vs TPS
        prev = 0.0
        for tps in TPS_AXIS:
            val = down_map[(gear_hi, tps)]
            if val < prev:
                val = prev
                down_map[(gear_hi, tps)] = val
            prev = val
    return down_map


def map_to_df(up_map: Dict[Tuple[int, int], float], down_map: Dict[Tuple[int, int], float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    up_rows = []
    for gear in range(1, 6):
        label = f"{gear} -> {gear+1} Shift"
        row = {"mph": label}
        for tps in TPS_AXIS:
            row[str(tps)] = up_map[(gear, tps)]
        up_rows.append(row)

    down_rows = []
    for gear_hi in range(2, 7):
        label = f"{gear_hi} -> {gear_hi-1} Shift"
        row = {"mph": label}
        for tps in TPS_AXIS:
            row[str(tps)] = down_map[(gear_hi, tps)]
        down_rows.append(row)

    cols = ["mph"] + [str(tps) for tps in TPS_AXIS]
    return pd.DataFrame(up_rows, columns=cols), pd.DataFrame(down_rows, columns=cols)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")
    print(f"[INFO] Wrote {path}")


def print_row_stats(df: pd.DataFrame, label_contains: str) -> None:
    for _, row in df.iterrows():
        label = str(row["mph"])
        if label_contains not in label:
            continue
        vals = [float(v) for v in row[1:]]
        print(f"  {label}: min={min(vals):.1f} max={max(vals):.1f}")


def main() -> None:
    curve = load_torque_curve(prefer_mech=True)
    print(f"[INFO] Loaded torque curve. Peak torque={curve.T_peak:.1f} at {curve.rpm_tq_peak:.0f} rpm; "
          f"Peak HP={curve.HP_peak:.1f} at {curve.rpm_hp_peak:.0f} rpm")

    out_dir = Path("newlogs") / "output" / "01_tables" / "shift"
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in ("comfort", "performance"):
        print(f"[INFO] Building {mode.title()} torque-aware shift tables...")
        up_map = build_up_table(mode, curve)
        down_map = build_down_table(mode, curve, up_map)
        up_df, down_df = map_to_df(up_map, down_map)

        suffix = "COMFORT_TORQAWARE" if mode == "comfort" else "PERF_TORQAWARE"
        up_path = out_dir / f"SHIFT_TABLES__UP__Throttle17__{suffix}.tsv"
        down_path = out_dir / f"SHIFT_TABLES__DOWN__Throttle17__{suffix}.tsv"

        write_table(up_df, up_path)
        write_table(down_df, down_path)

        print("[INFO] Row stats (UP):")
        print_row_stats(up_df, "Shift")
        print("[INFO] Row stats (DOWN):")
        print_row_stats(down_df, "Shift")


if __name__ == "__main__":
    main()
