#!/usr/bin/env python3
"""Generate neutral shift and TCC tables from torque data and an optional log."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

THROTTLE_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]
TPS_COLUMNS = [str(int(v)) for v in THROTTLE_AXIS]

SHIFT_UP_LABELS = [f"{g} -> {g + 1} Shift" for g in range(1, 6)]
SHIFT_DOWN_LABELS = [f"{g + 1} -> {g} Shift" for g in range(1, 6)]
TCC_APPLY_LABELS = [f"{g}rd Apply" if g == 3 else f"{g}th Apply" if g != 3 else f"{g}rd Apply" for g in range(3, 7)]
TCC_RELEASE_LABELS = [f"{g}rd Release" if g == 3 else f"{g}th Release" if g != 3 else f"{g}rd Release" for g in range(3, 7)]

TCC_LOCK_SENTINEL = 318.0
TCC_HIGH_SPEED_CAP = 200.0
TCC_HIGH_TPS_SENTINEL = 75.0

GEAR_RATIOS = {1: 4.03, 2: 2.36, 3: 1.53, 4: 1.15, 5: 0.85, 6: 0.67}
FINAL_DRIVE = 3.08
TIRE_DIAMETER_IN = 32.5
RPM_LIMIT = 6600
RPM_SHIFT_MAX = 6200
RPM_SHIFT_MIN = 2500
RPM_ANALYSIS_MAX = 6400
RPM_GRID_STEP = 50

LOW_RPM_TARGET = {1: 1700, 2: 1600, 3: 1500, 4: 1500, 5: 1500}

TORQUE_CURVE_DIR = REPO_ROOT / "newlogs" / "output" / "02_passes" / "TORQUE"
CONFIG = {
    "global_curve": TORQUE_CURVE_DIR / "TORQUE_CURVE__GLOBAL_WOT.tsv",
    "mode_lock_curve": TORQUE_CURVE_DIR / "TORQUE_CURVES__MODE_LOCK_WOT.tsv",
    "gear_curve": TORQUE_CURVE_DIR / "TORQUE_CURVES__MODE_LOCK_GEAR_WOT.tsv",
    "torque_log": Path("CHANGE_THIS_TO_MY_TORQUE_TEST_LOG.csv"),
}

OUTPUT_PATHS = {
    "shift_up": REPO_ROOT / "SHIFT_TABLES__UP__Throttle17__NEUTRAL__LOGFIRST_v1.tsv",
    "shift_down": REPO_ROOT / "SHIFT_TABLES__DOWN__Throttle17__NEUTRAL__LOGFIRST_v1.tsv",
    "tcc_apply": REPO_ROOT / "TCC_APPLY__Throttle17__NEUTRAL__LOGFIRST_v1.tsv",
    "tcc_release": REPO_ROOT / "TCC_RELEASE__Throttle17__NEUTRAL__LOGFIRST_v1.tsv",
}

LOG_TIME_COLUMNS = ["time_s", "offset", "Time", "time"]
LOG_GEAR_CANDIDATES = ["gear_actual__canon", "gear_actual", "gear"]
LOG_RPM_CANDIDATES = ["engine_rpm", "rpm"]
LOG_THROTTLE_CANDIDATES = ["throttle_pct", "pedal_pct", "throttle", "pedal_position"]
LOG_BRAKE_CANDIDATES = ["brake", "brake_pct"]
LOG_THROTTLE_MIN = 80.0


def normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def find_column_name(
    df: pd.DataFrame, candidates: Sequence[str], required: bool = True, *, context: str = "column"
) -> Optional[str]:
    normalized = {normalize_name(col): col for col in df.columns}
    candidate_keys = [normalize_name(cand) for cand in candidates]
    for key in candidate_keys:
        if key in normalized:
            return normalized[key]
    for key in candidate_keys:
        for col_key, real_name in normalized.items():
            if key in col_key or col_key in key:
                return real_name
    if required:
        raise ValueError(f"Missing {context}; expected one of {candidates}")
    return None


class TorqueCurve:
    def __init__(self, rpm: np.ndarray, torque: np.ndarray) -> None:
        order = np.argsort(rpm)
        rpm_sorted = rpm[order]
        torque_sorted = torque[order]
        self.rpm = rpm_sorted
        self.torque = torque_sorted

    def torque_at(self, rpm_values: np.ndarray) -> np.ndarray:
        rpm_values = np.asarray(rpm_values, dtype=float)
        if rpm_values.ndim == 0:
            rpm_values = rpm_values[None]
        return np.interp(
            rpm_values,
            self.rpm,
            self.torque,
            left=self.torque[0],
            right=self.torque[-1],
        )


def read_tabular(path: Path, sep: str = "\t") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected file does not exist: {path}")
    return pd.read_csv(path, sep=sep)


def build_curve_from_df(df: pd.DataFrame) -> TorqueCurve:
    rpm_col = find_column_name(df, LOG_RPM_CANDIDATES, context="torque RPM column")
    torque_col = find_column_name(
        df,
        ["torque", "torque_ftlb", "torque_lbft", "torque_nm"],
        context="torque column",
    )
    selection = df[[rpm_col, torque_col]].copy()
    selection[rpm_col] = pd.to_numeric(selection[rpm_col], errors="coerce")
    selection[torque_col] = pd.to_numeric(selection[torque_col], errors="coerce")
    selection = selection.dropna(subset=[rpm_col, torque_col])
    if len(selection) < 2:
        raise ValueError("Torque curve needs at least two valid points")
    grouped = selection.groupby(rpm_col, as_index=False).mean()
    rpm_values = grouped[rpm_col].to_numpy(dtype=float)
    torque_values = grouped[torque_col].to_numpy(dtype=float)
    return TorqueCurve(rpm_values, torque_values)


def load_global_curve() -> TorqueCurve:
    return build_curve_from_df(read_tabular(CONFIG["global_curve"]))


def load_gear_curves(base_curve: TorqueCurve) -> Dict[int, TorqueCurve]:
    path = CONFIG["gear_curve"]
    if not path.exists():
        return {}
    df = read_tabular(path)
    gear_col = find_column_name(df, LOG_GEAR_CANDIDATES, context="gear column")
    rpm_col = find_column_name(df, LOG_RPM_CANDIDATES, context="torque RPM column")
    torque_col = find_column_name(
        df,
        ["torque", "torque_ftlb", "torque_lbft", "torque_nm"],
        context="torque column",
    )
    gear_curves: Dict[int, TorqueCurve] = {}
    for gear_value, group in df.groupby(gear_col):
        gear_number = pd.to_numeric(gear_value, errors="coerce")
        if pd.isna(gear_number):
            continue
        gear_number = int(gear_number)
        if gear_number not in GEAR_RATIOS:
            continue
        selection = group[[rpm_col, torque_col]].copy()
        selection[rpm_col] = pd.to_numeric(selection[rpm_col], errors="coerce")
        selection[torque_col] = pd.to_numeric(selection[torque_col], errors="coerce")
        selection = selection.dropna(subset=[rpm_col, torque_col])
        if len(selection) < 2:
            continue
        grouped = selection.groupby(rpm_col, as_index=False).mean()
        rpm_values = grouped[rpm_col].to_numpy(dtype=float)
        torque_values = grouped[torque_col].to_numpy(dtype=float)
        gear_curves[gear_number] = TorqueCurve(rpm_values, torque_values)
    return gear_curves


def rpm_to_mph(rpm: float, gear: int) -> float:
    ratio = GEAR_RATIOS[gear]
    return rpm * TIRE_DIAMETER_IN / (ratio * FINAL_DRIVE * 336.0)


def tps_effort_fraction(tps_fraction: float) -> float:
    raw = 1.0 / (1.0 + np.exp(-7.0 * (tps_fraction - 0.55)))
    return float(np.clip(raw, 0.0, 1.0))


def enforce_non_decreasing(values: Sequence[float]) -> List[float]:
    out: List[float] = []
    prev = -float("inf")
    for value in values:
        current = value
        if current < prev:
            current = prev
        out.append(current)
        prev = current
    return out


def build_shift_points(
    global_curve: TorqueCurve, gear_curves: Dict[int, TorqueCurve]
) -> Dict[int, Dict[str, float]]:
    rpm_grid = np.arange(1500, RPM_ANALYSIS_MAX + RPM_GRID_STEP, RPM_GRID_STEP)
    window_mask = (rpm_grid >= RPM_SHIFT_MIN) & (rpm_grid <= RPM_ANALYSIS_MAX)
    rpm_filtered = rpm_grid[window_mask]
    shift_points: Dict[int, Dict[str, float]] = {}
    for gear in range(1, 6):
        current_curve = gear_curves.get(gear, global_curve)
        next_curve = gear_curves.get(gear + 1, global_curve)
        mph_grid = rpm_to_mph(rpm_filtered, gear)
        rpm_next = (
            mph_grid * GEAR_RATIOS[gear + 1] * FINAL_DRIVE * 336.0 / TIRE_DIAMETER_IN
        )
        torque_current = current_curve.torque_at(rpm_filtered)
        torque_next = next_curve.torque_at(rpm_next)
        wheel_current = torque_current * GEAR_RATIOS[gear] * FINAL_DRIVE
        wheel_next = torque_next * GEAR_RATIOS[gear + 1] * FINAL_DRIVE
        delta = np.abs(wheel_current - wheel_next)
        best_index = int(np.nanargmin(delta))
        best_rpm = float(rpm_filtered[best_index])
        if best_rpm > RPM_SHIFT_MAX:
            best_rpm = float(RPM_SHIFT_MAX)
        mph_value = rpm_to_mph(best_rpm, gear)
        shift_points[gear] = {"rpm": best_rpm, "mph": mph_value}
    return shift_points


def rpm_to_mph(rpm: Iterable[float], gear: int) -> np.ndarray:
    ratio = GEAR_RATIOS[gear]
    arr = np.asarray(list(rpm), dtype=float)
    return arr * TIRE_DIAMETER_IN / (ratio * FINAL_DRIVE * 336.0)


def build_shift_up_table(shift_points: Dict[int, Dict[str, float]]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for gear in range(1, 6):
        row_label = SHIFT_UP_LABELS[gear - 1]
        rpm_min = max(LOW_RPM_TARGET[gear], 1400.0)
        wot_rpm = min(shift_points[gear]["rpm"], RPM_SHIFT_MAX)
        delta = max(wot_rpm - rpm_min, 0.0)
        values = []
        for tps in THROTTLE_AXIS:
            tps_frac = tps / 100.0
            target_rpm = rpm_min + tps_effort_fraction(tps_frac) * delta
            target_rpm = float(np.clip(target_rpm, rpm_min, RPM_SHIFT_MAX))
            mph_value = rpm_to_mph(target_rpm, gear)
            values.append(mph_value)
        values = enforce_non_decreasing(values)
        values = enforce_non_decreasing([round(v, 1) for v in values])
        row = {"mph": row_label}
        for col, val in zip(TPS_COLUMNS, values):
            row[col] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    return df[["mph"] + TPS_COLUMNS]


def hysteresis_gap(tps: float) -> float:
    if tps <= 12:
        return 3.0 + (tps / 12.0) * 0.3
    if tps <= 50:
        return 3.5 + ((tps - 12) / 38.0) * 1.5
    return 5.0 + ((tps - 50) / 50.0) * 1.5


def build_shift_down_table(shift_up: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for index, label in enumerate(SHIFT_DOWN_LABELS):
        up_row = shift_up.iloc[index]
        values = []
        for col in TPS_COLUMNS:
            up_value = float(up_row[col])
            gap = hysteresis_gap(float(col))
            down_value = up_value - gap
            down_value = max(0.0, down_value)
            if down_value > up_value - 1.0:
                down_value = max(0.0, up_value - 1.0)
            values.append(down_value)
        values = enforce_non_decreasing(values)
        values = enforce_non_decreasing([round(v, 1) for v in values])
        row = {"mph": label}
        for col, val in zip(TPS_COLUMNS, values):
            row[col] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    return df[["mph"] + TPS_COLUMNS]


def safe_read_log_table() -> Optional[pd.DataFrame]:
    path = CONFIG["torque_log"]
    if not path.exists():
        print(f"Torque log missing at {path}; skipping WOT log comparison.")
        return None
    df = pd.read_csv(path)
    time_col = find_column_name(df, LOG_TIME_COLUMNS, context="time column")
    df = df.sort_values(by=time_col)
    return df


def detect_log_shifts(df: pd.DataFrame) -> Dict[int, float]:
    gear_col = find_column_name(df, LOG_GEAR_CANDIDATES, context="gear column")
    rpm_col = find_column_name(df, LOG_RPM_CANDIDATES, context="engine rpm column")
    throttle_col = find_column_name(
        df, LOG_THROTTLE_CANDIDATES, context="throttle column"
    )
    brake_col = find_column_name(df, LOG_BRAKE_CANDIDATES, required=False)
    df = df.dropna(subset=[gear_col, rpm_col, throttle_col])
    df[gear_col] = pd.to_numeric(df[gear_col], errors="coerce")
    df[rpm_col] = pd.to_numeric(df[rpm_col], errors="coerce")
    df[throttle_col] = pd.to_numeric(df[throttle_col], errors="coerce")
    if brake_col:
        df[brake_col] = pd.to_numeric(df[brake_col], errors="coerce")
    df = df.dropna(subset=[gear_col, rpm_col, throttle_col])
    df = df.reset_index(drop=True)
    shift_bins: Dict[int, List[float]] = {}
    for prev, curr in zip(df.itertuples(index=False), df.itertuples(index=False)):
        pass


