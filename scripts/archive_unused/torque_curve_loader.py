#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class TorqueCurve:
    rpm: np.ndarray
    torque: np.ndarray
    hp: np.ndarray
    T_peak: float
    rpm_tq_peak: float
    HP_peak: float
    rpm_hp_peak: float
    interp: Callable[[np.ndarray], np.ndarray]


def _load_curve(path: Path, torque_cols: list[str]) -> Optional[TorqueCurve]:
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    if "rpm_bin_center" not in df.columns:
        return None
    rpm = df["rpm_bin_center"].to_numpy(dtype=float)

    torque_col = None
    for col in torque_cols:
        if col in df.columns:
            torque_col = col
            break
    if torque_col is None:
        return None

    torque = df[torque_col].to_numpy(dtype=float)
    if len(rpm) < 4:
        return None

    hp = torque * rpm / 5252.0

    T_peak_idx = int(np.nanargmax(torque))
    HP_peak_idx = int(np.nanargmax(hp))

    T_peak = float(torque[T_peak_idx])
    rpm_tq_peak = float(rpm[T_peak_idx])
    HP_peak = float(hp[HP_peak_idx])
    rpm_hp_peak = float(rpm[HP_peak_idx])

    def interp(x: np.ndarray | float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        x_clamped = np.clip(x_arr, rpm.min(), rpm.max())
        return np.interp(x_clamped, rpm, torque)

    return TorqueCurve(
        rpm=rpm,
        torque=torque,
        hp=hp,
        T_peak=T_peak,
        rpm_tq_peak=rpm_tq_peak,
        HP_peak=HP_peak,
        rpm_hp_peak=rpm_hp_peak,
        interp=interp,
    )


def load_torque_curve(prefer_mech: bool = True) -> TorqueCurve:
    root = Path(".").resolve()
    mech_path = root / "newlogs" / "output" / "02_passes" / "TORQUE_MECH" / "TORQUE_CURVE__MECH_WOT.tsv"
    ecu_path = root / "newlogs" / "output" / "02_passes" / "TORQUE" / "TORQUE_CURVE__GLOBAL_WOT.tsv"

    mech = _load_curve(mech_path, ["torque_mech_roadload", "torque_mech_raw", "torque_ecu"])
    ecu = _load_curve(ecu_path, ["torque_ecu", "torque_abs", "torque"])

    if prefer_mech and mech is not None:
        return mech
    if ecu is not None:
        return ecu
    if mech is not None:
        return mech
    raise RuntimeError("No usable torque curve found (MECH or ECU).")

