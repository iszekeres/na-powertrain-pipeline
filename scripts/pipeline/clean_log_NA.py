#!/usr/bin/env python3
"""
Minimal FULL-only cleaner for NA Trans logs.

Usage:
  python -m scripts.pipeline.clean_log_NA --in-file <raw_csv> --out-file <clean_csv>

The script enforces required HP Tuners columns, derives canonical fields, and
writes the enhanced CSV without dropping anything.
"""

import argparse
import sys

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "Time",
    "Vehicle Speed",
    "Engine RPM",
    "Throttle Position (%)",
    "Trans Current Gear",
    "Trans Commanded Gear",
    "Trans Input Shaft RPM",
    "Trans Output Shaft RPM",
    "Brake Pressure",
    "Engine Coolant Temp",
    "Trans Fluid Temp",
]

OPTIONAL_COLUMNS = ["Accelerator Pedal Position"]


def load_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False)


def require_columns(df, path):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        available = ", ".join(df.columns.tolist())
        print(f"[ERROR] Missing required columns in {path}: {missing}")
        print(f"[ERROR] Available columns: {available}")
        sys.exit(1)


def to_float(series):
    return pd.to_numeric(series, errors="coerce")


def canonical_gear(series):
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    rounded = np.floor(numeric + 0.5)
    clamped = np.clip(rounded, 0, 6)
    mapped = pd.Series(clamped, index=series.index)
    mapped = mapped.fillna(method="ffill").fillna(0).astype(int)
    return mapped


def build_canonical(df):
    df["time_s"] = to_float(df["Time"])
    df["speed_mph"] = to_float(df["Vehicle Speed"])
    df["engine_rpm"] = to_float(df["Engine RPM"])
    df["throttle_pct"] = to_float(df["Throttle Position (%)"])

    if "Accelerator Pedal Position" in df.columns:
        df["pedal_pct"] = to_float(df["Accelerator Pedal Position"])
    else:
        df["pedal_pct"] = float("nan")

    df["ect_c"] = to_float(df["Engine Coolant Temp"])
    df["tft_c"] = to_float(df["Trans Fluid Temp"])
    df["brake_pressure_kpa"] = to_float(df["Brake Pressure"])
    df["brake"] = (df["brake_pressure_kpa"] >= 15).astype(int)
    df["turbine_rpm"] = to_float(df["Trans Input Shaft RPM"])
    df["output_rpm"] = to_float(df["Trans Output Shaft RPM"])

    df["gear_actual__canon"] = canonical_gear(df["Trans Current Gear"])
    df["gear_cmd__canon"] = canonical_gear(df["Trans Commanded Gear"])

    df["tcc_slip_rpm"] = df["engine_rpm"] - df["turbine_rpm"]
    df["tcc_slip_fused"] = df["tcc_slip_rpm"]
    df["tcc_locked_built"] = (df["tcc_slip_rpm"].abs() <= 50).astype(int)

    return df


def main():
    parser = argparse.ArgumentParser(description="Clean NA Trans logs into a FULL file.")
    parser.add_argument("--in-file", required=True, help="Path to raw HP Tuners CSV.")
    parser.add_argument("--out-file", required=True, help="Destination for __clean_full CSV.")
    args = parser.parse_args()

    in_path = args.in_file
    out_path = args.out_file

    df = load_csv(in_path)
    require_columns(df, in_path)
    df = build_canonical(df)

    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[clean_log_NA] Cleaning {in_path} -> {out_path}")


if __name__ == "__main__":
    main()
