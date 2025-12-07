#!/usr/bin/env python3
"""
Minimal FULL-only cleaner for NA Trans logs.

Usage:
  python -m scripts.pipeline.clean_log_NA --in-file <raw_csv> --out-file <clean_csv>

The script enforces required HP Tuners columns, derives canonical fields, and
writes the enhanced CSV without dropping anything.
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_SOURCES = {
    "Time": ["Time", "Offset", "time_s"],
    "Vehicle Speed": ["Vehicle Speed (SAE)", "Vehicle Speed"],
    "Engine RPM": ["Engine RPM (SAE)", "Engine RPM"],
    "Throttle Position (%)": ["Throttle Position (%)", "Throttle Position"],
    "Accelerator Pedal Position": ["Accelerator Pedal Position"],
    "Trans Current Gear": [("Trans Current Gear", 0)],
    "Trans Commanded Gear": [("Trans Current Gear", 1)],
    "Trans Input Shaft RPM": ["Trans Input Shaft RPM"],
    "Trans Output Shaft RPM": ["Trans Output Shaft RPM"],
    "Brake Pressure": ["Brake Pressure"],
    "Engine Coolant Temp": ["Engine Coolant Temp (SAE)", "Engine Coolant Temp"],
    "Trans Fluid Temp": ["Trans Fluid Temp"],
}

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

OPTIONAL_COLUMN = "Accelerator Pedal Position"


def read_text(path):
    for encoding in ("utf-8", "latin1"):
        try:
            return Path(path).read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "Unable to decode log file")


def load_csv(path):
    text = read_text(path)
    lines = [line for line in text.splitlines() if line.strip() != ""]
    header_idx = next(
        (idx for idx, line in enumerate(lines) if line.strip().startswith("Offset,")),
        None,
    )
    if header_idx is None:
        raise RuntimeError(f"[clean_log_NA] Could not find header row in {path}")
    csv_text = "\n".join(lines[header_idx:])
    return pd.read_csv(io.StringIO(csv_text), skiprows=[1])


def normalize_col_name(col):
    if "." in col:
        root, suffix = col.rsplit(".", 1)
        if suffix.isdigit():
            return root
    return col


def match_column(df, alias):
    canonical = {col: normalize_col_name(col) for col in df.columns}
    if isinstance(alias, str):
        for col, norm in canonical.items():
            if norm == alias:
                return col
        return None
    name, occurrence = alias
    matches = [col for col, norm in canonical.items() if norm == name]
    return matches[occurrence] if len(matches) > occurrence else None


def resolve_sources(df):
    resolved = {}
    for canonical, aliases in CANONICAL_SOURCES.items():
        resolved[canonical] = None
        for alias in aliases:
            candidate = match_column(df, alias)
            if candidate is not None:
                resolved[canonical] = candidate
                break
    return resolved


def require_columns(sources, _path):
    missing = [col for col in REQUIRED_COLUMNS if sources.get(col) is None]
    if missing:
        available = ", ".join(
            f"{col}->{sources.get(col) or '<missing>'}" for col in REQUIRED_COLUMNS
        )
        print(f"[ERROR] missing required columns: {missing}")
        print(f"[ERROR] Available mapping: {available}")
        sys.exit(1)


def to_float(series):
    return pd.to_numeric(series, errors="coerce")


def canon_gear(series):
    values = pd.to_numeric(series, errors="coerce")
    rounded = np.floor(values + 0.5)
    clamped = np.clip(rounded, 0, 6)
    filled = pd.Series(clamped).ffill().fillna(0).astype(int)
    return filled


def build_canonical(df, sources):
    df["time_s"] = to_float(df[sources["Time"]])
    df["speed_mph"] = to_float(df[sources["Vehicle Speed"]])
    df["engine_rpm"] = to_float(df[sources["Engine RPM"]])
    df["throttle_pct"] = to_float(df[sources["Throttle Position (%)"]])

    pedal_col = sources.get(OPTIONAL_COLUMN)
    if pedal_col:
        df["pedal_pct"] = to_float(df[pedal_col])
    else:
        df["pedal_pct"] = float("nan")

    df["ect_c"] = to_float(df[sources["Engine Coolant Temp"]])
    df["tft_c"] = to_float(df[sources["Trans Fluid Temp"]])
    df["brake_pressure_kpa"] = to_float(df[sources["Brake Pressure"]])
    df["brake"] = (df["brake_pressure_kpa"] >= 15).astype(int)
    df["turbine_rpm"] = to_float(df[sources["Trans Input Shaft RPM"]])
    df["output_rpm"] = to_float(df[sources["Trans Output Shaft RPM"]])
    df["gear_actual__canon"] = canon_gear(df[sources["Trans Current Gear"]])
    df["gear_cmd__canon"] = canon_gear(df[sources["Trans Commanded Gear"]])
    df["tcc_slip_rpm"] = df["engine_rpm"] - df["turbine_rpm"]
    df["tcc_slip_fused"] = df["tcc_slip_rpm"]
    df["tcc_locked_built"] = (df["tcc_slip_rpm"].abs() <= 50).astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(description="Clean NA Trans logs into a FULL file.")
    parser.add_argument("--in-file", required=True, help="Raw HP Tuners CSV input.")
    parser.add_argument("--out-file", required=True, help="Destination for __clean_full CSV.")
    args = parser.parse_args()

    df = load_csv(args.in_file)
    sources = resolve_sources(df)
    require_columns(sources, args.in_file)
    df = build_canonical(df, sources)
    df.to_csv(args.out_file, index=False, encoding="utf-8")

    print(f"[clean_log_NA] Cleaning {args.in_file} -> {args.out_file}")


if __name__ == "__main__":
    main()
