#!/usr/bin/env python3
"""
Minimal trans_clean_analyze that writes __trans_focus__ artifacts for NA Trans.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "time_s",
    "speed_mph",
    "throttle_pct",
    "pedal_pct",
    "gear_actual__canon",
    "gear_cmd__canon",
    "engine_rpm",
    "turbine_rpm",
    "output_rpm",
    "ect_c",
    "tft_c",
    "brake",
    "tcc_locked_built",
]

SLIP_COLUMNS = ["tcc_slip_rpm", "tcc_slip_fused"]


def ensure_columns(df, path):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    slip_present = any(col in df.columns for col in SLIP_COLUMNS)
    if missing or not slip_present:
        print(f"[ERROR] Missing canonical columns in {path}")
        if missing:
            print(f"[ERROR]  Missing: {missing}")
        if not slip_present:
            print(f"[ERROR]  Missing slip column (tcc_slip_rpm or tcc_slip_fused)")
        sys.exit(1)


def canonical_slip(df):
    if "tcc_slip_rpm" in df.columns:
        return df["tcc_slip_rpm"]
    return df["tcc_slip_fused"]


def build_shift_events(df):
    sorted_df = df.sort_values("time_s")
    events = []
    gears = sorted_df["gear_actual__canon"].fillna(0).astype(int)
    for idx in range(1, len(sorted_df)):
        prev = gears.iloc[idx - 1]
        current = gears.iloc[idx]
        row = sorted_df.iloc[idx]
        if prev != current and 1 <= prev <= 6 and 1 <= current <= 6:
            events.append(
                {
                    "from_gear": prev,
                    "to_gear": current,
                    "time_s": row["time_s"],
                    "speed_mph": row["speed_mph"],
                    "throttle_pct": row["throttle_pct"],
                    "pedal_pct": row["pedal_pct"],
                    "ect_c": row["ect_c"],
                    "tft_c": row["tft_c"],
                    "tcc_locked_built": row["tcc_locked_built"],
                }
            )
    return pd.DataFrame(events)


def build_mapping(df):
    pedal_raw = "Accelerator Pedal Position" if "Accelerator Pedal Position" in df.columns else ""
    rows = [
        ("time_s", "Time"),
        ("speed_mph", "Vehicle Speed"),
        ("engine_rpm", "Engine RPM"),
        ("throttle_pct", "Throttle Position (%)"),
        ("pedal_pct", pedal_raw),
        ("turbine_rpm", "Trans Input Shaft RPM"),
        ("output_rpm", "Trans Output Shaft RPM"),
        ("gear_actual__canon", "Trans Current Gear"),
        ("gear_cmd__canon", "Trans Commanded Gear"),
        ("brake_pressure_kpa", "Brake Pressure"),
        ("ect_c", "Engine Coolant Temp"),
        ("tft_c", "Trans Fluid Temp"),
        ("tcc_slip_rpm", "derived"),
        ("tcc_locked_built", "derived"),
    ]
    return pd.DataFrame(rows, columns=["canonical_name", "raw_name"])


def build_summary(df):
    summary = []
    summary.append(f"Total samples: {len(df)}")
    duration = df["time_s"].max() - df["time_s"].min() if len(df) >= 2 else 0.0
    summary.append(f"Duration sec: {duration:.2f}")

    summary.append("Gear usage fraction:")
    for gear in range(7):
        frac = (df["gear_actual__canon"] == gear).mean()
        summary.append(f"  gear {gear}: {frac:.2%}")

    locked = df["tcc_locked_built"].mean()
    summary.append(f"TCC locked fraction (all gears): {locked:.2%}")
    slip = canonical_slip(df).abs()
    for gear in range(3, 7):
        mask = df["gear_actual__canon"] == gear
        if mask.any():
            locked_frac = df.loc[mask, "tcc_locked_built"].mean()
            slip_mean = slip.loc[mask].mean()
            summary.append(
                f"  gear {gear} locked {locked_frac:.2%}, mean slip {slip_mean:.1f} rpm"
            )

    summary.append("Temps (ect_c / tft_c):")
    summary.append(
        f"  ect_c min/max/mean: {df['ect_c'].min():.1f} / {df['ect_c'].max():.1f} / {df['ect_c'].mean():.1f}"
    )
    summary.append(
        f"  tft_c min/max/mean: {df['tft_c'].min():.1f} / {df['tft_c'].max():.1f} / {df['tft_c'].mean():.1f}"
    )

    summary.append("Speed (mph) min/max/mean:")
    summary.append(
        f"  {df['speed_mph'].min():.1f} / {df['speed_mph'].max():.1f} / {df['speed_mph'].mean():.1f}"
    )

    return "\n".join(summary)


def safe_write(df, path):
    df.to_csv(path, index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Minimal trans_clean_analyze for NA Trans logs.")
    parser.add_argument("--in-file", required=True, help="Clean FULL CSV input.")
    parser.add_argument("--out-dir", required=True, help="Directory for trans_focus outputs.")
    args = parser.parse_args()

    in_path = args.in_file
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(in_path, low_memory=False)
    ensure_columns(df, in_path)
    df["tcc_slip_rpm"] = canonical_slip(df)

    basename = os.path.basename(in_path)
    clean_out = os.path.join(out_dir, f"__trans_focus__clean_FULL__{basename}")
    shifts_out = os.path.join(out_dir, f"__trans_focus__shift_events__{basename}")
    mapping_out = os.path.join(out_dir, f"__trans_focus__mapping__{basename}")
    summary_out = os.path.join(out_dir, f"__trans_focus__summary__{basename}.txt")

    safe_write(df, clean_out)
    build_shift_events(df).to_csv(shifts_out, index=False, encoding="utf-8")
    build_mapping(df).to_csv(mapping_out, index=False, encoding="utf-8")
    with open(summary_out, "w", encoding="utf-8") as fh:
        fh.write(build_summary(df))

    print(f"[trans_clean_analyze] wrote clean/mapping/shift/summary for {basename}")


if __name__ == "__main__":
    main()
