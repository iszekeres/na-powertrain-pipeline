#!/usr/bin/env python3
"""
Summarize pedal vs throttle usage from __NOCRUISE logs.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

REQUIRED_COLS = [
    "speed_mph",
    "pedal_pct",
    "throttle_pct",
    "gear_actual",
    "brake",
    "time_s",
]

# Canonical 17-point TPS axis
TPS_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]


def check_required_cols(df, path):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"[ERROR] {os.path.basename(path)} is missing required columns:")
        for c in missing:
            print(f"   - {c}")
        return False
    return True


def summarize_file(path, highway_only=True):
    print(f"[INFO] Reading {path} ...")
    df = pd.read_csv(path)

    if not check_required_cols(df, path):
        return None

    for col in ["speed_mph", "pedal_pct", "throttle_pct", "gear_actual", "brake"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mask = df["brake"] == 0
    mask &= df["gear_actual"].between(3, 6)
    if highway_only:
        mask &= df["speed_mph"].between(35, 90)

    df = df.loc[mask].copy()
    if df.empty:
        print(f"[WARN] No samples left after filters in {os.path.basename(path)}.")
        return None

    df["throttle_pct"] = df["throttle_pct"].clip(lower=0, upper=100)

    edges = [0]
    for a, b in zip(TPS_AXIS[:-1], TPS_AXIS[1:]):
        edges.append((a + b) / 2.0)
    edges.append(100.0001)

    df["tps_bin"] = pd.cut(
        df["throttle_pct"],
        bins=edges,
        labels=TPS_AXIS,
        include_lowest=True,
        right=False,
    ).astype(float)
    df = df.dropna(subset=["tps_bin"])
    if df.empty:
        print(f"[WARN] No samples with valid TPS bins in {os.path.basename(path)}.")
        return None

    grouped = df.groupby(["gear_actual", "tps_bin"])
    rows = []
    for (gear, tps_bin), g in grouped:
        pedal = g["pedal_pct"].dropna()
        if pedal.empty:
            continue
        rows.append(
            {
                "source_file": os.path.basename(path),
                "gear": int(gear),
                "tps_bin": float(tps_bin),
                "n_samples": int(len(pedal)),
                "pedal_pct_mean": float(pedal.mean()),
                "pedal_pct_median": float(pedal.median()),
                "pedal_pct_p25": float(pedal.quantile(0.25)),
                "pedal_pct_p75": float(pedal.quantile(0.75)),
            }
        )
    if not rows:
        print(f"[WARN] No grouped rows produced for {os.path.basename(path)}.")
        return None
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Summarize pedal vs throttle usage from __NOCRUISE logs."
    )
    ap.add_argument(
        "--prepped-dir",
        required=True,
        help="Directory containing *__NOCRUISE.csv files (e.g. newlogs\\highway_MAX_analysis\\prepped)",
    )
    ap.add_argument(
        "--glob-substring",
        default="__NOCRUISE",
        help="Substring that files must contain to be included (default: __NOCRUISE)",
    )
    ap.add_argument(
        "--out-dir",
        default="newlogs\\highway_MAX_analysis",
        help="Directory for output summary CSV file",
    )
    args = ap.parse_args()

    prepped_dir = args.prepped_dir
    out_dir = args.out_dir

    if not os.path.isdir(prepped_dir):
        print(f"[ERROR] Prepped dir not found: {prepped_dir}")
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    all_frames = []
    for name in sorted(os.listdir(prepped_dir)):
        if not name.lower().endswith(".csv"):
            continue
        if args.glob_substring not in name:
            continue
        path = os.path.join(prepped_dir, name)
        df_summary = summarize_file(path)
        if df_summary is None:
            continue
        all_frames.append(df_summary)

    if not all_frames:
        print("[ERROR] No summary data produced from any file.")
        sys.exit(1)

    combined = pd.concat(all_frames, ignore_index=True)
    grouped = combined.groupby(["gear", "tps_bin"])
    agg_rows = []
    for (gear, tps_bin), g in grouped:
        n = int(g["n_samples"].sum())
        if n == 0:
            continue
        weights = g["n_samples"].values.astype(float)
        pedal_mean = float(np.average(g["pedal_pct_mean"], weights=weights))
        pedal_median = float(g["pedal_pct_median"].median())
        pedal_p25 = float(g["pedal_pct_p25"].median())
        pedal_p75 = float(g["pedal_pct_p75"].median())
        agg_rows.append(
            {
                "gear": int(gear),
                "tps_bin": float(tps_bin),
                "n_samples_total": n,
                "pedal_pct_mean_w": pedal_mean,
                "pedal_pct_median_med": pedal_median,
                "pedal_pct_p25_med": pedal_p25,
                "pedal_pct_p75_med": pedal_p75,
            }
        )
    agg = pd.DataFrame(agg_rows)

    out_path_detail = os.path.join(out_dir, "pedal_vs_throttle__NOCRUISE__by_file.csv")
    out_path_agg = os.path.join(out_dir, "pedal_vs_throttle__NOCRUISE__summary.csv")
    combined.to_csv(out_path_detail, index=False)
    agg.to_csv(out_path_agg, index=False)

    print(f"[INFO] Wrote detailed per-file summary to: {out_path_detail}")
    print(f"[INFO] Wrote aggregated summary to:      {out_path_agg}")
    print("[DONE] Pedal vs TPS usage summary complete.")


if __name__ == "__main__":
    main()
