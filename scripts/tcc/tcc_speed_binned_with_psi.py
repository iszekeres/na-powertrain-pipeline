#!/usr/bin/env python3
"""
tcc_speed_binned_with_psi.py

Bin TCC slip + TCC line pressure by:
  - gear (1–6)
  - TCC state (LOCKED / PARTIAL / OPEN, from slip)
  - vehicle speed band (e.g., 0–5, 5–10, ..., 95–100 mph)

Strict / no-fallback required signals:
  - gear_actual: gear_actual__canon, gear_actual, "Trans Current Gear", "Gear"
  - time_s: time_s, Time (s), Time, etc.
  - tcc slip (rpm): tcc_slip_fused, "TCC Slip", "TCC Slip (rpm)", TCC_Slip
  - TCC line pressure (psi): tcc_line_psi, "TCC Line Pressure", "TCC Line Pressure (psi)", "TCC Apply Pressure", "TCC Apply Pressure (psi)"
  - vehicle speed (mph): speed_mph, "Vehicle Speed", "Vehicle Speed (MPH)", "Speed (mph)", "Speed"
"""

import argparse
import glob
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd


def pick_col(df: pd.DataFrame, aliases: List[str], label: str) -> str:
    found = [c for c in df.columns if c in aliases]
    if not found:
        raise ValueError(
            f"[tcc_speed_binned_with_psi] Missing required column for {label}. Need one of: {aliases}"
        )
    if len(found) > 1:
        sys.stderr.write(
            f"[tcc_speed_binned_with_psi] Warning: multiple candidates for {label}: {found}; using {found[0]!r}\n"
        )
    return found[0]


def classify_tcc_state(abs_slip_rpm: float) -> str:
    if abs_slip_rpm <= 50.0:
        return "LOCKED"
    elif abs_slip_rpm >= 120.0:
        return "OPEN"
    else:
        return "PARTIAL"


def safe_percentile(arr: np.ndarray, p: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, p))


def build_speed_bins(bin_width: float = 5.0, max_speed: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.arange(0.0, max_speed + bin_width, bin_width)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return edges, centers


def main() -> None:
    ap = argparse.ArgumentParser(description="Speed-binned TCC slip+psi stats by gear and TCC state.")
    ap.add_argument("--prepped-dir", required=True, help="Directory with prepped NOCRUISE CSVs.")
    ap.add_argument("--out-csv", required=True, help="Path to output CSV.")
    ap.add_argument("--bin-width", type=float, default=5.0, help="Speed bin width in mph (default 5.0).")
    ap.add_argument("--max-speed", type=float, default=100.0, help="Maximum speed for bins (default 100 mph).")
    args = ap.parse_args()

    in_dir = args.prepped_dir
    out_csv = args.out_csv
    bin_width = float(args.bin_width)
    max_speed = float(args.max_speed)

    if not os.path.isdir(in_dir):
        sys.stderr.write(f"[tcc_speed_binned_with_psi] ERROR: --prepped-dir is not a directory: {in_dir}\n")
        sys.exit(1)

    csv_paths = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
    csv_paths = [p for p in csv_paths if "__prepped" in os.path.basename(p)]
    if not csv_paths:
        sys.stderr.write(f"[tcc_speed_binned_with_psi] ERROR: No CSV files found in {in_dir}\n")
        sys.exit(1)

    edges, centers = build_speed_bins(bin_width=bin_width, max_speed=max_speed)

    agg = {}

    for path in csv_paths:
        sys.stderr.write(f"[tcc_speed_binned_with_psi] Processing {path}\n")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            sys.stderr.write(f"[tcc_speed_binned_with_psi] WARNING: Failed to read {path}: {e}\n")
            continue

        try:
            gear_col = pick_col(df, ["gear_actual__canon", "gear_actual", "Trans Current Gear", "Gear"], "gear_actual")
            time_col = pick_col(df, ["time_s", "Time_s", "Time (s)", "Time", "Elapsed Time (s)", "Elapsed Time"], "time_s")
            slip_col = pick_col(df, ["tcc_slip_fused", "TCC Slip", "TCC Slip (rpm)", "TCC_Slip"], "TCC slip")
            psi_col = pick_col(
                df,
                ["tcc_line_psi", "TCC Line Pressure", "TCC Line Pressure (psi)", "TCC Apply Pressure", "TCC Apply Pressure (psi)"],
                "TCC line pressure",
            )
            speed_col = pick_col(
                df,
                ["speed_mph", "Vehicle Speed", "Vehicle Speed (MPH)", "Speed (mph)", "Speed"],
                "vehicle speed (mph)",
            )
        except ValueError as ve:
            sys.stderr.write(str(ve) + "\n")
            sys.stderr.write("[tcc_speed_binned_with_psi] Aborting due to missing required columns.\n")
            sys.exit(1)

        use = df[[gear_col, time_col, slip_col, psi_col, speed_col]].copy()
        use = use.replace([np.inf, -np.inf], np.nan).dropna()
        if use.empty:
            sys.stderr.write(f"[tcc_speed_binned_with_psi] WARNING: No valid rows after cleaning for {path}\n")
            continue

        use[time_col] = pd.to_numeric(use[time_col], errors="coerce")
        use[slip_col] = pd.to_numeric(use[slip_col], errors="coerce")
        use[psi_col] = pd.to_numeric(use[psi_col], errors="coerce")
        use[gear_col] = pd.to_numeric(use[gear_col], errors="coerce")
        use[speed_col] = pd.to_numeric(use[speed_col], errors="coerce")
        use = use.dropna()
        if use.empty:
            sys.stderr.write(f"[tcc_speed_binned_with_psi] WARNING: No valid numeric rows for {path}\n")
            continue

        use = use.sort_values(time_col)
        use["dt"] = use[time_col].diff().fillna(0.0)
        use = use[(use["dt"] >= 0.0) & (use["dt"] <= 5.0)]
        if use.empty:
            sys.stderr.write(f"[tcc_speed_binned_with_psi] WARNING: No rows after dt filtering for {path}\n")
            continue

        abs_slip = use[slip_col].abs().values
        psi = use[psi_col].values
        gear = use[gear_col].round().astype(int).values
        speed = use[speed_col].values
        dt = use["dt"].values

        bin_idx = np.digitize(speed, edges) - 1
        valid_mask = (bin_idx >= 0) & (bin_idx < len(edges) - 1)
        abs_slip = abs_slip[valid_mask]
        psi = psi[valid_mask]
        gear = gear[valid_mask]
        dt = dt[valid_mask]
        bin_idx = bin_idx[valid_mask]

        states = np.array([classify_tcc_state(s) for s in abs_slip], dtype=object)

        for g, s, b, dti, sl, p in zip(gear, states, bin_idx, dt, abs_slip, psi):
            key = (int(g), str(s), int(b))
            rec = agg.setdefault(
                key,
                {
                    "n_samples": 0,
                    "total_time_s": 0.0,
                    "slip_values": [],
                    "psi_values": [],
                    "slip_psi_time": 0.0,
                },
            )
            rec["n_samples"] += 1
            rec["total_time_s"] += float(dti)
            rec["slip_values"].append(float(sl))
            rec["psi_values"].append(float(p))
            rec["slip_psi_time"] += float(dti) * float(sl) * float(p)

    if not agg:
        sys.stderr.write("[tcc_speed_binned_with_psi] ERROR: No data aggregated; check prepped-dir and columns.\n")
        sys.exit(1)

    rows = []
    for (g, s, b), rec in agg.items():
        slip_arr = np.array(rec["slip_values"], dtype=float)
        psi_arr = np.array(rec["psi_values"], dtype=float)
        low = float(edges[b])
        high = float(edges[b + 1])
        center = float(centers[b])
        rows.append(
            {
                "gear": int(g),
                "tcc_state": s,
                "speed_bin_low_mph": low,
                "speed_bin_high_mph": high,
                "speed_bin_center_mph": center,
                "n_samples": int(rec["n_samples"]),
                "total_time_s": float(rec["total_time_s"]),
                "median_abs_slip_rpm": float(np.median(slip_arr)) if slip_arr.size else float("nan"),
                "p95_abs_slip_rpm": safe_percentile(slip_arr, 95),
                "median_tcc_line_psi": float(np.median(psi_arr)) if psi_arr.size else float("nan"),
                "p95_tcc_line_psi": safe_percentile(psi_arr, 95),
                "slip_psi_time_integral": float(rec["slip_psi_time"]),
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["gear", "tcc_state", "speed_bin_low_mph"])
    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    sys.stderr.write(f"[tcc_speed_binned_with_psi] Wrote {out_csv}\n")


if __name__ == "__main__":
    main()
