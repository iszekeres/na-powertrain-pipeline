#!/usr/bin/env python3
"""
tcc_truth_with_psi.py

Analyze TCC behavior using BOTH slip (rpm) and TCC line pressure (psi)
on prepped NOCRUISE logs (e.g., from highway_trans_MAX_analysis.py).

For each gear and TCC state (LOCKED / PARTIAL / OPEN), computes:
  - total_time_s, n_samples
  - median_abs_slip_rpm, p95_abs_slip_rpm
  - median_tcc_line_psi, p95_tcc_line_psi
  - slip_psi_time_integral = sum(dt * slip * psi)  (rough “stress index”)

TCC state classification (EC3-oriented):
  LOCKED : |slip| <= 50 rpm
  PARTIAL:  50 < |slip| < 120 rpm
  OPEN   : |slip| >= 120 rpm
"""

import argparse
import glob
import os
import sys
from typing import List

import numpy as np
import pandas as pd


def pick_col(df: pd.DataFrame, aliases: List[str], label: str) -> str:
    """
    Strict/no-fallback column picker: returns the first matching alias or raises.
    """
    found = [c for c in df.columns if c in aliases]
    if not found:
        raise ValueError(
            f"[tcc_truth_with_psi] Missing required column for {label}. "
            f"Need one of: {aliases}"
        )
    if len(found) > 1:
        sys.stderr.write(
            f"[tcc_truth_with_psi] Warning: multiple candidates for {label}: "
            f"{found}; using {found[0]!r}\n"
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="TCC truth with slip + line-psi, based on prepped NOCRUISE logs."
    )
    ap.add_argument(
        "--prepped-dir",
        required=True,
        help="Directory with prepped NOCRUISE CSVs.",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Directory for outputs; will be created if missing.",
    )
    args = ap.parse_args()

    in_dir = args.prepped_dir
    out_dir = args.out_dir

    if not os.path.isdir(in_dir):
        sys.stderr.write(
            f"[tcc_truth_with_psi] ERROR: --prepped-dir does not exist or is not a directory: {in_dir}\n"
        )
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
    csv_paths = [p for p in csv_paths if "__prepped" in os.path.basename(p)]
    if not csv_paths:
        sys.stderr.write(
            f"[tcc_truth_with_psi] ERROR: No prepped CSV files found in {in_dir}\n"
        )
        sys.exit(1)

    agg = {}  # (gear, state) -> accumulators

    for path in csv_paths:
        sys.stderr.write(f"[tcc_truth_with_psi] Processing {path}\n")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            sys.stderr.write(
                f"[tcc_truth_with_psi] WARNING: Failed to read {path}: {e}\n"
            )
            continue

        try:
            gear_col = pick_col(
                df,
                ["gear_actual__canon", "gear_actual", "Trans Current Gear", "Gear"],
                "gear_actual",
            )
            time_col = pick_col(
                df,
                ["time_s", "Time_s", "Time (s)", "Time", "Elapsed Time (s)", "Elapsed Time"],
                "time_s",
            )
            slip_col = pick_col(
                df,
                ["tcc_slip_fused", "tcc_slip_rpm", "tcc_slip", "TCC Slip", "TCC_Slip", "TCC Slip RPM"],
                "TCC slip",
            )
            psi_col = pick_col(
                df,
                ["tcc_line_psi", "TCC Line Pressure", "TCC Line Pressure (psi)", "TCC Apply Pressure", "TCC Apply Pressure (psi)"],
                "TCC line pressure",
            )
        except ValueError as ve:
            sys.stderr.write(str(ve) + "\n")
            sys.stderr.write("[tcc_truth_with_psi] Aborting due to missing required columns.\n")
            sys.exit(1)

        use = df[[gear_col, time_col, slip_col, psi_col]].copy()
        use = use.replace([np.inf, -np.inf], np.nan).dropna()
        if use.empty:
            sys.stderr.write(
                f"[tcc_truth_with_psi] WARNING: No valid rows after cleaning for {path}\n"
            )
            continue

        # Ensure numeric types
        use[time_col] = pd.to_numeric(use[time_col], errors="coerce")
        use[slip_col] = pd.to_numeric(use[slip_col], errors="coerce")
        use[psi_col] = pd.to_numeric(use[psi_col], errors="coerce")
        use[gear_col] = pd.to_numeric(use[gear_col], errors="coerce")
        use = use.dropna()
        if use.empty:
            sys.stderr.write(
                f"[tcc_truth_with_psi] WARNING: No valid numeric rows for {path}\n"
            )
            continue

        # Sort by time and compute dt
        use = use.sort_values(time_col)
        use["dt"] = use[time_col].diff().fillna(0.0)
        use = use[(use["dt"] >= 0.0) & (use["dt"] <= 5.0)]
        if use.empty:
            sys.stderr.write(
                f"[tcc_truth_with_psi] WARNING: No rows after dt filtering for {path}\n"
            )
            continue

        abs_slip = use[slip_col].abs()
        psi = use[psi_col]
        gear = use[gear_col].round().astype(int)
        dt = use["dt"]
        states = abs_slip.apply(classify_tcc_state)

        for g, s, dti, sl, p in zip(
            gear.values, states.values, dt.values, abs_slip.values, psi.values
        ):
            key = (g, s)
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
        sys.stderr.write(
            "[tcc_truth_with_psi] ERROR: No data aggregated; check prepped-dir and columns.\n"
        )
        sys.exit(1)

    rows = []
    for (g, s), rec in agg.items():
        slip_arr = np.array(rec["slip_values"], dtype=float)
        psi_arr = np.array(rec["psi_values"], dtype=float)
        rows.append(
            {
                "gear": int(g),
                "tcc_state": s,
                "n_samples": int(rec["n_samples"]),
                "total_time_s": float(rec["total_time_s"]),
                "median_abs_slip_rpm": float(np.median(slip_arr)) if slip_arr.size else float("nan"),
                "p95_abs_slip_rpm": safe_percentile(slip_arr, 95),
                "median_tcc_line_psi": float(np.median(psi_arr)) if psi_arr.size else float("nan"),
                "p95_tcc_line_psi": safe_percentile(psi_arr, 95),
                "slip_psi_time_integral": float(rec["slip_psi_time"]),
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["gear", "tcc_state"])
    out_path = os.path.join(out_dir, "tcc_truth_by_gear_with_psi.csv")
    out_df.to_csv(out_path, index=False)
    sys.stderr.write(f"[tcc_truth_with_psi] Wrote {out_path}\n")


if __name__ == "__main__":
    main()
