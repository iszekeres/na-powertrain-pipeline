#!/usr/bin/env python3
"""
classify_tcc_coupling_modes.py

Usage:
    python classify_tcc_coupling_modes.py --in <speed_binned_csv> --out <classified_csv>

Input CSV columns expected:
    gear,tcc_state,speed_bin_low_mph,speed_bin_high_mph,speed_bin_center_mph,
    n_samples,total_time_s,median_abs_slip_rpm,p95_abs_slip_rpm,
    median_tcc_line_psi,p95_tcc_line_psi,slip_psi_time_integral

Strict/no-fallback: fails if required columns missing.
"""
import argparse
import math
import sys
import pandas as pd

HYDRO_SLIP_MAX_RPM = 50.0
HYDRO_PSI_MAX = 5.0
HARDLOCK_SLIP_MAX_RPM = 25.0
HARDLOCK_PSI_MIN = 25.0
EC3_PARTIAL_SLIP_MIN = 40.0
EC3_PARTIAL_SLIP_MAX = 120.0
EC3_PARTIAL_PSI_MIN = 5.0
EC3_PARTIAL_PSI_MAX = 60.0
HIGH_SLIP_RPM = 150.0
MIN_SAMPLES = 10
MIN_TIME_S = 0.1

def classify_row(row) -> str:
    n = row.get("n_samples", 0) or row.get("count", 0)
    t = row.get("total_time_s", 0.0)
    slip = float(row.get("median_abs_slip_rpm", 0.0))
    psi = float(row.get("median_tcc_line_psi", 0.0))
    if n < MIN_SAMPLES or t < MIN_TIME_S:
        return "LOW_DATA"
    if math.isnan(slip):
        slip = 0.0
    if math.isnan(psi):
        psi = 0.0
    if psi <= HYDRO_PSI_MAX:
        if slip <= HYDRO_SLIP_MAX_RPM:
            return "HYDRO_COUPLED"
        elif slip >= HIGH_SLIP_RPM:
            return "HYDRO_HIGH_SLIP"
        else:
            return "HYDRO_MID_SLIP"
    if slip <= HARDLOCK_SLIP_MAX_RPM and psi >= HARDLOCK_PSI_MIN:
        return "HARD_LOCK"
    if EC3_PARTIAL_SLIP_MIN <= slip <= EC3_PARTIAL_SLIP_MAX and psi >= EC3_PARTIAL_PSI_MIN:
        return "EC3_PARTIAL_SOFT" if psi <= EC3_PARTIAL_PSI_MAX else "EC3_PARTIAL_FIRM"
    if slip <= HYDRO_SLIP_MAX_RPM and psi > HYDRO_PSI_MAX:
        return "HARD_LOCK_SOFTSLIP" if psi >= HARDLOCK_PSI_MIN else "EC3_ASSIST"
    if slip >= HIGH_SLIP_RPM and psi > HYDRO_PSI_MAX:
        return "EC3_HIGH_SLIP"
    return "OTHER_MIXED"


def main():
    ap = argparse.ArgumentParser(description="Classify TCC coupling modes from speed-binned TCC summary.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV (speed-binned TCC summary).")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV with coupling_mode column.")
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    required_cols = [
        "gear",
        "speed_bin_low_mph",
        "speed_bin_high_mph",
        "speed_bin_center_mph",
        "n_samples",
        "total_time_s",
        "median_abs_slip_rpm",
        "median_tcc_line_psi",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("ERROR: Missing required columns:", ", ".join(missing))
        sys.exit(1)

    df["coupling_mode"] = df.apply(classify_row, axis=1)
    df.to_csv(args.out_path, index=False)
    print(f"Wrote classified table to: {args.out_path}")

    if "total_time_s" in df.columns:
        summary = (
            df.groupby(["gear", "coupling_mode"])["total_time_s"]
            .sum()
            .reset_index()
            .sort_values(["gear", "coupling_mode"])
        )
        print("\n=== Total time by gear & coupling_mode (s) ===")
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(summary)
    else:
        print("NOTE: total_time_s column not found; skipping time summary.")


if __name__ == "__main__":
    main()
