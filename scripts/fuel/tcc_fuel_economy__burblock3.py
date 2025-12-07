#!/usr/bin/env python
"""
Fuel economy surface builder for burblock5.csv

Uses "Inst Fuel Used" (L/h) and vehicle speed to compute:
  - Per-sample distance (miles) and fuel (gallons)
  - Global MPG
  - MPG vs speed (1 mph bins)
  - MPG vs speed × gear
  - MPG vs speed × pedal (1% bins)
  - MPG vs speed × gear × TCC state (LOCKED / PARTIAL / OPEN)

TCC state is classified using EC3-aware slip thresholds:
  |slip| <= 50 rpm   -> LOCKED
  |slip| >= 120 rpm  -> OPEN
  otherwise          -> PARTIAL

Input:
  newlogs/burblock5.csv

Outputs (to newlogs/output/fuel_economy_burblock5/):
  fuel_global_summary__burblock5.txt
  fuel_vs_speed__burblock5.csv
  fuel_vs_speed_gear__burblock5.csv
  fuel_vs_speed_pedal__burblock5.csv
  fuel_vs_speed_gear_tcc__burblock5.csv
"""

import math
import os
import pathlib

import numpy as np
import pandas as pd
from tcc_state_utils import classify_tcc_state_psi

LOG_PATH = r"newlogs/burblock5.csv"
OUT_DIR = r"newlogs/output/fuel_economy_burblock5"

# Fuel conversion
L_PER_GAL = 3.785411784

# Speed range for "highway" slice in summary
HWY_SPEED_MIN = 55.0
# common use cases for top end; keep 80 for this cut
HWY_SPEED_MAX = 80.0


def pick_column(df, aliases, required=True, desc=""):
    cols_lower = {c.lower(): c for c in df.columns}
    for name in aliases:
        if name.lower() in cols_lower:
            col = cols_lower[name.lower()]
            print(f"[OK] Using column '{col}' for {desc or name}")
            return col
    if required:
        raise ValueError(f"Required column for {desc or aliases[0]} not found. Tried: {aliases}")
    print(f"[WARN] Optional column for {desc or aliases[0]} not found. Tried: {aliases}")
    return None


def classify_tcc_state(abs_slip, psi):
    return classify_tcc_state_psi(abs_slip, psi)


def main():
    print(f"Loading log: {LOG_PATH}")
    # Find header row (Offset/Time)
    header_line = 0
    with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f_in:
        for idx, line in enumerate(f_in):
            if line.startswith("Offset,") or line.startswith("Time,") or line.startswith("Time (s)"):
                header_line = idx
                break
    df = pd.read_csv(LOG_PATH, skiprows=header_line)
    # Drop units row
    if not df.empty:
        df = df.iloc[1:].reset_index(drop=True)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns (header line {header_line}).")

    # Required columns
    time_col = pick_column(
        df, ["time_s", "Elapsed Time", "Elapsed Time (s)", "Offset", "Time"], True, "time_s"
    )
    speed_col = pick_column(
        df, ["speed_mph", "Vehicle Speed (SAE)", "Vehicle Speed (mph)", "Vehicle Speed"], True, "speed_mph"
    )
    fuel_flow_col = pick_column(
        df,
        [
            "Inst Fuel Used",
            "Instantaneous Fuel Flow Estimate",
            "Inst Fuel Flow Estimate",
            "Advance Fuel Flow Estimate",
        ],
        True,
        "fuel_flow_L_per_h",
    )
    gear_col = pick_column(
        df,
        ["gear_actual__canon", "Trans Current Gear", "Trans Current Gear 1", "Trans Current Gear.1", "Current Gear", "Gear"],
        True,
        "gear_actual",
    )

    # Optional for TCC classification
    slip_col = pick_column(df, ["tcc_slip_fused", "TCC Slip", "TCC Slip RPM", "Torque Converter Slip"], False, "tcc_slip_rpm")
    engine_rpm_col = pick_column(df, ["Engine RPM (SAE)", "Engine RPM", "RPM"], False, "engine_rpm")
    turbine_rpm_col = pick_column(df, ["Trans Input Shaft RPM", "Trans Turbine RPM"], False, "turbine_rpm")
    pedal_col = pick_column(
        df, ["pedal_pct", "Accelerator Pedal Position", "Accelerator Pedal Position (%)"], False, "pedal_pct"
    )
    tcc_line_col = pick_column(df, ["TCC Line Pressure", "TCC Apply Pressure", "TCC Line (PSI)", "TCC Pressure"], False, "tcc_line_psi")

    # Build slip
    if slip_col:
        slip = pd.to_numeric(df[slip_col], errors="coerce")
    elif engine_rpm_col and turbine_rpm_col:
        print("[INFO] Computing slip as Engine RPM - Turbine RPM.")
        slip = pd.to_numeric(df[engine_rpm_col], errors="coerce") - pd.to_numeric(
            df[turbine_rpm_col], errors="coerce"
        )
    else:
        print("[WARN] No slip or engine+turbine RPM; TCC state will be 'UNKNOWN'.")
        slip = pd.Series([math.nan] * len(df))

    # Core working frame
    work = pd.DataFrame(
        {
            "time_s": pd.to_numeric(df[time_col], errors="coerce"),
            "speed_mph": pd.to_numeric(df[speed_col], errors="coerce"),
            "fuel_L_per_h": pd.to_numeric(df[fuel_flow_col], errors="coerce"),
            "gear": pd.to_numeric(df[gear_col], errors="coerce"),
            "tcc_line_psi": pd.to_numeric(df[tcc_line_col], errors="coerce") if tcc_line_col else np.nan,
        }
    )

    # dt (seconds)
    work["dt"] = work["time_s"].diff().fillna(0.0).clip(lower=0.0, upper=1.0)

    # Distance and fuel per-sample
    work["d_miles"] = work["speed_mph"] * (work["dt"] / 3600.0)
    work["d_L"] = work["fuel_L_per_h"] * (work["dt"] / 3600.0)
    work["d_gal"] = work["d_L"] / L_PER_GAL

    # Pedal
    if pedal_col:
        work["pedal_pct"] = pd.to_numeric(df[pedal_col], errors="coerce").clip(0, 100)
    else:
        work["pedal_pct"] = 0.0

    # TCC state (psi-aware). NaN psi => None (drop for TCC grouping).
    abs_slip = pd.to_numeric(slip, errors="coerce").abs()
    psi_series = work["tcc_line_psi"]
    work["tcc_state"] = [classify_tcc_state(s, p) for s, p in zip(abs_slip, psi_series)]

    # Filter: forward gears, positive dt
    work = work[(work["gear"].between(1, 6)) & (work["dt"] > 0)].copy()
    work.reset_index(drop=True, inplace=True)
    print(f"Rows after gear/dt filter: {len(work):,}")

    # Global MPG
    total_miles = work["d_miles"].sum()
    total_gal = work["d_gal"].sum()
    avg_mpg = total_miles / total_gal if total_gal > 0 else float("nan")

    # Highway slice: 55-80 mph, gear >= 5, TCC LOCKED (if known)
    hwy_mask = (
        work["speed_mph"].between(HWY_SPEED_MIN, HWY_SPEED_MAX)
        & (work["gear"] >= 5)
        & (work["tcc_state"] == "LOCKED")
    )
    hwy = work[hwy_mask]
    hwy_miles = hwy["d_miles"].sum()
    hwy_gal = hwy["d_gal"].sum()
    hwy_mpg = hwy_miles / hwy_gal if hwy_gal > 0 else float("nan")

    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    summary_path = os.path.join(OUT_DIR, "fuel_global_summary__burblock5.txt")
    with open(summary_path, "w") as f:
        f.write("Global fuel economy summary (burblock5)\n")
        f.write("---------------------------------------\n")
        f.write(f"Total distance: {total_miles:.2f} miles\n")
        f.write(f"Total fuel:     {total_gal:.3f} gallons\n")
        f.write(f"Average MPG:    {avg_mpg:.2f}\n\n")
        f.write(
            f"Highway slice (speed {HWY_SPEED_MIN}-{HWY_SPEED_MAX} mph, gear>=5, TCC LOCKED):\n"
        )
        f.write(f"  Distance: {hwy_miles:.2f} miles\n")
        f.write(f"  Fuel:     {hwy_gal:.3f} gallons\n")
        f.write(f"  MPG:      {hwy_mpg:.2f}\n")
    print(f"Wrote global summary to: {summary_path}")

    # 1) MPG vs speed (1 mph bins)
    work["speed_bin_mph"] = work["speed_mph"].astype(int)
    grp_speed = (
        work.groupby("speed_bin_mph", observed=True)
        .agg(total_time_s=("dt", "sum"), distance_miles=("d_miles", "sum"), fuel_gal=("d_gal", "sum"))
        .reset_index()
    )
    grp_speed["mpg"] = grp_speed["distance_miles"] / grp_speed["fuel_gal"].where(
        grp_speed["fuel_gal"] > 0, math.nan
    )
    out_speed = os.path.join(OUT_DIR, "fuel_vs_speed__burblock5.csv")
    grp_speed.to_csv(out_speed, index=False)
    print(f"Wrote MPG vs speed to: {out_speed}")

    # 2) MPG vs speed x gear
    grp_speed_gear = (
        work.groupby(["gear", "speed_bin_mph"], observed=True)
        .agg(total_time_s=("dt", "sum"), distance_miles=("d_miles", "sum"), fuel_gal=("d_gal", "sum"))
        .reset_index()
    )
    grp_speed_gear["mpg"] = grp_speed_gear["distance_miles"] / grp_speed_gear["fuel_gal"].where(
        grp_speed_gear["fuel_gal"] > 0, math.nan
    )
    out_speed_gear = os.path.join(OUT_DIR, "fuel_vs_speed_gear__burblock5.csv")
    grp_speed_gear.to_csv(out_speed_gear, index=False)
    print(f"Wrote MPG vs speed x gear to: {out_speed_gear}")

    # 3) MPG vs speed x pedal (1% bins)
    work["pedal_bin_pct"] = work["pedal_pct"].astype(int)
    grp_speed_pedal = (
        work.groupby(["speed_bin_mph", "pedal_bin_pct"], observed=True)
        .agg(total_time_s=("dt", "sum"), distance_miles=("d_miles", "sum"), fuel_gal=("d_gal", "sum"))
        .reset_index()
    )
    grp_speed_pedal["mpg"] = grp_speed_pedal["distance_miles"] / grp_speed_pedal["fuel_gal"].where(
        grp_speed_pedal["fuel_gal"] > 0, math.nan
    )
    out_speed_pedal = os.path.join(OUT_DIR, "fuel_vs_speed_pedal__burblock5.csv")
    grp_speed_pedal.to_csv(out_speed_pedal, index=False)
    print(f"Wrote MPG vs speed x pedal to: {out_speed_pedal}")

    # 4) MPG vs speed x gear x TCC state (drop unknown psi rows)
    work_tcc = work[work["tcc_state"].notna()].copy()
    grp_speed_gear_tcc = (
        work_tcc.groupby(["gear", "speed_bin_mph", "tcc_state"], observed=True)
        .agg(total_time_s=("dt", "sum"), distance_miles=("d_miles", "sum"), fuel_gal=("d_gal", "sum"))
        .reset_index()
    )
    grp_speed_gear_tcc["mpg"] = grp_speed_gear_tcc["distance_miles"] / grp_speed_gear_tcc["fuel_gal"].where(
        grp_speed_gear_tcc["fuel_gal"] > 0, math.nan
    )
    out_speed_gear_tcc = os.path.join(OUT_DIR, "fuel_vs_speed_gear_tcc__burblock5.csv")
    grp_speed_gear_tcc.to_csv(out_speed_gear_tcc, index=False)
    print(f"Wrote MPG vs speed x gear x TCC to: {out_speed_gear_tcc}")
    print("Done. Open those CSVs in Excel and we can start hunting for the best MPG zones by gear/TCC/pedal.")


if __name__ == "__main__":
    main()



