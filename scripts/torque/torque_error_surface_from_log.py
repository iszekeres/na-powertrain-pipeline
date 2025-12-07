#!/usr/bin/env python3
"""
Build a torque error surface from a single HP Tuners log.

Usage (from project root):
    python torque_error_surface_from_log.py --log newlogs/burblock3.csv --out-dir newlogs/torque_error

Defaults:
    --log     newlogs/burblock3.csv
    --out-dir newlogs/torque_error

Bins:
    - 100 rpm bins (centered)
    - 1 mph speed bins (centered)
"""

import argparse
import os
import textwrap

import numpy as np
import pandas as pd

# --- Vehicle constants for your Tahoe ---
MASS_KG = 2676.0  # 5900 lb canonical
FD = 3.08
TIRE_DIAM_IN = 32.5
INCH_TO_M = 0.0254
TIRE_RADIUS_M = (TIRE_DIAM_IN * INCH_TO_M) / 2.0

# 6L80 gear ratios (2015 Tahoe)
GEAR_RATIOS = {
    1: 4.027,
    2: 2.364,
    3: 1.532,
    4: 1.152,
    5: 0.852,
    6: 0.667,
}


# --- Helpers ---------------------------------------------------------
def find_header_line(path: str) -> int:
    """
    HP Tuners CSV has metadata; detect the header row (Offset/Time).
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f_in:
        for idx, line in enumerate(f_in):
            if line.startswith("Offset,") or line.startswith("Time,") or line.startswith("Time (s)"):
                return idx
    return 0


def pick_col(df: pd.DataFrame, candidates):
    """
    Return the first column name from `candidates` (case-insensitive) that exists in df.
    Raise a clear error if none are found.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        key = name.lower()
        if key in cols_lower:
            return cols_lower[key]
    raise KeyError(f"None of {candidates!r} found in columns.")


def classify_tcc_state(slip_rpm: float) -> str:
    """
    Simple EC³-ish TCC state classifier from slip rpm.
    """
    if pd.isna(slip_rpm):
        return "UNKNOWN"
    s = abs(float(slip_rpm))
    if s <= 50.0:
        return "LOCKED"
    elif s <= 120.0:
        return "PARTIAL"
    else:
        return "OPEN"


def build_torque_error_surfaces(df: pd.DataFrame, log_name: str, out_dir: str):
    """
    Core logic: compute physics torque, compare to ECM torque, and build
    gear×RPM and gear×speed error surfaces.
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Pick columns (robust to HP Tuners headers) ---
    time_col = pick_col(df, ["time_s", "time", "time (s)", "time (sec)", "elapsed time", "offset"])
    speed_col = pick_col(df, ["speed_mph", "vehicle speed (sae)", "vehicle speed", "mph"])
    rpm_col = pick_col(df, ["engine rpm (sae)", "engine rpm", "rpm"])
    gear_col = pick_col(df, ["gear_actual", "gear_actual__canon", "trans current gear", "trans current gear 1", "trans current gear.1", "gear"])
    # Torque: prefer Delivered Engine Torque, fall back to Engine Torque
    try:
        torq_col = pick_col(df, ["delivered engine torque"])
    except KeyError:
        torq_col = pick_col(df, ["engine torque", "engine torque (sae)"])
    # TCC slip: prefer tcc slip, allow trans slip
    try:
        tcc_slip_col = pick_col(df, ["tcc slip", "tcc slip rpm", "tcc_slip_fused"])
    except KeyError:
        try:
            tcc_slip_col = pick_col(df, ["trans slip rpm", "trans slip"])
        except KeyError:
            tcc_slip_col = None

    # Brake is optional
    brake_col = None
    for cand in ["brake pressure", "brake pressure (kpa)", "brake"]:
        if cand in {c.lower() for c in df.columns}:
            brake_col = pick_col(df, [cand])
            break

    print("Using columns:")
    print(f"  time    -> {time_col}")
    print(f"  speed   -> {speed_col}")
    print(f"  rpm     -> {rpm_col}")
    print(f"  gear    -> {gear_col}")
    print(f"  torque  -> {torq_col}")
    print(f"  tccSlip -> {tcc_slip_col if tcc_slip_col else '<none>'}")
    if brake_col:
        print(f"  brake   -> {brake_col} (for gating)")
    else:
        print("  brake   -> <none found, no brake gating>")

    # --- Basic derived signals ---
    df = df.copy()

    # Time in seconds, relative to start
    df["time_s_raw"] = pd.to_numeric(df[time_col], errors="coerce")
    df = df[df["time_s_raw"].notna()].copy()
    df["time_s"] = df["time_s_raw"] - df["time_s_raw"].iloc[0]

    # Speed mph & m/s
    df["speed_mph"] = pd.to_numeric(df[speed_col], errors="coerce")
    df["speed_mps"] = df["speed_mph"] * 0.44704

    # RPM & gear & torque
    df["rpm"] = pd.to_numeric(df[rpm_col], errors="coerce")
    df["gear_raw"] = pd.to_numeric(df[gear_col], errors="coerce")
    df["gear_int"] = df["gear_raw"].round().astype("Int64")
    df["torque_ecm_nm"] = pd.to_numeric(df[torq_col], errors="coerce")

    # TCC slip & state
    if tcc_slip_col:
        df["tcc_slip_rpm"] = pd.to_numeric(df[tcc_slip_col], errors="coerce")
    else:
        df["tcc_slip_rpm"] = np.nan
    df["tcc_state"] = df["tcc_slip_rpm"].apply(classify_tcc_state)

    # Brake gating if available
    if brake_col:
        df["brake_kpa"] = pd.to_numeric(df[brake_col], errors="coerce")
    else:
        df["brake_kpa"] = np.nan

    # --- Compute acceleration from speed & time ---
    df = df.sort_values("time_s").reset_index(drop=True)
    dt = df["time_s"].diff()
    dv = df["speed_mps"].diff()
    df["dt"] = dt
    df["dv"] = dv
    df["accel_mps2"] = df["dv"] / df["dt"]
    # Drop rows with non-positive dt or insane accel spikes
    df.loc[(df["dt"] <= 0) | (~np.isfinite(df["accel_mps2"])), "accel_mps2"] = np.nan

    # --- Physics torque at engine (approximate, TCC locked only) ---
    # F = m * a
    df["wheel_force_N"] = MASS_KG * df["accel_mps2"]
    # Twheel = F * r
    df["wheel_torque_nm"] = df["wheel_force_N"] * TIRE_RADIUS_M

    def calc_engine_torque_phys(row):
        gear = row["gear_int"]
        if pd.isna(gear) or gear not in GEAR_RATIOS:
            return np.nan
        gr = GEAR_RATIOS[int(gear)]
        if gr <= 0 or FD <= 0:
            return np.nan
        return row["wheel_torque_nm"] / (FD * gr)

    df["torque_phys_nm"] = df.apply(calc_engine_torque_phys, axis=1)

    # --- Gating: clean samples for error calculation ---
    mask = (
        df["gear_int"].between(1, 6)
        & (df["tcc_state"] == "LOCKED")
        & df["torque_ecm_nm"].notna()
        & df["torque_phys_nm"].notna()
        & df["accel_mps2"].between(-3.0, 3.0)  # avoid wild spikes
        & df["dt"].between(0.01, 1.0)  # dt between 10ms and 1s
        & df["speed_mph"].notna()
        & df["rpm"].notna()
    )

    if brake_col:
        mask &= ((df["brake_kpa"].isna()) | (df["brake_kpa"] < 15.0))

    df_clean = df[mask].copy()

    if df_clean.empty:
        print("No clean samples after gating; check column names or conditions.")
        return

    # --- Bin by gear × RPM (100 rpm bins, centered) ---
    rpm_bin_size = 100.0  # rpm
    df_clean["rpm_bin"] = (
        (df_clean["rpm"] // rpm_bin_size) * rpm_bin_size + rpm_bin_size / 2.0
    ).astype(float)

    grp_rpm = df_clean.groupby(["gear_int", "rpm_bin"], observed=True)
    surf_rpm = grp_rpm.agg(
        n_samples=("torque_phys_nm", "size"),
        mean_phys_torque=("torque_phys_nm", "mean"),
        mean_ecm_torque=("torque_ecm_nm", "mean"),
    ).reset_index()

    surf_rpm["mean_error_nm"] = surf_rpm["mean_phys_torque"] - surf_rpm["mean_ecm_torque"]
    surf_rpm["ratio_phys_over_ecm"] = surf_rpm["mean_phys_torque"] / surf_rpm["mean_ecm_torque"]

    # --- Bin by gear × speed (1 mph bins, centered) ---
    speed_bin_size = 1.0
    df_clean["speed_bin_mph_center"] = (
        (df_clean["speed_mph"] // speed_bin_size) * speed_bin_size + speed_bin_size / 2.0
    ).astype(float)

    grp_speed = df_clean.groupby(["gear_int", "speed_bin_mph_center"], observed=True)
    surf_speed = grp_speed.agg(
        n_samples=("torque_phys_nm", "size"),
        mean_phys_torque=("torque_phys_nm", "mean"),
        mean_ecm_torque=("torque_ecm_nm", "mean"),
    ).reset_index()

    surf_speed["mean_error_nm"] = surf_speed["mean_phys_torque"] - surf_speed["mean_ecm_torque"]
    surf_speed["ratio_phys_over_ecm"] = surf_speed["mean_phys_torque"] / surf_speed["mean_ecm_torque"]

    # --- Write outputs ---
    stem = os.path.splitext(os.path.basename(log_name))[0]
    out_rpm = os.path.join(out_dir, f"torque_error_surface__gear_rpm__{stem}.csv")
    out_speed = os.path.join(out_dir, f"torque_error_surface__gear_speed__{stem}.csv")

    surf_rpm.to_csv(out_rpm, index=False)
    surf_speed.to_csv(out_speed, index=False)

    print("\nWrote torque error surfaces:")
    print(f"  gear × RPM   -> {out_rpm}")
    print(f"  gear × speed -> {out_speed}")

    # Quick console peek: top 10 highest absolute error bins
    peek = surf_rpm.copy()
    peek["abs_error"] = peek["mean_error_nm"].abs()
    peek = peek.sort_values("abs_error", ascending=False).head(10)

    print("\nTop 10 gear×RPM bins by |mean_error_nm|:")
    for _, row in peek.iterrows():
        print(
            f"  Gear {int(row['gear_int'])}, RPM~{row['rpm_bin']:.0f}: "
            f"phys={row['mean_phys_torque']:.1f} Nm, "
            f"ECM={row['mean_ecm_torque']:.1f} Nm, "
            f"err={row['mean_error_nm']:.1f} Nm "
            f"(ratio={row['ratio_phys_over_ecm']:.2f}, n={int(row['n_samples'])})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compute physics-vs-ECM torque error surfaces from a single log.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Example:

                python torque_error_surface_from_log.py \\
                    --log newlogs/burblock3.csv \\
                    --out-dir newlogs/torque_error
            """
        ),
    )
    parser.add_argument(
        "--log",
        type=str,
        default="newlogs/burblock3.csv",
        help="Path to the HP Tuners CSV log (default: newlogs/burblock3.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="newlogs/torque_error",
        help="Directory for output CSVs (default: newlogs/torque_error)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.log):
        raise SystemExit(f"log file not found: {args.log}")

    # Handle HP Tuners metadata headers
    header_line = find_header_line(args.log)
    df = pd.read_csv(args.log, skiprows=header_line)
    if not df.empty:
        # Drop the units row right after the header
        df = df.iloc[1:].reset_index(drop=True)

    # Lowercase all columns for easier matching
    df.columns = [c.strip() for c in df.columns]

    build_torque_error_surfaces(df, args.log, args.out_dir)


if __name__ == "__main__":
    main()

