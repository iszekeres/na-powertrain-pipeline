#!/usr/bin/env python
"""
Light-accel decision surface for burblock3.csv

Goal:
  For episodes where the converter is LOCKED and the driver rolls into the pedal
  lightly, figure out what the transmission tends to do over the next ~2 seconds:

    - Stay in the same gear and remain LOCKED  ("STAY_LOCKED")
    - Stay in the same gear but UNLOCK the TCC ("UNLOCK")
    - Downshift to a lower gear ("DOWNSHIFT")
    - Upshift to a higher gear ("UPSHIFT")

We aggregate these outcomes on a 1 mph × 1% pedal grid for each starting gear,
so we can see where the truck tends to unlock vs. downshift vs. stay locked
during light acceleration.

Input:
  newlogs/burblock3.csv

Output:
  newlogs/output/tcc_decisions/tcc_light_accel_decision_map__burblock3.csv
"""

import os
import pathlib

import numpy as np
import pandas as pd
from tcc_state_utils import classify_tcc_state_psi

LOG_PATH = r"newlogs/burblock5.csv"
OUT_DIR = r"newlogs/output/tcc_decisions_burblock5"

# Slip thresholds (rpm)
SLIP_LOCK_MAX = 50.0  # |slip| <= 50 rpm -> LOCKED

# Episode window (seconds)
WINDOW_SEC = 2.0

# Light-accel pedal range
PEDAL_MIN = 3.0
# Use < 40 to avoid heavy tip-in; 40 is inclusive upper bound via <=
PEDAL_MAX = 40.0

# Speed range of interest
SPEED_MIN = 20.0
SPEED_MAX = 80.0


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
    # psi-aware: NaN -> None; psi==0 -> OPEN; psi>0 then LOCKED<=50 else PARTIAL
    return classify_tcc_state_psi(abs_slip, psi)


def main():
    print(f"Loading log: {LOG_PATH}")
    # Find header line in HP Tuners CSV
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

    time_col = pick_column(
        df, ["time_s", "Elapsed Time", "Elapsed Time (s)", "Offset", "Time"], True, "time_s"
    )
    speed_col = pick_column(
        df, ["speed_mph", "Vehicle Speed (SAE)", "Vehicle Speed (mph)", "Vehicle Speed"], True, "speed_mph"
    )
    gear_col = pick_column(
        df,
        [
            "gear_actual__canon",
            "Trans Current Gear",
            "Trans Current Gear 1",
            "Trans Current Gear.1",
            "Current Gear",
            "Gear",
        ],
        True,
        "gear_actual",
    )
    slip_col = pick_column(
        df, ["tcc_slip_fused", "TCC Slip", "TCC Slip RPM", "Torque Converter Slip"], False, "tcc_slip_rpm"
    )
    engine_rpm_col = pick_column(
        df, ["Engine RPM (SAE)", "Engine RPM", "RPM"], False, "engine_rpm"
    )
    turbine_rpm_col = pick_column(
        df, ["Trans Input Shaft RPM", "Trans Turbine RPM"], False, "turbine_rpm"
    )
    pedal_col = pick_column(
        df,
        ["pedal_pct", "Accelerator Pedal Position", "Accelerator Pedal Position (%)"],
        False,
        "pedal_pct",
    )
    tcc_line_col = pick_column(
        df,
        ["TCC Line Pressure", "TCC Apply Pressure", "TCC Line (PSI)", "TCC Pressure"],
        False,
        "tcc_line_psi",
    )

    # Build slip signal
    if slip_col:
        slip = pd.to_numeric(df[slip_col], errors="coerce")
    elif engine_rpm_col and turbine_rpm_col:
        print("[INFO] Computing slip as Engine RPM - Turbine RPM.")
        slip = pd.to_numeric(df[engine_rpm_col], errors="coerce") - pd.to_numeric(
            df[turbine_rpm_col], errors="coerce"
        )
    else:
        raise SystemExit("No slip or engine+turbine RPM available for TCC.")

    # Core working frame
    work = pd.DataFrame(
        {
            "time_s": pd.to_numeric(df[time_col], errors="coerce"),
            "speed_mph": pd.to_numeric(df[speed_col], errors="coerce"),
            "gear": pd.to_numeric(df[gear_col], errors="coerce"),
            "slip_rpm": pd.to_numeric(slip, errors="coerce"),
            "tcc_line_psi": pd.to_numeric(df[tcc_line_col], errors="coerce") if tcc_line_col else np.nan,
        }
    )
    work["dt"] = work["time_s"].diff().fillna(0.0).clip(lower=0.0, upper=1.0)
    work["abs_slip"] = work["slip_rpm"].abs()
    work["tcc_state"] = work.apply(lambda r: classify_tcc_state(r["abs_slip"], r["tcc_line_psi"]), axis=1)
    work = work[work["tcc_state"].notna()].copy()

    if pedal_col:
        work["pedal_pct"] = pd.to_numeric(df[pedal_col], errors="coerce")
    else:
        work["pedal_pct"] = 0.0

    # Restrict to forward gears and speed range
    work = work[
        (work["gear"].between(1, 6))
        & (work["speed_mph"].between(SPEED_MIN, SPEED_MAX))
    ].copy()
    work.reset_index(drop=True, inplace=True)
    print(f"Rows after gear/speed filter: {len(work):,}")

    times = work["time_s"].values
    n_rows = len(work)

    decisions = []

    for i in range(n_rows):
        row = work.iloc[i]
        gear0 = int(row["gear"])
        t0 = float(row["time_s"])
        speed0 = float(row["speed_mph"])
        pedal0 = float(row["pedal_pct"])
        state0 = row["tcc_state"]

        # Only start from LOCKED episodes in light-accel band
        if state0 != "LOCKED":
            continue
        if not (PEDAL_MIN <= pedal0 <= PEDAL_MAX):
            continue

        # Find window endpoint index j where time_s <= t0 + WINDOW_SEC
        t_end = t0 + WINDOW_SEC
        j = i + 1
        while j < n_rows and times[j] <= t_end:
            j += 1

        if j <= i + 1:
            continue

        window = work.iloc[i + 1 : j]
        if window.empty:
            continue

        # Require some net acceleration over the window (at least +1 mph)
        speed_end = float(window["speed_mph"].iloc[-1])
        if speed_end < speed0 + 1.0:
            continue

        gear_series = window["gear"].astype(int)
        state_series = window["tcc_state"]

        # Outcome classification (priority order)
        outcome = "OTHER"

        # Downshift: any gear < gear0
        if (gear_series < gear0).any():
            outcome = "DOWNSHIFT"
        # Upshift: any gear > gear0
        elif (gear_series > gear0).any():
            outcome = "UPSHIFT"
        else:
            # Same gear throughout
            if (state_series != "LOCKED").any():
                outcome = "UNLOCK"
            else:
                outcome = "STAY_LOCKED"

        if outcome == "OTHER":
            continue

        speed_bin = int(speed0)
        pedal_bin = int(max(0, min(100, pedal0)))

        decisions.append(
            {
                "gear_start": gear0,
                "speed_bin_mph": speed_bin,
                "pedal_bin_pct": pedal_bin,
                "outcome": outcome,
            }
        )

    if not decisions:
        raise SystemExit("No qualifying light-accel episodes found.")

    dec_df = pd.DataFrame(decisions)
    print(f"Captured {len(dec_df):,} decision episodes.")

    # Aggregate to 1 mph × 1% pedal grid
    agg = (
        dec_df.groupby(["gear_start", "speed_bin_mph", "pedal_bin_pct", "outcome"])
        .size()
        .reset_index(name="count")
    )

    pivot = (
        agg.pivot_table(
            index=["gear_start", "speed_bin_mph", "pedal_bin_pct"],
            columns="outcome",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    for col in ["STAY_LOCKED", "UNLOCK", "DOWNSHIFT", "UPSHIFT"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["total_episodes"] = (
        pivot["STAY_LOCKED"] + pivot["UNLOCK"] + pivot["DOWNSHIFT"] + pivot["UPSHIFT"]
    ).astype(float)

    denom = pivot["total_episodes"].where(pivot["total_episodes"] > 0, 1.0)
    pivot["frac_STAY_LOCKED"] = pivot["STAY_LOCKED"] / denom
    pivot["frac_UNLOCK"] = pivot["UNLOCK"] / denom
    pivot["frac_DOWNSHIFT"] = pivot["DOWNSHIFT"] / denom
    pivot["frac_UPSHIFT"] = pivot["UPSHIFT"] / denom

    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "tcc_light_accel_decision_map__burblock3.csv")
    pivot.to_csv(out_path, index=False)
    print(f"Saved decision surface to: {out_path}")

    # Quick peek at the busiest bins for gears 4–6
    with pd.option_context("display.max_rows", 40):
        print(
            pivot[
                (pivot["gear_start"].between(4, 6))
                & (pivot["speed_bin_mph"].between(35, 70))
                & (pivot["pedal_bin_pct"].between(5, 30))
            ]
            .sort_values("total_episodes", ascending=False)
            .head(40)
        )


if __name__ == "__main__":
    main()
