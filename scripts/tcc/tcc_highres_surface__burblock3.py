#!/usr/bin/env python
"""
High-resolution TCC surface for burblock3.csv with 3×3 smoothing.

- 1 mph × 1% pedal bins
- Gears 3–6
- For each bin: time in LOCKED/PARTIAL/OPEN, lock fractions, etc.
- Then apply a 3×3 smoothing kernel (separable rolling in speed & pedal) to the
  LOCKED/PARTIAL/OPEN time fields, and recompute smoothed fractions.

Outputs:
  newlogs/output/tcc_highres/tcc_highres_surface__burblock3.csv
  newlogs/output/tcc_highres/tcc_highres_surface__burblock3__SMOOTHED.csv
"""

import os
import pathlib

import pandas as pd
from tcc_state_utils import classify_tcc_state_psi

# ---- CONFIG ----

LOG_PATH = r"newlogs/burblock5.csv"
OUT_DIR = r"newlogs/output/tcc_highres_burblock5"

SPEED_MIN = 20.0
SPEED_MAX = 80.0

GEARS_FOCUS = {3, 4, 5, 6}

# EC³-aware slip thresholds (rpm)
SLIP_LOCK_MAX = 50.0  # |slip| <= 50 rpm   => LOCKED
SLIP_OPEN_MIN = 120.0  # |slip| >= 120 rpm  => OPEN


# ---- HELPERS ----

def pick_column(df, aliases, required=True, desc=""):
    """
    Pick a column from df given a list of alias names (case-insensitive).
    """
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
    # Delegate to shared psi-aware helper (None for NaN psi)
    return classify_tcc_state_psi(abs_slip, psi)


def smooth_state_times(pivot):
    """
    Given a pivot DataFrame with columns:
      gear, speed_bin_mph, pedal_bin_pct, LOCKED, PARTIAL, OPEN
    apply a 3×3 smoothing kernel (separable) to the time-in-state fields
    for each gear, and return a DataFrame with smoothed time fields:
      LOCKED_smoothed, PARTIAL_smoothed, OPEN_smoothed
    Only bins present in the original pivot are returned.
    """
    gears = sorted(pivot["gear"].unique())
    smooth_rows = []

    for gear in gears:
        sub = pivot[pivot["gear"] == gear].copy()
        speeds = sorted(sub["speed_bin_mph"].unique())
        pedals = sorted(sub["pedal_bin_pct"].unique())

        # Build rectangular grids for each state (fill missing with 0)
        lock_grid = (
            sub.pivot(index="speed_bin_mph", columns="pedal_bin_pct", values="LOCKED")
            .reindex(index=speeds, columns=pedals)
            .fillna(0.0)
        )
        part_grid = (
            sub.pivot(index="speed_bin_mph", columns="pedal_bin_pct", values="PARTIAL")
            .reindex(index=speeds, columns=pedals)
            .fillna(0.0)
        )
        open_grid = (
            sub.pivot(index="speed_bin_mph", columns="pedal_bin_pct", values="OPEN")
            .reindex(index=speeds, columns=pedals)
            .fillna(0.0)
        )

        def smooth_grid(grid):
            g1 = grid.rolling(window=3, min_periods=1, center=True, axis=0).mean()
            g2 = g1.rolling(window=3, min_periods=1, center=True, axis=1).mean()
            return g2

        lock_s = smooth_grid(lock_grid)
        part_s = smooth_grid(part_grid)
        open_s = smooth_grid(open_grid)

        for spd in speeds:
            for ped in pedals:
                smooth_rows.append(
                    {
                        "gear": gear,
                        "speed_bin_mph": spd,
                        "pedal_bin_pct": ped,
                        "LOCKED_smoothed": float(lock_s.loc[spd, ped]),
                        "PARTIAL_smoothed": float(part_s.loc[spd, ped]),
                        "OPEN_smoothed": float(open_s.loc[spd, ped]),
                    }
                )

    smoothed = pd.DataFrame(smooth_rows)

    smoothed = (
        smoothed.merge(
            pivot[["gear", "speed_bin_mph", "pedal_bin_pct"]],
            on=["gear", "speed_bin_mph", "pedal_bin_pct"],
            how="inner",
        )
        .drop_duplicates(subset=["gear", "speed_bin_mph", "pedal_bin_pct"])
    )

    return smoothed


# ---- MAIN ----

print(f"Loading log: {LOG_PATH}")
# Find header line where the real headers start (Offset/Time row)
header_line = 0
with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f_in:
    for idx, line in enumerate(f_in):
        if line.startswith("Offset,") or line.startswith("Time,") or line.startswith("Time (s)"):
            header_line = idx
            break
df = pd.read_csv(LOG_PATH, skiprows=header_line)
# Drop the units row immediately following the header
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
engine_rpm_col = pick_column(df, ["Engine RPM (SAE)", "Engine RPM", "RPM"], False, "engine_rpm")
turbine_rpm_col = pick_column(df, ["Trans Input Shaft RPM", "Trans Turbine RPM"], False, "turbine_rpm")
pedal_col = pick_column(
    df,
    ["pedal_pct", "Accelerator Pedal Position", "Accelerator Pedal Position (%)"],
    False,
    "pedal_pct",
)
throttle_col = pick_column(df, ["throttle_pct", "Throttle Position", "Throttle Position (SAE)"], False, "throttle_pct")
tcc_line_col = pick_column(df, ["TCC Line Pressure", "TCC Apply Pressure", "TCC Line (PSI)", "TCC Pressure"], False, "tcc_line_psi")
delivered_tq_col = pick_column(df, ["Delivered Engine Torque", "Engine Torque"], False, "delivered_engine_torque")
trans_tq_col = pick_column(df, ["Trans Engine Torque"], False, "trans_engine_torque")
trans_temp_col = pick_column(df, ["Trans Fluid Temp", "Transmission Fluid Temp"], False, "trans_fluid_temp")

# Build slip
if slip_col:
    slip = pd.to_numeric(df[slip_col], errors="coerce")
elif engine_rpm_col and turbine_rpm_col:
    print("[INFO] Computing slip as Engine RPM - Turbine RPM.")
    slip = pd.to_numeric(df[engine_rpm_col], errors="coerce") - pd.to_numeric(
        df[turbine_rpm_col], errors="coerce"
    )
else:
    raise SystemExit("No slip or engine+turbine RPM available for TCC.")

work = pd.DataFrame(
    {
        "time_s": pd.to_numeric(df[time_col], errors="coerce"),
        "speed_mph": pd.to_numeric(df[speed_col], errors="coerce"),
        "gear": pd.to_numeric(df[gear_col], errors="coerce"),
        "slip_rpm": pd.to_numeric(slip, errors="coerce"),
        "tcc_line_psi": pd.to_numeric(df[tcc_line_col], errors="coerce") if tcc_line_col else 0.0,
    }
)
work["dt"] = work["time_s"].diff().fillna(0.0).clip(lower=0.0, upper=1.0)
work["abs_slip"] = work["slip_rpm"].abs()
work["tcc_state"] = work.apply(lambda r: classify_tcc_state(r["abs_slip"], r["tcc_line_psi"]), axis=1)
# Drop rows where tcc_state could not be determined (NaN psi)
work = work[work["tcc_state"].notna()].copy()

work["pedal_pct"] = pd.to_numeric(df[pedal_col], errors="coerce") if pedal_col else 0.0
work["throttle_pct"] = pd.to_numeric(df[throttle_col], errors="coerce") if throttle_col else 0.0
work["tcc_line_psi"] = pd.to_numeric(df[tcc_line_col], errors="coerce") if tcc_line_col else 0.0
work["delivered_torque"] = pd.to_numeric(df[delivered_tq_col], errors="coerce") if delivered_tq_col else 0.0
work["trans_engine_torque"] = pd.to_numeric(df[trans_tq_col], errors="coerce") if trans_tq_col else 0.0
work["trans_fluid_temp"] = (
    pd.to_numeric(df[trans_temp_col], errors="coerce") if trans_temp_col else float("nan")
)

# Filter to gears & speed range
mask = (work["gear"].isin(GEARS_FOCUS)) & (work["speed_mph"] >= SPEED_MIN) & (work["speed_mph"] <= SPEED_MAX)
work = work.loc[mask].copy().reset_index(drop=True)
print(f"Rows in gears {GEARS_FOCUS} and {SPEED_MIN}-{SPEED_MAX} mph: {len(work):,}")

# 1 mph and 1% pedal bins
work["speed_bin_mph"] = work["speed_mph"].astype(int)  # floor
work["pedal_bin_pct"] = work["pedal_pct"].clip(0, 100).astype(int)

group_cols = ["gear", "speed_bin_mph", "pedal_bin_pct", "tcc_state"]
g = work.groupby(group_cols, observed=True)

rows = []
for (gear, sb, pb, state), sub in g:
    total_time = sub["dt"].sum()
    if total_time <= 0:
        continue
    rows.append(
        {
            "gear": int(gear),
            "speed_bin_mph": int(sb),
            "pedal_bin_pct": int(pb),
            "tcc_state": state,
            "total_time_s": total_time,
            "n_samples": len(sub),
        }
    )

highres = pd.DataFrame(rows)
if highres.empty:
    raise SystemExit("No data in highres aggregation; check filters.")

# Pivot to get time per state per bin
pivot = highres.pivot_table(
    index=["gear", "speed_bin_mph", "pedal_bin_pct"],
    columns="tcc_state",
    values="total_time_s",
    aggfunc="sum",
    fill_value=0.0,
).reset_index()

# Ensure all state columns exist
for state in ["LOCKED", "PARTIAL", "OPEN"]:
    if state not in pivot.columns:
        pivot[state] = 0.0

# Compute raw totals & fractions
pivot["total_time_s"] = pivot["LOCKED"] + pivot["PARTIAL"] + pivot["OPEN"]
denom_raw = pivot["total_time_s"].where(pivot["total_time_s"] > 0, 1.0)
pivot["frac_locked"] = pivot["LOCKED"] / denom_raw
pivot["frac_partial"] = pivot["PARTIAL"] / denom_raw
pivot["frac_open"] = pivot["OPEN"] / denom_raw

# Save unsmoothed surface
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
out_path_raw = os.path.join(OUT_DIR, "tcc_highres_surface__burblock3.csv")
pivot.to_csv(out_path_raw, index=False)
print(f"Saved unsmoothed high-res TCC surface to: {out_path_raw}")

# Smooth the time-in-state fields with 3×3 kernel
print("Applying 3×3 smoothing to LOCKED/PARTIAL/OPEN time fields...")
smoothed_times = smooth_state_times(pivot)

merged = pivot.merge(
    smoothed_times,
    on=["gear", "speed_bin_mph", "pedal_bin_pct"],
    how="left",
)

# Compute smoothed totals & fractions
merged["total_time_s_smoothed"] = merged["LOCKED_smoothed"] + merged["PARTIAL_smoothed"] + merged["OPEN_smoothed"]
den_sm = merged["total_time_s_smoothed"].where(merged["total_time_s_smoothed"] > 0, 1.0)
merged["frac_locked_smoothed"] = merged["LOCKED_smoothed"] / den_sm
merged["frac_partial_smoothed"] = merged["PARTIAL_smoothed"] / den_sm
merged["frac_open_smoothed"] = merged["OPEN_smoothed"] / den_sm

out_path_smooth = os.path.join(OUT_DIR, "tcc_highres_surface__burblock3__SMOOTHED.csv")
merged.to_csv(out_path_smooth, index=False)
print(f"Saved SMOOTHED high-res TCC surface to: {out_path_smooth}")

print("Example (gear 5, 40-50 mph, pedal 5-25%):")
with pd.option_context("display.max_rows", 50):
    print(
        merged[
            (merged["gear"] == 5)
            & (merged["speed_bin_mph"].between(40, 50))
            & (merged["pedal_bin_pct"].between(5, 25))
        ][
            [
                "gear",
                "speed_bin_mph",
                "pedal_bin_pct",
                "frac_locked",
                "frac_locked_smoothed",
                "frac_partial",
                "frac_partial_smoothed",
                "frac_open",
                "frac_open_smoothed",
                "total_time_s",
            ]
        ]
    )
