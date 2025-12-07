#!/usr/bin/env python
"""
TCC trigger analysis for burblock3.csv in 4th & 5th gears, 38–52 mph.

Focus:
- Find every TCC state change (LOCKED/PARTIAL/OPEN) in 4th/5th between 38–52 mph.
- For each transition, capture context 1s before and 1s after:
  pedal, throttle, torque, brake, TCS, temp, slip, psi, etc.

Outputs (to newlogs/output/tcc_burblock3_triggers_4_5_38_52/):

1) tcc_4_5_38_52_transitions__burblock3.csv
   One row per TCC state transition with "before/after" context and a rough "reason" tag.

2) tcc_4_5_38_52_state_vs_pedal__burblock3.csv
   Gear × TCC state × pedal-bin summary (how state depends on pedal in that speed window).
"""

import math
import os
import pathlib
import sys

import pandas as pd
from tcc_state_utils import classify_tcc_state_psi

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

LOG_PATH = r"newlogs/burblock5.csv"
OUT_DIR = r"newlogs/output/tcc_burblock5_triggers_4_5_38_52"

SPEED_MIN = 38.0
SPEED_MAX = 52.0

GEARS_FOCUS = {4, 5}

# TCC thresholds
SLIP_LOCK_MAX = 50.0  # |slip| <= 50 rpm  => LOCKED

# Brake threshold (kPa-ish) to count as "braking"
BRAKE_ON_KPA = 15.0

# Pedal rate thresholds (% per second) to classify tip-in vs lift
PEDAL_RATE_TIPIN = 5.0
PEDAL_RATE_LIFT = -5.0

# Pedal bins for state-vs-pedal summary
PEDAL_BIN_EDGES = [0, 5, 10, 15, 20, 30, 40, 60, 100]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def pick_column(df, aliases, required=True, desc=""):
    """
    Pick a column by trying a list of aliases (case-insensitive).
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for name in aliases:
        if name.lower() in cols_lower:
            col = cols_lower[name.lower()]
            print(f"[OK] Using column '{col}' for {desc or name}")
            return col
    if required:
        raise ValueError(
            f"Required column for {desc or aliases[0]} not found. Tried aliases: {aliases}"
        )
    print(f"[WARN] Optional column for {desc or aliases[0]} not found. Tried: {aliases}")
    return None


def classify_tcc_state(slip_abs: float, psi: float) -> str:
    # psi-aware helper
    return classify_tcc_state_psi(slip_abs, psi)


# ------------------------------------------------------------
# Load log
# ------------------------------------------------------------

print(f"Loading log: {LOG_PATH}")

# HP Tuners exports often have a few metadata lines before the real header.
header_line = 0
with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f_in:
    for idx, line in enumerate(f_in):
        # HP Tuners CSV has a line starting with "Offset," before the unit row.
        if line.startswith("Offset,") or line.startswith("Time,") or line.startswith("Time (s),"):
            header_line = idx
            break

df = pd.read_csv(LOG_PATH, skiprows=header_line)
# The row immediately after the header is units; drop it.
if not df.empty:
    df = df.iloc[1:].reset_index(drop=True)
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns (header line {header_line}).")


# ------------------------------------------------------------
# Pick columns
# ------------------------------------------------------------

time_col = pick_column(
    df,
    ["time_s", "Time (s)", "Elapsed Time", "Elapsed Time (s)", "Offset", "Time"],
    required=True,
    desc="time_s",
)

speed_col = pick_column(
    df,
    ["speed_mph", "Vehicle Speed (SAE)", "Vehicle Speed", "Vehicle Speed (mph)", "MPH"],
    required=True,
    desc="speed_mph",
)

gear_col = pick_column(
    df,
    [
        "gear_actual__canon",
        "Trans Current Gear",
        "Transmission Current Gear",
        "Current Gear",
        "Trans Current Gear 1",
        "Trans Current Gear.1",
        "Gear",
    ],
    required=True,
    desc="gear_actual",
)

slip_col = pick_column(
    df,
    ["tcc_slip_fused", "TCC Slip", "TCC Slip RPM", "Torque Converter Slip"],
    required=False,
    desc="tcc_slip_rpm",
)

tcc_desired_slip_col = pick_column(
    df,
    ["TCC Desired Slip", "TCC Slip Desired"],
    required=False,
    desc="tcc_desired_slip",
)

tcc_line_col = pick_column(
    df,
    ["TCC Line Pressure", "TCC Apply Pressure", "TCC Line (PSI)", "TCC Pressure"],
    required=False,
    desc="tcc_line_psi",
)

engine_rpm_col = pick_column(
    df,
    ["Engine RPM (SAE)", "Engine RPM", "RPM", "Engine Speed"],
    required=False,
    desc="engine_rpm",
)

turbine_rpm_col = pick_column(
    df,
    ["Trans Input Shaft RPM", "Trans Turbine RPM", "Turbine RPM", "Trans Input Speed"],
    required=False,
    desc="turbine_rpm",
)

pedal_col = pick_column(
    df,
    [
        "pedal_pct",
        "Accelerator Pedal Position",
        "Accelerator Pedal Position (%)",
        "Accel Pedal Position",
        "Accel Pedal %",
    ],
    required=False,
    desc="pedal_pct",
)

throttle_col = pick_column(
    df,
    ["throttle_pct", "Throttle Position", "Throttle Position (SAE)", "Throttle Position (%)"],
    required=False,
    desc="throttle_pct",
)

driver_axle_tq_req_col = pick_column(
    df,
    ["Driver Final Axle Torque Req", "Driver Pedal Axle Torque Req"],
    required=False,
    desc="driver_axle_tq_req",
)

delivered_tq_col = pick_column(
    df,
    ["Delivered Engine Torque", "Engine Torque"],
    required=False,
    desc="delivered_engine_torque",
)

trans_engine_tq_col = pick_column(
    df,
    ["Trans Engine Torque"],
    required=False,
    desc="trans_engine_torque",
)

brake_col = pick_column(
    df,
    ["Brake Pressure"],
    required=False,
    desc="brake_pressure",
)

tcs_req_col = pick_column(
    df,
    ["TCS Request"],
    required=False,
    desc="tcs_request",
)

tcs_sys_col = pick_column(
    df,
    ["Traction Control System"],
    required=False,
    desc="tcs_system",
)

tcs_des_tq_col = pick_column(
    df,
    ["TCS Desired Engine Torque", "Traction Control Desired Torque"],
    required=False,
    desc="tcs_desired_torque",
)

tcs_eng_tq_req_col = pick_column(
    df,
    ["TCS Engine Torque Req", "Traction Control Torque"],
    required=False,
    desc="tcs_engine_torque_req",
)

trans_temp_col = pick_column(
    df,
    ["Trans Fluid Temp", "Transmission Fluid Temp"],
    required=False,
    desc="trans_fluid_temp",
)


# ------------------------------------------------------------
# Build working frame
# ------------------------------------------------------------

# TCC slip: direct if present, else compute from engine - turbine
if slip_col is not None:
    slip_series = pd.to_numeric(df[slip_col], errors="coerce")
elif engine_rpm_col and turbine_rpm_col:
    print("[INFO] Computing TCC slip as Engine RPM − Turbine RPM.")
    slip_series = pd.to_numeric(df[engine_rpm_col], errors="coerce") - pd.to_numeric(
        df[turbine_rpm_col], errors="coerce"
    )
else:
    raise SystemExit("No usable TCC slip or engine+turbine RPM columns found.")

work = pd.DataFrame(
    {
        "time_s": pd.to_numeric(df[time_col], errors="coerce"),
        "speed_mph": pd.to_numeric(df[speed_col], errors="coerce"),
        "gear": pd.to_numeric(df[gear_col], errors="coerce"),
        "slip_rpm": pd.to_numeric(slip_series, errors="coerce"),
    }
)

# Add optional columns if present
work["tcc_line_psi"] = pd.to_numeric(df[tcc_line_col], errors="coerce") if tcc_line_col else 0.0
if tcc_desired_slip_col:
    work["tcc_desired_slip"] = pd.to_numeric(df[tcc_desired_slip_col], errors="coerce")
else:
    work["tcc_desired_slip"] = float("nan")

if pedal_col:
    work["pedal_pct"] = pd.to_numeric(df[pedal_col], errors="coerce")
else:
    work["pedal_pct"] = float("nan")

if throttle_col:
    work["throttle_pct"] = pd.to_numeric(df[throttle_col], errors="coerce")
else:
    work["throttle_pct"] = float("nan")

if driver_axle_tq_req_col:
    work["driver_axle_tq_req"] = pd.to_numeric(df[driver_axle_tq_req_col], errors="coerce")
else:
    work["driver_axle_tq_req"] = float("nan")

if delivered_tq_col:
    work["delivered_engine_torque"] = pd.to_numeric(df[delivered_tq_col], errors="coerce")
else:
    work["delivered_engine_torque"] = float("nan")

if trans_engine_tq_col:
    work["trans_engine_torque"] = pd.to_numeric(df[trans_engine_tq_col], errors="coerce")
else:
    work["trans_engine_torque"] = float("nan")

if brake_col:
    work["brake_pressure"] = pd.to_numeric(df[brake_col], errors="coerce")
else:
    work["brake_pressure"] = 0.0

if tcs_req_col:
    work["tcs_request"] = df[tcs_req_col]
else:
    work["tcs_request"] = 0

if tcs_sys_col:
    work["tcs_system"] = df[tcs_sys_col]
else:
    work["tcs_system"] = 0

if tcs_des_tq_col:
    work["tcs_desired_torque"] = pd.to_numeric(df[tcs_des_tq_col], errors="coerce")
else:
    work["tcs_desired_torque"] = float("nan")

if tcs_eng_tq_req_col:
    work["tcs_engine_torque_req"] = pd.to_numeric(df[tcs_eng_tq_req_col], errors="coerce")
else:
    work["tcs_engine_torque_req"] = float("nan")

if trans_temp_col:
    work["trans_fluid_temp"] = pd.to_numeric(df[trans_temp_col], errors="coerce")
else:
    work["trans_fluid_temp"] = float("nan")

# Clean + sort
work = work.dropna(subset=["time_s", "speed_mph", "gear", "slip_rpm"]).copy()
work = work.sort_values("time_s").reset_index(drop=True)

dt = work["time_s"].diff().fillna(0.0).clip(lower=0.0, upper=1.0)
work["dt"] = dt
work["abs_slip_rpm"] = work["slip_rpm"].abs()
work["tcc_state"] = work.apply(lambda r: classify_tcc_state(r["abs_slip_rpm"], r["tcc_line_psi"]), axis=1)
# Drop rows where tcc_state could not be determined (NaN psi)
work = work[work["tcc_state"].notna()].copy()

print("Global TCC state distribution:")
print(work["tcc_state"].value_counts(dropna=False))


# ------------------------------------------------------------
# Focus subset: gears 4 & 5, speed 38–52 mph
# ------------------------------------------------------------

in_speed = (work["speed_mph"] >= SPEED_MIN) & (work["speed_mph"] <= SPEED_MAX)
in_gear = work["gear"].isin(GEARS_FOCUS)
focus = work.loc[in_speed & in_gear].copy().reset_index(drop=True)

print(f"\nRows in 4th/5th and {SPEED_MIN}-{SPEED_MAX} mph: {len(focus):,}")
if focus.empty:
    raise SystemExit("No rows in 4th/5th within the requested speed band.")

# Estimate pedal rate (%/s)
if focus["pedal_pct"].notna().any():
    dpedal = focus["pedal_pct"].diff().fillna(0.0)
    focus["pedal_rate"] = (dpedal / focus["dt"].replace(0, float("nan"))).fillna(0.0)
else:
    focus["pedal_rate"] = 0.0


# ------------------------------------------------------------
# 1) State-vs-pedal summary
# ------------------------------------------------------------

if focus["pedal_pct"].notna().any():
    pedal_bins = PEDAL_BIN_EDGES
    focus["pedal_bin"] = pd.cut(
        focus["pedal_pct"],
        bins=pedal_bins,
        right=False,
        include_lowest=True,
    )
else:
    focus["pedal_bin"] = "[all]"

group_cols = ["gear", "tcc_state", "pedal_bin"]
g = focus.groupby(group_cols)

rows = []
for (gear, state, pbin), sub in g:
    total_time = sub["dt"].sum()
    if total_time <= 0:
        continue
    rows.append(
        {
            "gear": int(gear),
            "tcc_state": state,
            "pedal_bin": str(pbin),
            "total_time_s": total_time,
            "n_samples": len(sub),
            "mean_speed_mph": sub["speed_mph"].mean(),
            "mean_pedal_pct": sub["pedal_pct"].mean(),
            "mean_throttle_pct": sub["throttle_pct"].mean(),
            "mean_slip_rpm": sub["abs_slip_rpm"].mean(),
            "mean_tcc_line_psi": sub["tcc_line_psi"].mean(),
        }
    )

state_vs_pedal_df = pd.DataFrame(rows).sort_values(
    ["gear", "tcc_state", "pedal_bin"]
).reset_index(drop=True)


# ------------------------------------------------------------
# 2) TCC state transitions with context windows
# ------------------------------------------------------------

transitions = []


def window_stats(sub):
    if sub.empty:
        return {}
    return {
        "mean_speed_mph": sub["speed_mph"].mean(),
        "mean_pedal_pct": sub["pedal_pct"].mean(),
        "mean_pedal_rate": sub["pedal_rate"].mean(),
        "mean_throttle_pct": sub["throttle_pct"].mean(),
        "mean_slip_rpm": sub["abs_slip_rpm"].mean(),
        "mean_tcc_line_psi": sub["tcc_line_psi"].mean(),
        "mean_brake_pressure": sub["brake_pressure"].mean(),
        "max_brake_pressure": sub["brake_pressure"].max(),
        "mean_delivered_engine_torque": sub["delivered_engine_torque"].mean(),
        "mean_trans_engine_torque": sub["trans_engine_torque"].mean(),
        "mean_driver_axle_tq_req": sub["driver_axle_tq_req"].mean(),
        "mean_tcs_desired_torque": sub["tcs_desired_torque"].mean(),
        "mean_tcs_engine_torque_req": sub["tcs_engine_torque_req"].mean(),
        "any_tcs_request": int((sub["tcs_request"] != 0).any()),
    }


for idx in range(1, len(focus)):
    prev = focus.iloc[idx - 1]
    curr = focus.iloc[idx]

    if curr["tcc_state"] == prev["tcc_state"]:
        continue
    if curr["gear"] != prev["gear"]:
        continue

    time_tr = curr["time_s"]
    gear = int(curr["gear"])
    old_state = prev["tcc_state"]
    new_state = curr["tcc_state"]

    # 1-second windows before and after the transition
    pre_mask = (focus["time_s"] >= time_tr - 1.0) & (focus["time_s"] < time_tr)
    post_mask = (focus["time_s"] > time_tr) & (focus["time_s"] <= time_tr + 1.0)
    pre = focus.loc[pre_mask]
    post = focus.loc[post_mask]

    pre_stats = window_stats(pre)
    post_stats = window_stats(post)

    # Reason heuristic
    reason = "OTHER"
    brake_pre = pre_stats.get("max_brake_pressure", 0.0) if pre_stats else 0.0
    brake_post = post_stats.get("max_brake_pressure", 0.0) if post_stats else 0.0
    any_tcs = pre_stats.get("any_tcs_request", 0) or post_stats.get("any_tcs_request", 0)

    pedal_rate_pre = pre_stats.get("mean_pedal_rate", 0.0)
    pedal_rate_post = post_stats.get("mean_pedal_rate", 0.0)
    pedal_rate = (pedal_rate_pre + pedal_rate_post) / 2.0

    if brake_pre > BRAKE_ON_KPA or brake_post > BRAKE_ON_KPA:
        reason = "BRAKE"
    elif any_tcs:
        reason = "TRACTION"
    elif pedal_rate >= PEDAL_RATE_TIPIN:
        reason = "TIP_IN"
    elif pedal_rate <= PEDAL_RATE_LIFT:
        reason = "LIFT"

    row = {
        "gear": gear,
        "old_tcc_state": old_state,
        "new_tcc_state": new_state,
        "time_s": time_tr,
        "speed_mph": curr["speed_mph"],
        "slip_rpm": curr["slip_rpm"],
        "abs_slip_rpm": curr["abs_slip_rpm"],
        "tcc_line_psi": curr["tcc_line_psi"],
        "tcc_desired_slip": curr["tcc_desired_slip"],
        "brake_pressure": curr["brake_pressure"],
        "pedal_pct": curr["pedal_pct"],
        "throttle_pct": curr["throttle_pct"],
        "delivered_engine_torque": curr["delivered_engine_torque"],
        "trans_engine_torque": curr["trans_engine_torque"],
        "driver_axle_tq_req": curr["driver_axle_tq_req"],
        "tcs_request": curr["tcs_request"],
        "tcs_system": curr["tcs_system"],
        "tcs_desired_torque": curr["tcs_desired_torque"],
        "tcs_engine_torque_req": curr["tcs_engine_torque_req"],
        "trans_fluid_temp": curr["trans_fluid_temp"],
        "reason": reason,
    }

    # Prefix pre/post stats
    for k, v in pre_stats.items():
        row["pre_" + k] = v
    for k, v in post_stats.items():
        row["post_" + k] = v

    transitions.append(row)

transitions_df = (
    pd.DataFrame(transitions)
    .sort_values(["gear", "time_s"])
    .reset_index(drop=True)
)

print(f"\nFound {len(transitions_df):,} TCC state transitions in 4th/5th, {SPEED_MIN}-{SPEED_MAX} mph.")


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

transitions_path = os.path.join(OUT_DIR, "tcc_4_5_38_52_transitions__burblock3.csv")
state_vs_pedal_path = os.path.join(OUT_DIR, "tcc_4_5_38_52_state_vs_pedal__burblock3.csv")

transitions_df.to_csv(transitions_path, index=False)
state_vs_pedal_df.to_csv(state_vs_pedal_path, index=False)

print(f"Saved transitions to: {transitions_path}")
print(f"Saved state-vs-pedal summary to: {state_vs_pedal_path}")

# Quick console summary
if not transitions_df.empty:
    summary = (
        transitions_df.groupby(["gear", "old_tcc_state", "new_tcc_state", "reason"])
        .size()
        .reset_index(name="n_events")
        .sort_values(["gear", "n_events"], ascending=[True, False])
    )
    print("\n=== Transition summary by gear / old->new / reason ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(summary)
else:
    print("No transitions detected in the target window.")
