import argparse
import glob
import os

import numpy as np
import pandas as pd


TPS_BINS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]
ALLOWED_KICKDOWN_ROWS = {"3 -> 2 Shift", "4 -> 3 Shift", "5 -> 4 Shift", "6 -> 5 Shift"}
ALLOWED_KICKDOWN_BINS = {19, 25, 31, 37, 44, 50, 56, 62, 69, 75}

# Tahoe-specific speed ranges for kickdown rows
KICKDOWN_SPEED_RANGES = {
    "3 -> 2 Shift": (20.0, 55.0),
    "4 -> 3 Shift": (28.0, 65.0),
    "5 -> 4 Shift": (35.0, 80.0),
    "6 -> 5 Shift": (45.0, 90.0),
}

# Comfort DOWN baseline path (used to compare median kickdown mph)
DEFAULT_BASELINE_DOWN = os.path.join(
    "newlogs", "output", "01_tables", "shift", "SHIFT_TABLES__DOWN__Throttle17__COMFORT.tsv"
)

# Tahoe v1 gates
MIN_TPS = 28.0          # minimum TPS peak OR ...
MIN_PEDAL = 25.0        # ... minimum pedal peak
MIN_DTPS = 5.0          # minimum dTPS/dt peak (%/s)
MIN_SPEED_GLOBAL = 20.0 # global floor for kickdown consideration
BRAKE_MAX = 0.5         # brake veto threshold (for 0/1 brake flags)

# Aggregation thresholds
MIN_COUNT = 2
STD_MAX = 8.0
DELTA_MPH = 0.3
BASE_GAP_MIN = 0.2  # mph


def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def floor_tps_bin(tps: float) -> int:
    # Floor TPS to the nearest lower standard bin.
    for b in reversed(TPS_BINS):
        if tps >= b:
            return b
    return TPS_BINS[0]


def ensure_canon(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Canonicalize key columns (time, speed, tps, pedal, gear, brake).
    Works on BESTINTERP logs (time_s__canon, speed_mph__canon, etc.)
    and falls back to reasonable raw names if needed.
    """
    col_time = pick_col(df, ["time_s__canon", "time_s", "Time", "Time (s)", "OFFSET", "offset"])
    col_speed = pick_col(df, ["speed_mph__canon", "Vehicle Speed (SAE)", "Vehicle Speed", "Speed (mph)"])
    col_tps = pick_col(df, ["throttle_pct__canon", "Throttle Position", "Throttle Position (SAE)", "Throttle Position (%)"])
    col_pedal = pick_col(df, ["pedal_pct__canon", "Accelerator Pedal Position", "Accelerator Pedal Position (SAE)"])
    col_gear = pick_col(df, ["gear_actual__canon", "Trans Current Gear", "Trans Current Gear (SAE)"])
    col_brake = pick_col(df, ["brake__canon", "brake", "Brake", "Brake (on/off)"])

    missing = [name for name, col in [
        ("time", col_time),
        ("speed", col_speed),
        ("tps", col_tps),
        ("gear", col_gear),
    ] if col is None]

    if missing:
        raise RuntimeError(f"{os.path.basename(path)}: missing required column(s): {missing}")

    out = df.copy()

    out["time"] = pd.to_numeric(out[col_time], errors="coerce")
    out["speed_mph"] = pd.to_numeric(out[col_speed], errors="coerce")
    out["tps"] = pd.to_numeric(out[col_tps], errors="coerce")
    out["gear"] = pd.to_numeric(out[col_gear], errors="coerce")

    if col_pedal is not None:
        out["pedal"] = pd.to_numeric(out[col_pedal], errors="coerce")
    else:
        out["pedal"] = np.nan

    if col_brake is not None:
        out["brake"] = pd.to_numeric(out[col_brake], errors="coerce")
    else:
        out["brake"] = 0.0

    # Drop rows where the canonical quartet is missing
    out = out.dropna(subset=["time", "speed_mph", "tps", "gear"]).copy()

    # Snap gear to integer 1–6
    gear_int = out["gear"].round().astype("Int64")
    out = out.assign(gear_int=gear_int)
    out = out.dropna(subset=["gear_int"]).copy()

    # Compute dTPS/dt
    out["dt"] = out["time"].diff()
    out["dthr"] = out["tps"].diff()
    out["dthr_dt"] = 0.0
    mask_dt = out["dt"] > 0
    out.loc[mask_dt, "dthr_dt"] = out.loc[mask_dt, "dthr"] / out.loc[mask_dt, "dt"]

    return out


def detect_kickdown_events(df: pd.DataFrame, filename: str) -> list:
    """
    Scan a canonical, dense BESTINTERP dataframe for downshifts and
    select Tahoe-style kickdowns based on TPS/pedal/dTPS gates and
    per-row speed bands.
    """
    events = []

    # Previous gear
    df = df.copy()
    df["gear_prev"] = df["gear_int"].shift(1)
    transitions = df[
        df["gear_prev"].notna()
        & df["gear_int"].notna()
        & (df["gear_prev"] != df["gear_int"])
    ]

    for idx, row in transitions.iterrows():
        g_before = int(row["gear_prev"])
        g_after = int(row["gear_int"])

        # Only consider downshifts (previous gear > current gear)
        if g_before <= g_after:
            continue

        if not (1 <= g_before <= 6 and 1 <= g_after <= 6):
            continue

        # Build row label like "4 -> 3 Shift"
        row_label = f"{g_before} -> {g_after} Shift"

        if row_label not in KICKDOWN_SPEED_RANGES:
            # Skip 2 -> 1 etc. here; 2 -> 1 is STOPGO territory.
            continue

        speed = float(row["speed_mph"])
        if speed < MIN_SPEED_GLOBAL:
            continue

        min_spd, max_spd = KICKDOWN_SPEED_RANGES[row_label]
        if not (min_spd <= speed <= max_spd):
            continue

        t0 = float(row["time"])
        # Window around the shift: -0.25s to +0.75s
        t_start = t0 - 0.25
        t_end = t0 + 0.75

        window = df[(df["time"] >= t_start) & (df["time"] <= t_end)]
        if window.empty:
            continue

        tps_peak = float(window["tps"].max())
        pedal_peak = float(window["pedal"].max()) if "pedal" in window.columns else np.nan
        dthr_peak = float(window["dthr_dt"].max())

        # Brake veto
        brake_max = float(window["brake"].max()) if "brake" in window.columns else 0.0
        if brake_max > BRAKE_MAX:
            continue

        # Demand gate: TPS OR pedal
        demand_ok = (tps_peak >= MIN_TPS) or (not np.isnan(pedal_peak) and pedal_peak >= MIN_PEDAL)
        if not demand_ok:
            continue

        # Ramp gate
        if dthr_peak < MIN_DTPS:
            continue

        # Use TPS at the actual transition point for binning
        tps_event = float(row["tps"])
        tps_bin = floor_tps_bin(tps_event)

        event = {
            "file": os.path.basename(filename),
            "time_s": t0,
            "speed_mph": speed,
            "tps": tps_event,
            "tps_peak": tps_peak,
            "pedal": pedal_peak,
            "gear_before": g_before,
            "gear_after": g_after,
            "row": row_label,
            "tps_bin": tps_bin,
            "dthr_dt_peak": dthr_peak,
        }
        events.append(event)

    return events


def load_baseline_down(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline DOWN table not found at: {path}")
    base = pd.read_csv(path, sep="\t")
    return base


def build_delta_from_events(events_df: pd.DataFrame, baseline_down: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Tahoe kickdown events into a SHIFT DOWN delta grid.
    Only 3->2, 4->3, 5->4, 6->5 rows and mid/high TPS bins are allowed
    to receive +0.3 mph bumps, and only when median mph is > baseline+0.5
    with at least MIN_COUNT events and limited spread.
    """
    delta = baseline_down.copy()

    # Ensure numeric columns are present and initialized to 0.0
    tps_cols = [c for c in delta.columns if c != "mph"]
    for c in tps_cols:
        delta[c] = 0.0

    if events_df.empty:
        return delta

    # Aggregate by (row, tps_bin)
    grouped = events_df.groupby(["row", "tps_bin"])["speed_mph"].agg(
        count="count", median="median", std="std"
    ).reset_index()

    for _, r in grouped.iterrows():
        row_label = r["row"]
        tps_bin = int(r["tps_bin"])
        count = int(r["count"])
        median_mph = float(r["median"])
        std_mph = float(r["std"]) if not np.isnan(r["std"]) else float("inf")

        if row_label not in ALLOWED_KICKDOWN_ROWS:
            continue
        if tps_bin not in ALLOWED_KICKDOWN_BINS:
            continue
        if count < MIN_COUNT:
            continue
        if std_mph > STD_MAX:
            continue

        # Look up baseline mph for this row/TPS bin
        col_name = str(tps_bin)
        if col_name not in delta.columns:
            continue

        row_mask = delta["mph"] == row_label
        if not row_mask.any():
            continue

        base_vals = baseline_down.loc[row_mask, col_name]
        if base_vals.empty:
            continue

        base_val = float(base_vals.iloc[0])
        gap = median_mph - base_val
        if gap < BASE_GAP_MIN:
            continue
        delta.loc[row_mask, col_name] = min(DELTA_MPH, gap)

    return delta


def main():
    parser = argparse.ArgumentParser(description="Tahoe-specific KICKDOWN pass (BESTINTERP-aware).")
    parser.add_argument("--logs-glob", required=True, help="Glob for BESTINTERP CSVs.")
    parser.add_argument("--out-dir", required=True, help="Output directory for DELTA + events.")
    parser.add_argument(
        "--baseline-down",
        default=DEFAULT_BASELINE_DOWN,
        help="Path to SHIFT_TABLES__DOWN__Throttle17__COMFORT.tsv",
    )
    args = parser.parse_args()

    logs = glob.glob(args.logs_glob)
    if not logs:
        raise RuntimeError(f"No logs matched glob: {args.logs_glob}")

    os.makedirs(args.out_dir, exist_ok=True)

    all_events = []

    for path in logs:
        print(f"[KICKDOWN_TAHOE] scanning {path}")
        df_raw = pd.read_csv(path)
        df = ensure_canon(df_raw, path)
        events = detect_kickdown_events(df, path)
        print(f"[KICKDOWN_TAHOE]   found {len(events)} candidate events")
        all_events.extend(events)

    events_cols = [
        "file",
        "time_s",
        "speed_mph",
        "tps",
        "tps_peak",
        "pedal",
        "gear_before",
        "gear_after",
        "row",
        "tps_bin",
        "dthr_dt_peak",
    ]
    events_df = pd.DataFrame(all_events, columns=events_cols)

    events_path = os.path.join(args.out_dir, "KICKDOWN__EVENTS_RAW_DEBUG__TAHOE.csv")
    events_df.to_csv(events_path, index=False)
    print(f"[KICKDOWN_TAHOE] events debug -> {events_path}")

    baseline_down = load_baseline_down(args.baseline_down)
    delta_df = build_delta_from_events(events_df, baseline_down)

    delta_path = os.path.join(args.out_dir, "KICKDOWN__SHIFT_DOWN__DELTA.tsv")
    delta_df.to_csv(delta_path, sep="\t", index=False)
    nonzero = (delta_df.drop(columns=["mph"]) != 0).sum().sum()
    total = delta_df.drop(columns=["mph"]).size
    print(f"[KICKDOWN_TAHOE] delta -> {delta_path} | nonzero {nonzero}/{total}")


if __name__ == "__main__":
    main()


