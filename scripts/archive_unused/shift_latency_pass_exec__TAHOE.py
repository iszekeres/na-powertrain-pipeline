#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Tahoe-specific constants
FINAL_DRIVE = 3.08
TIRE_DIAM_IN = 32.5
# mph -> output shaft rpm ≈ mph * 336 * FD / tire_diam
OUT_RPM_K = 336.0 * FINAL_DRIVE / TIRE_DIAM_IN

TPS_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]

CANON_FALLBACKS = {
    "time_s__canon": [
        "time_s",
        "Time",
        "Time (s)",
        "Elapsed Time",
    ],
    "speed_mph__canon": [
        "speed_mph",
        "Vehicle Speed",
        "Vehicle Speed (SAE)",
        "Vehicle Speed (mph)",
    ],
    "throttle_pct__canon": [
        "throttle_pct",
        "Throttle Position",
        "Throttle Position (%)",
        "TPS %",
        "Throttle (SAE)",
    ],
    "gear_actual__canon": [
        "gear_actual",
        "Trans Current Gear",
        "Trans Current Gear (SAE)",
        "Transmission Current Gear",
    ],
    "brake__canon": [
        "brake",
        "Brake",
        "Brake (on/off)",
        "Brake Applied",
    ],
    "turbine_rpm__canon": [
        "turbine_rpm",
        "Trans Input Shaft RPM",
        "Trans Input Shaft Speed",
        "ISS",
    ],
}

REQUIRED_CANON = [
    "time_s__canon",
    "speed_mph__canon",
    "throttle_pct__canon",
    "gear_actual__canon",
    "turbine_rpm__canon",
]


def ensure_canon(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    df = df.copy()
    missing_required = []

    for canon, alts in CANON_FALLBACKS.items():
        if canon in df.columns and df[canon].notna().any():
            continue
        found = None
        for alt in alts:
            if alt in df.columns and df[alt].notna().any():
                found = alt
                break
        if found is not None:
            df[canon] = df[found]
        elif canon in REQUIRED_CANON:
            missing_required.append(canon)

    if missing_required:
        raise RuntimeError(f"{path.name}: missing required canonical column(s): {missing_required}")

    # If brake isn't present after aliasing, assume "no brake"
    if "brake__canon" not in df.columns:
        df["brake__canon"] = 0.0

    # Force numeric types
    num_cols = REQUIRED_CANON + ["brake__canon"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaNs in key fields
    df = df.dropna(subset=REQUIRED_CANON).copy()
    return df


def tps_to_bin(tps: float) -> int:
    # Floor to the nearest TPS_AXIS bin
    for b in reversed(TPS_AXIS):
        if tps >= b:
            return b
    return 0


def detect_shift_events_exec(df: pd.DataFrame, path: Path):
    """
    Detect adjacent upshifts and compute execution time using
    turbine/output ratio plateaus.
    """
    # Sort by time just in case
    df = df.sort_values("time_s__canon").reset_index(drop=True)

    # Build output shaft rpm and ratio
    df["output_rpm"] = df["speed_mph__canon"].clip(lower=0.0) * OUT_RPM_K
    df["output_rpm"] = df["output_rpm"].replace(0, np.nan)
    df["ratio"] = df["turbine_rpm__canon"] / df["output_rpm"]
    df = df.dropna(subset=["ratio"]).copy()

    # Snap gear to Int64 for detection
    df["gear_int"] = df["gear_actual__canon"].round().astype("Int64")

    time = df["time_s__canon"].to_numpy()
    speed = df["speed_mph__canon"].to_numpy()
    tps = df["throttle_pct__canon"].to_numpy()
    brake = df["brake__canon"].to_numpy()
    gear = df["gear_int"].to_numpy()
    ratio = df["ratio"].to_numpy()

    # Tahoe gating
    MIN_MPH = 15.0
    MAX_MPH = 90.0
    MIN_TPS = 8.0
    MAX_TPS = 80.0
    BRAKE_MAX = 0.5  # brake flag is 0/1 in CLEAN_FULL

    # Windows for plateau measurement
    PRE_WIN = 0.25
    PRE_MARGIN = 0.05
    POST_MARGIN = 0.20
    POST_WIN = 0.60
    MIN_PRE_POINTS = 5
    MIN_POST_POINTS = 5
    MIN_RATIO_CHANGE_FRAC = 0.10  # require at least 10% ratio change
    EPS_OLD = 0.03
    EPS_NEW = 0.03
    MAX_DURATION = 1.0  # s

    events = []

    # Find indices where gear changes
    gear_prev = gear[:-1]
    gear_next = gear[1:]
    idx_changes = np.where(gear_next != gear_prev)[0] + 1

    for idx in idx_changes:
        g_before = gear[idx - 1]
        g_after = gear[idx]
        if pd.isna(g_before) or pd.isna(g_after):
            continue

        # Adjacent upshifts 1->2, 2->3, ..., 5->6
        if not (1 <= g_before <= 5 and g_after == g_before + 1):
            continue

        # Basic gating at event point
        t0 = float(time[idx])
        mph0 = float(speed[idx])
        tps0 = float(tps[idx])
        brake0 = float(brake[idx])

        if not (MIN_MPH <= mph0 <= MAX_MPH):
            continue
        if not (MIN_TPS <= tps0 <= MAX_TPS):
            continue
        if brake0 > BRAKE_MAX:
            continue

        row_label = f"{int(g_before)} -> {int(g_after)} Shift"
        tps_bin = tps_to_bin(tps0)

        # Pre and post windows for ratio plateaus
        pre_mask = (time >= t0 - PRE_WIN) & (time <= t0 - PRE_MARGIN)
        post_mask = (time >= t0 + POST_MARGIN) & (time <= t0 + POST_WIN)

        if pre_mask.sum() < MIN_PRE_POINTS or post_mask.sum() < MIN_POST_POINTS:
            continue

        r_old = np.median(ratio[pre_mask])
        r_new = np.median(ratio[post_mask])

        if not np.isfinite(r_old) or not np.isfinite(r_new):
            continue
        if r_old <= 0 or r_new <= 0:
            continue

        # Require a meaningful ratio change
        ratio_change = abs(r_new - r_old) / max(abs(r_old), 1e-6)
        if ratio_change < MIN_RATIO_CHANGE_FRAC:
            continue

        # "Old plateau" = last time we are close to old ratio before event
        rel_old = np.abs(ratio - r_old) / max(abs(r_old), 1e-6)
        is_old = (rel_old <= EPS_OLD) & (time <= t0)
        if not is_old.any():
            continue
        t_old_end = float(time[is_old].max())

        # "New plateau" = first time we are close to new ratio after event
        rel_new = np.abs(ratio - r_new) / max(abs(r_new), 1e-6)
        is_new = (rel_new <= EPS_NEW) & (time >= t0)
        if not is_new.any():
            continue
        t_new_start = float(time[is_new].min())

        duration = t_new_start - t_old_end
        if duration <= 0 or duration > MAX_DURATION:
            continue

        events.append(
            {
                "file": path.name,
                "row": row_label,
                "gear_before": int(g_before),
                "gear_after": int(g_after),
                "tps": tps0,
                "tps_bin": tps_bin,
                "speed_mph": mph0,
                "time_event": t0,
                "time_old_end": t_old_end,
                "time_new_start": t_new_start,
                "duration_s": duration,
                "ratio_old": r_old,
                "ratio_new": r_new,
                "ratio_change_frac": ratio_change,
            }
        )

    return events


def run(logs_glob: str, out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    events_all = []

    for path in sorted(Path().glob(logs_glob)):
        if not path.is_file():
            continue
        print(f"[LAT_EXEC_TAHOE] scanning {path}")
        df = pd.read_csv(path)
        df = ensure_canon(df, path)
        events = detect_shift_events_exec(df, path)
        print(f"[LAT_EXEC_TAHOE]   found {len(events)} events in {path.name}")
        events_all.extend(events)

    events_csv = out_path / "LAT_EXEC__EVENTS_DEBUG__TAHOE.csv"
    summary_csv = out_path / "LAT_EXEC__SUMMARY__TAHOE.csv"

    if not events_all:
        print("[LAT_EXEC_TAHOE] no qualifying events found; writing empty CSVs.")
        pd.DataFrame(
            columns=[
                "file",
                "row",
                "gear_before",
                "gear_after",
                "tps",
                "tps_bin",
                "speed_mph",
                "time_event",
                "time_old_end",
                "time_new_start",
                "duration_s",
                "ratio_old",
                "ratio_new",
                "ratio_change_frac",
            ]
        ).to_csv(events_csv, index=False)
        pd.DataFrame(
            columns=[
                "row",
                "tps_bin",
                "count",
                "median_latency_s",
                "std_latency_s",
                "median_mph",
            ]
        ).to_csv(summary_csv, index=False)
        return

    events_df = pd.DataFrame(events_all)
    events_df.to_csv(events_csv, index=False)

    summary = (
        events_df.groupby(["row", "tps_bin"], as_index=False)
        .agg(
            count=("duration_s", "size"),
            median_latency_s=("duration_s", "median"),
            std_latency_s=("duration_s", "std"),
            median_mph=("speed_mph", "median"),
        )
        .sort_values(["row", "tps_bin"])
    )
    summary.to_csv(summary_csv, index=False)

    total_events = len(events_df)
    nonzero_cells = (summary["count"] > 0).sum()
    print(
        f"[LAT_EXEC_TAHOE] total events: {total_events}, summary cells: {nonzero_cells}"
    )
    print(f"[LAT_EXEC_TAHOE] events debug -> {events_csv}")
    print(f"[LAT_EXEC_TAHOE] summary      -> {summary_csv}")


def main():
    ap = argparse.ArgumentParser(
        description="Tahoe shift execution latency pass (BESTINTERP, ratio-based)."
    )
    ap.add_argument(
        "--logs-glob",
        required=True,
        help="Glob for BESTINTERP CSVs, e.g. newlogs\\cleaned_bestinterp\\__trans_focus__clean_FULL__*__BESTINTERP.csv",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory under newlogs\\output\\02_passes",
    )
    args = ap.parse_args()
    run(args.logs_glob, args.out_dir)


if __name__ == "__main__":
    main()

