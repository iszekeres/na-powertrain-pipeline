#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

TPS_BINS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]

def pick_col(df, candidates, label, required=True):
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise KeyError(f"Missing required column for {label}; tried {candidates}")
    return None

def detect_edges_in_file(path: Path, args):
    df = pd.read_csv(path)

    # Canonical / alias mapping
    time_col = pick_col(
        df,
        ["time_s__canon", "time_s", "Time_s", "Time", "TIME", "offset"],
        "time_s",
    )
    speed_col = pick_col(
        df,
        ["speed_mph__canon", "speed_mph", "Vehicle Speed (SAE)", "Vehicle Speed", "VSS"],
        "speed_mph",
    )
    gear_col = pick_col(
        df,
        ["gear_actual__canon", "gear_int", "Trans Current Gear", "Trans Current Gear (SAE)", "gear_actual"],
        "gear_actual",
    )
    tps_col = pick_col(
        df,
        ["throttle_pct__canon", "Throttle Position", "Throttle Position (SAE)", "Throttle (%)", "Throttle"],
        "throttle_pct",
    )

    # Slip: prefer fused, else build from engine - turbine
    slip_candidates = [
        "tcc_slip_fused__canon",
        "tcc_slip_fused",
        "TCC Slip",
        "TCC Slip (RPM)",
        "TCC Slip RPM",
    ]
    slip_col = None
    for name in slip_candidates:
        if name in df.columns:
            slip_col = name
            break

    if slip_col is None:
        eng_col = pick_col(
            df,
            ["engine_rpm__canon", "Engine Speed", "Engine RPM", "Engine RPM (SAE)"],
            "engine_rpm",
            required=False,
        )
        tur_col = pick_col(
            df,
            [
                "turbine_rpm__canon",
                "Trans Input Shaft RPM",
                "Trans Input Shaft Speed",
                "Turbine Speed",
                "Transmission ISS",
                "Trans ISS",
            ],
            "turbine_rpm",
            required=False,
        )
        if eng_col and tur_col:
            df["__tcc_slip_built"] = pd.to_numeric(df[eng_col], errors="coerce") - pd.to_numeric(
                df[tur_col], errors="coerce"
            )
            slip_col = "__tcc_slip_built"

    if slip_col is None:
        print(f"[TCC_EDGE_TAHOE] {path.name}: no slip column; skipping", file=sys.stderr)
        return []

    # Coerce numerics and drop NaNs
    for col in [time_col, speed_col, gear_col, tps_col, slip_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[[time_col, speed_col, gear_col, tps_col, slip_col]].dropna()
    if df.empty:
        print(f"[TCC_EDGE_TAHOE] {path.name}: empty after dropna; skipping", file=sys.stderr)
        return []

    df = df.sort_values(time_col)

    t = df[time_col].to_numpy()
    v = df[speed_col].to_numpy()
    g = df[gear_col].to_numpy()
    tps = df[tps_col].to_numpy()
    slip = df[slip_col].to_numpy()

    # Snap gear to int
    g_int = np.round(g).astype("int64")

    on_rpm = float(args.on_rpm)
    off_rpm = float(args.off_rpm)
    min_speed = float(args.min_speed)
    min_gear = int(args.min_gear)
    dead_s = float(args.dead_s)

    events = []
    state = "unknown"
    n = len(df)
    last_apply_time = -1e9
    last_release_time = -1e9

    # Initial state guess
    if abs(slip[0]) <= on_rpm:
        state = "locked"
    elif abs(slip[0]) >= off_rpm:
        state = "unlocked"
    else:
        state = "unknown"

    for i in range(1, n):
        now = float(t[i])
        s_abs = float(abs(slip[i]))
        gear_now = int(g_int[i])
        speed_now = float(v[i])
        tps_now = float(tps[i])

        # Only care about cruising gears and speeds
        if gear_now < min_gear or speed_now < min_speed:
            # Still update state but don't log events
            if s_abs <= on_rpm:
                new_state = "locked"
            elif s_abs >= off_rpm:
                new_state = "unlocked"
            else:
                new_state = state
            state = new_state
            continue

        # Determine current slip state relative to thresholds
        if s_abs <= on_rpm:
            new_state = "locked"
        elif s_abs >= off_rpm:
            new_state = "unlocked"
        else:
            new_state = state

        # Detect edges with a dead-time debounce
        if state != "locked" and new_state == "locked" and (now - last_apply_time) >= dead_s:
            events.append(
                {
                    "file": path.name,
                    "edge": "APPLY",
                    "time_s": now,
                    "speed_mph": speed_now,
                    "tps": tps_now,
                    "gear": gear_now,
                    "slip_rpm": float(slip[i]),
                }
            )
            last_apply_time = now

        if state != "unlocked" and new_state == "unlocked" and (now - last_release_time) >= dead_s:
            events.append(
                {
                    "file": path.name,
                    "edge": "RELEASE",
                    "time_s": now,
                    "speed_mph": speed_now,
                    "tps": tps_now,
                    "gear": gear_now,
                    "slip_rpm": float(slip[i]),
                }
            )
            last_release_time = now

        state = new_state

    return events

def write_events_and_summary(events, out_dir: Path):
    events_path = out_dir / "TCC_EDGE__EVENTS__TAHOE.csv"
    summary_path = out_dir / "TCC_EDGE__SUMMARY__TAHOE.csv"

    # Events CSV (always written)
    if events:
        df_events = pd.DataFrame(events)
    else:
        df_events = pd.DataFrame(
            columns=["file", "edge", "time_s", "speed_mph", "tps", "gear", "slip_rpm"]
        )
    df_events.to_csv(events_path, index=False)

    # Summary by edge + gear (always written)
    summary_rows = []
    if not df_events.empty:
        grouped = df_events.groupby(["edge", "gear"])
        for (edge, gear), grp in grouped:
            summary_rows.append(
                {
                    "edge": edge,
                    "gear": int(gear),
                    "count": int(len(grp)),
                    "mean_speed_mph": float(grp["speed_mph"].mean()),
                    "median_speed_mph": float(grp["speed_mph"].median()),
                    "mean_tps": float(grp["tps"].mean()),
                    "median_tps": float(grp["tps"].median()),
                    "mean_slip_rpm": float(grp["slip_rpm"].mean()),
                    "median_slip_rpm": float(grp["slip_rpm"].median()),
                }
            )

    df_summary = pd.DataFrame(
        summary_rows,
        columns=[
            "edge",
            "gear",
            "count",
            "mean_speed_mph",
            "median_speed_mph",
            "mean_tps",
            "median_tps",
            "mean_slip_rpm",
            "median_slip_rpm",
        ],
    )
    df_summary.to_csv(summary_path, index=False)

    return events_path, summary_path

def write_zero_tcc_delta(out_path: Path, label_suffix: str):
    cols = ["mph"] + [str(b) for b in TPS_BINS] + ["%"]
    rows = []
    for gear_label in ["3rd", "4th", "5th", "6th"]:
        row_name = f"{gear_label} {label_suffix}"
        rows.append([row_name] + [0.0] * len(TPS_BINS) + [""])
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_path, sep="\t", index=False, float_format="%.1f")

def main():
    parser = argparse.ArgumentParser(
        description="Tahoe EC3 TCC edge detector (diagnostic, neutral deltas)."
    )
    parser.add_argument("--clean-list", required=True, help="Text file of BESTINTERP CSV paths")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--on-rpm", type=float, default=60.0, help="Lock threshold |slip| <= on_rpm (rpm)")
    parser.add_argument("--off-rpm", type=float, default=120.0, help="Unlock threshold |slip| >= off_rpm (rpm)")
    parser.add_argument("--min-speed", type=float, default=25.0, help="Min speed for edges (mph)")
    parser.add_argument("--min-gear", type=int, default=3, help="Min gear for edges")
    parser.add_argument("--dead-s", type=float, default=0.4, help="Min time between edges of same type (s)")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_list_path = Path(args.clean_list)
    if not clean_list_path.exists():
        print(f"[TCC_EDGE_TAHOE] clean-list not found: {clean_list_path}", file=sys.stderr)
        sys.exit(1)

    with open(clean_list_path, "r", encoding="utf-8") as f:
        files = [line.strip() for line in f if line.strip()]

    if not files:
        print("[TCC_EDGE_TAHOE] clean-list is empty", file=sys.stderr)

    all_events = []
    for line in files:
        p = Path(line)
        if not p.exists():
            print(f"[TCC_EDGE_TAHOE] missing file in clean-list: {p}", file=sys.stderr)
            continue
        try:
            ev = detect_edges_in_file(p, args)
            all_events.extend(ev)
        except Exception as e:
            print(f"[TCC_EDGE_TAHOE] error on {p.name}: {e}", file=sys.stderr)

    n_apply = sum(1 for e in all_events if e.get("edge") == "APPLY")
    n_release = sum(1 for e in all_events if e.get("edge") == "RELEASE")
    print(f"[TCC_EDGE_TAHOE] total events: {len(all_events)} (APPLY={n_apply}, RELEASE={n_release})")

    events_path, summary_path = write_events_and_summary(all_events, out_dir)

    # Neutral deltas (diagnostic only)
    apply_delta_path = out_dir / "TCC_EDGE__APPLY__DELTA__TAHOE.tsv"
    release_delta_path = out_dir / "TCC_EDGE__RELEASE__DELTA__TAHOE.tsv"
    write_zero_tcc_delta(apply_delta_path, "Apply")
    write_zero_tcc_delta(release_delta_path, "Release")

    print(f"[TCC_EDGE_TAHOE] events  -> {events_path}")
    print(f"[TCC_EDGE_TAHOE] summary -> {summary_path}")
    print(f"[TCC_EDGE_TAHOE] deltas  -> {apply_delta_path} / {release_delta_path} (all zeros)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
