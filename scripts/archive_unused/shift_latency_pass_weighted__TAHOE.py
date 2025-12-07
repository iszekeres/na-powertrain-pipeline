#!/usr/bin/env python3
"""
shift_latency_pass_weighted__TAHOE.py
BESTINTERP-aware latency pass with Tahoe-specific gating/aggregation.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

TPS_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]
ROW_UP = ["1 -> 2 Shift", "2 -> 3 Shift", "3 -> 4 Shift", "4 -> 5 Shift", "5 -> 6 Shift"]
TPS_COLS = [str(int(x)) for x in TPS_AXIS]

MIN_SPEED = 15.0
MAX_SPEED = 80.0
TPS_MIN = 8.0
TPS_MAX = 80.0
BRAKE_MAX = 0.1
DTHR_MAX = 40.0

MIN_COUNT_BY_ROW = {
    "1 -> 2 Shift": 4,
    "2 -> 3 Shift": 4,
    "3 -> 4 Shift": 4,
    "4 -> 5 Shift": 4,
    "5 -> 6 Shift": 3,
}
STD_MAX = 0.25
LAT_TARGET = 0.35
LAT_SLOW = 0.45
DELTA_CAP = 0.3

DEFAULT_OUT_PREFIX = os.path.join("newlogs", "output", "02_passes", "LAT_FROM_BEST_TAHOE", "LAT__")

ALIAS = {
    "time_s__canon": ["time_s__canon", "time_s", "Time", "Time (s)", "offset", "Offset"],
    "speed_mph__canon": ["speed_mph__canon", "speed_mph", "Vehicle Speed (SAE)", "Vehicle Speed"],
    "throttle_pct__canon": ["throttle_pct__canon", "throttle_pct", "Throttle Position", "Throttle Position (%)"],
    "gear_actual__canon": ["gear_actual__canon", "gear_actual", "Trans Current Gear", "Gear Actual"],
    "gear_cmd__canon": ["gear_cmd__canon", "gear_cmd", "Trans Commanded Gear", "Gear Commanded"],
    "brake__canon": ["brake__canon", "brake", "Brake", "Brake (on/off)"],
    "pedal_pct__canon": ["pedal_pct__canon", "pedal_pct", "Accelerator Pedal Position", "Accelerator Pedal Position (%)"],
}
CORE_COLS = ["time_s__canon", "speed_mph__canon", "throttle_pct__canon", "gear_actual__canon"]


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_canon(df, path):
    df = df.copy()
    for canon, candidates in ALIAS.items():
        need = canon not in df.columns or df[canon].dropna().empty
        if not need:
            continue
        for cand in candidates:
            if cand in df.columns and not df[cand].dropna().empty:
                df[canon] = df[cand]
                need = False
                break
        if need and canon in CORE_COLS:
            raise RuntimeError(f"{path}: missing required column {canon}")
    missing = [c for c in CORE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path}: missing required columns {missing}")
    df = df.dropna(subset=CORE_COLS, how="any").copy()
    df["gear_int"] = df["gear_actual__canon"].astype(float).round().astype("Int64")
    df = df.dropna(subset=["gear_int"]).copy()
    df["dt"] = df["time_s__canon"].diff()
    df["dthr_dt"] = 0.0
    valid = df["dt"] > 0
    df.loc[valid, "dthr_dt"] = df.loc[valid, "throttle_pct__canon"].diff()[valid] / df.loc[valid, "dt"]
    return df


def nearest_tps_bin(val):
    if np.isnan(val):
        return int(TPS_AXIS[0])
    v = min(100.0, max(0.0, float(val)))
    idx = int(np.argmin(np.abs(np.array(TPS_AXIS) - v)))
    return int(TPS_AXIS[idx])


def detect_latency_events(df, path):
    events = []
    gear = df["gear_int"].to_numpy()
    gear_actual = df["gear_actual__canon"].to_numpy()
    t = df["time_s__canon"].to_numpy()
    v = df["speed_mph__canon"].to_numpy()
    tps = df["throttle_pct__canon"].to_numpy()
    dthr = df["dthr_dt"].to_numpy()
    brake = df["brake__canon"].to_numpy() if "brake__canon" in df.columns else np.zeros(len(df))
    gear_cmd = df["gear_cmd__canon"].astype(float).round().to_numpy() if "gear_cmd__canon" in df.columns else None

    dgear = np.diff(gear)
    idxs = np.where(dgear == 1)[0]
    for i in idxs:
        j = i + 1
        before = gear[i]
        after = gear[j]
        row = f"{int(before)} -> {int(after)} Shift"
        start_idx = max(j - 1, 0)
        if gear_cmd is not None:
            search_start = max(1, j - 200)
            for k in range(search_start, j + 1):
                if gear_cmd[k - 1] == before and gear_cmd[k] == after:
                    start_idx = k
                    break
        lat = t[j] - t[start_idx]
        if not np.isfinite(lat) or lat <= 0:
            lat = abs(df["dt"].iloc[j]) if j < len(df) else 0.02
        event = {
            "file": os.path.basename(path),
            "time_s": float(t[j]),
            "speed_mph": float(v[j]),
            "tps": float(tps[j]),
            "gear_before": float(before),
            "gear_after": float(after),
            "lat_s": float(lat),
            "row": row,
            "tps_bin": nearest_tps_bin(tps[j]),
            "dthr_dt": float(dthr[j]) if np.isfinite(dthr[j]) else 0.0,
            "brake": float(brake[j]) if j < len(brake) else 0.0,
        }
        events.append(event)
    return events


def apply_gating(events):
    gated = []
    for ev in events:
        speed = ev["speed_mph"]
        tps = ev["tps"]
        if not (MIN_SPEED <= speed <= MAX_SPEED):
            continue
        if not (TPS_MIN <= tps <= TPS_MAX):
            continue
        if abs(ev["dthr_dt"]) > DTHR_MAX:
            continue
        if ev.get("brake", 0.0) > BRAKE_MAX:
            continue
        gated.append(ev)
    return gated


def build_summary(df_events):
    if df_events.empty:
        return pd.DataFrame(columns=["row", "tps_bin", "count", "median_latency", "std_latency", "median_mph"])
    grouped = df_events.groupby(["row", "tps_bin"]).agg(
        count=("lat_s", "count"),
        median_latency=("lat_s", "median"),
        std_latency=("lat_s", "std"),
        median_mph=("speed_mph", "median"),
    ).reset_index()
    return grouped


def blank_delta_table():
    data = pd.DataFrame(0.0, index=ROW_UP, columns=TPS_COLS)
    return data


def write_delta_table(table, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["mph"] + TPS_COLS + ["%"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in ROW_UP:
            vals = [f"{round(table.loc[row, col],1):.1f}" for col in TPS_COLS]
            fh.write("\t".join([row] + vals + [""]) + "\n")


def write_events(events_df, path):
    events_df.to_csv(path, index=False)


def write_summary(summary_df, path):
    summary_df.to_csv(path, index=False)


def build_delta(summary_df):
    delta = blank_delta_table()
    for _, row in summary_df.iterrows():
        row_label = row["row"]
        if row_label not in delta.index:
            continue
        tps_col = str(int(row["tps_bin"]))
        if tps_col not in delta.columns:
            continue
        count = int(row["count"])
        std = row["std_latency"]
        std = 0.0 if np.isnan(std) else float(std)
        median_lat = float(row["median_latency"])
        min_count = MIN_COUNT_BY_ROW.get(row_label, 4)
        if count < min_count or std > STD_MAX or median_lat <= LAT_SLOW:
            continue
        gap = median_lat - LAT_TARGET
        if gap <= 0:
            continue
        if gap > 0.20:
            delta_val = -0.3
        elif gap > 0.10:
            delta_val = -0.2
        else:
            delta_val = -0.1
        delta_val = max(-DELTA_CAP, min(0.0, delta_val))
        delta.loc[row_label, tps_col] = delta_val
    return delta


def main():
    parser = argparse.ArgumentParser(description="Tahoe BESTINTERP latency pass")
    parser.add_argument("--logs-glob", required=True)
    parser.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    args = parser.parse_args()

    logs = sorted(glob.glob(args.logs_glob))
    if not logs:
        raise RuntimeError(f"No logs matched {args.logs_glob}")

    all_events = []
    for path in logs:
        print(f"[LAT_TAHOE] scanning {path}")
        df_raw = pd.read_csv(path, low_memory=False)
        try:
            df = ensure_canon(df_raw, path)
        except RuntimeError as exc:
            print(f"[WARN] {exc}; skipping")
            continue
        events = detect_latency_events(df, path)
        events = apply_gating(events)
        print(f"[LAT_TAHOE]   kept {len(events)} events")
        all_events.extend(events)

    events_df = pd.DataFrame(all_events, columns=[
        "file","time_s","speed_mph","tps","gear_before","gear_after","lat_s","row","tps_bin"
    ])
    prefix = args.out_prefix
    base_dir = os.path.dirname(prefix) or "."
    os.makedirs(base_dir, exist_ok=True)

    events_path = f"{prefix}SHIFT_EVENTS_DEBUG__TAHOE.csv"
    write_events(events_df, events_path)
    print(f"[LAT_TAHOE] events -> {events_path}")

    summary_df = build_summary(events_df)
    summary_path = f"{prefix}DEBUG_SUMMARY__TAHOE.csv"
    write_summary(summary_df, summary_path)
    print(f"[LAT_TAHOE] summary -> {summary_path}")

    delta_df = build_delta(summary_df)
    delta_path = f"{prefix}SHIFT_UP__DELTA__TAHOE.tsv"
    write_delta_table(delta_df, delta_path)
    nz = int((delta_df.values != 0).sum())
    total = delta_df.size
    print(f"[LAT_TAHOE] delta -> {delta_path} | nonzero {nz}/{total}")


if __name__ == "__main__":
    main()

