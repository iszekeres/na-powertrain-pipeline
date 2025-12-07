#!/usr/bin/env python3
"""
Build NOCRUISE intent episodes using the same logic/schema as the original
highway_trans_MAX_analysis detect_intent output (ALL__intent_episodes.csv).

Strict/no-fallback: required columns must exist in the prepped logs:
    time_s, pedal_pct, pedal_rate_pct_per_s (or will be derived),
    speed_mph, gear_actual__canon, tcc_state
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REQ_COLS = [
    "time_s",
    "pedal_pct",
    "speed_mph",
    "gear_actual__canon",
    "tcc_state",
]


def detect_intent(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """
    Port of highway_trans_MAX_analysis.detect_intent (same thresholds/schema).
    """
    for col in REQ_COLS:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in {file_name}")

    df = df.copy()
    time = pd.to_numeric(df["time_s"], errors="coerce").to_numpy()
    pedal = pd.to_numeric(df["pedal_pct"], errors="coerce").to_numpy()
    speed = pd.to_numeric(df["speed_mph"], errors="coerce").to_numpy()
    gear = pd.to_numeric(df["gear_actual__canon"], errors="coerce").astype(int).to_numpy()

    if "pedal_rate_pct_per_s" in df.columns:
        rate = pd.to_numeric(df["pedal_rate_pct_per_s"], errors="coerce").to_numpy()
    else:
        dt = np.diff(time, prepend=time[0])
        dt[dt <= 0] = np.nanmedian(dt[dt > 0]) if np.any(dt > 0) else 0.1
        rate = np.diff(pedal, prepend=pedal[0]) / dt

    n = len(df)
    rows: List[dict] = []
    for i in range(n):
        if np.isnan(rate[i]) or np.isnan(pedal[i]) or np.isnan(time[i]):
            continue
        if rate[i] >= 5 and rate[i] < 20 and pedal[i] >= 5:
            j = i
            while j < n and time[j] - time[i] <= 5.0:
                j += 1
            pedal_gain = pedal[j - 1] - pedal[i]
            if pedal_gain < 5 or pedal_gain > 20:
                continue
            gear_window = gear[i:j]
            gear_window = gear_window[~np.isnan(gear_window)]
            if gear_window.size == 0:
                continue
            gear_min = int(np.nanmin(gear_window))
            tcc_slice = df["tcc_state"].astype(str).iloc[i:j]
            unlock = (tcc_slice != "LOCKED").any()
            speed_gain = speed[j - 1] - speed[i]
            label = "immediate" if speed_gain > 3 else "lazy"
            rows.append(
                {
                    "file_name": file_name,
                    "log_id": Path(file_name).stem,
                    "event_time_s": float(time[i]),
                    "t_start_s": float(time[i]),
                    "t_end_s": float(time[i] + 5.0),
                    "gear_start": int(gear[i]) if not np.isnan(gear[i]) else gear_min,
                    "gear_min": gear_min,
                    "speed_start_mph": float(speed[i]),
                    "speed_gain_5s": float(speed_gain),
                    "pedal_start_pct": float(pedal[i]),
                    "pedal_end_pct": float(pedal[j - 1]),
                    "tcc_unlock": bool(unlock),
                    "response_label": label,
                }
            )
    return pd.DataFrame(rows)


def build_intent_episodes(prepped_dir: Path, out_path: Path) -> pd.DataFrame:
    csvs = sorted([p for p in prepped_dir.glob("*.csv") if p.is_file() and p.stat().st_size > 1024])
    if not csvs:
        raise SystemExit(f"[ERROR] No prepped CSVs found in {prepped_dir}")

    all_eps = []
    for csv in csvs:
        df = pd.read_csv(csv, low_memory=False)
        missing = [c for c in REQ_COLS if c not in df.columns]
        if missing:
            raise SystemExit(f"[ERROR] {csv.name} missing columns: {missing}")
        eps = detect_intent(df, csv.name)
        all_eps.append(eps)

    df_all = pd.concat(all_eps, ignore_index=True) if all_eps else pd.DataFrame(
        columns=[
            "file_name",
            "log_id",
            "event_time_s",
            "t_start_s",
            "t_end_s",
            "gear_start",
            "gear_min",
            "speed_start_mph",
            "speed_gain_5s",
            "pedal_start_pct",
            "pedal_end_pct",
            "tcc_unlock",
            "response_label",
        ]
    )
    df_all.to_csv(out_path, index=False)

    total = len(df_all)
    avg_dur = 5.0  # fixed 5s window in logic
    by_gear = df_all["gear_start"].value_counts().sort_index()
    print(f"[INFO] Wrote {total} intent episodes to {out_path}")
    print(f"[INFO] Avg duration (fixed window): {avg_dur:.1f} s")
    for g, c in by_gear.items():
        print(f"  gear {g}: {c} episodes")
    return df_all


def parse_args():
    ap = argparse.ArgumentParser(description="Build NOCRUISE intent episodes (schema-compatible with ALL__intent_episodes.csv)")
    ap.add_argument("--prepped-dir", required=True, help="Directory containing *__prepped.csv logs")
    ap.add_argument("--out", required=True, help="Output CSV path for intent episodes")
    return ap.parse_args()


def main():
    args = parse_args()
    prepped_dir = Path(args.prepped_dir)
    out_path = Path(args.out)
    build_intent_episodes(prepped_dir, out_path)


if __name__ == "__main__":
    main()
