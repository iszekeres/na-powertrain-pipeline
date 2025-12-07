#!/usr/bin/env python3
"""
consist_pass_weighted__TAHOE.py
BESTINTERP-aware CONSIST pass with Tahoe aggregation gates.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

TPS_AXIS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
ROW_UP = ["1 -> 2 Shift","2 -> 3 Shift","3 -> 4 Shift","4 -> 5 Shift","5 -> 6 Shift"]
ROW_DN = ["2 -> 1 Shift","3 -> 2 Shift","4 -> 3 Shift","5 -> 4 Shift","6 -> 5 Shift"]
TPS_COLS = [str(int(x)) for x in TPS_AXIS]

MIN_COUNT_BY_ROW = {
    "1 -> 2 Shift": 6,
    "2 -> 3 Shift": 6,
    "3 -> 4 Shift": 6,
    "4 -> 5 Shift": 4,
    "5 -> 6 Shift": 4,
}
STD_MAX = 5.0
BASE_GAP_MIN = 0.3
DELTA_CAP = 0.3

DEFAULT_BASELINE_UP = os.path.join(
    "newlogs","output","01_tables","shift","SHIFT_TABLES__UP__Throttle17__COMFORT.tsv"
)
DEFAULT_BASELINE_DOWN = os.path.join(
    "newlogs","output","01_tables","shift","SHIFT_TABLES__DOWN__Throttle17__COMFORT.tsv"
)
DEFAULT_OUT_DIR = os.path.join(
    "newlogs","output","02_passes","CONSIST_FROM_BEST_TAHOE"
)

ALIAS = {
    "time_s__canon": ["time_s__canon","time_s","Time","Time (s)","Time_s","offset","Offset"],
    "speed_mph__canon": ["speed_mph__canon","speed_mph","Vehicle Speed (SAE)","Vehicle Speed"],
    "throttle_pct__canon": ["throttle_pct__canon","throttle_pct","Throttle Position","Throttle Position (%)"],
    "gear_actual__canon": ["gear_actual__canon","gear_actual","Trans Current Gear","Gear Actual"],
    "gear_cmd__canon": ["gear_cmd__canon","gear_cmd","Trans Commanded Gear","Gear Commanded"],
    "oncoming_clutch__canon": ["oncoming_clutch__canon","oncoming_clutch"],
}
CORE_COLS = ["time_s__canon","speed_mph__canon","throttle_pct__canon","gear_actual__canon"]


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
    df = df.assign(gear_int=df["gear_actual__canon"].astype(float).round().astype("Int64"))
    return df


def detect_shift_events(df, path):
    events = []
    ga = df["gear_int"].astype(float).to_numpy()
    actual = df["gear_actual__canon"].astype(float).to_numpy()
    t = df["time_s__canon"].to_numpy()
    v = df["speed_mph__canon"].to_numpy()
    tps = df["throttle_pct__canon"].to_numpy()
    cmd = df["gear_cmd__canon"].astype(float).to_numpy() if "gear_cmd__canon" in df.columns else None
    clutch = df["oncoming_clutch__canon"].astype(float).to_numpy() if "oncoming_clutch__canon" in df.columns else None

    dgear = np.diff(ga)
    idx = np.where(dgear != 0)[0]
    for i in idx:
        gb = ga[i]
        ga_ = ga[i+1]
        if np.isnan(gb) or np.isnan(ga_):
            continue
        if not (1 <= gb <= 6 and 1 <= ga_ <= 6):
            continue
        step = ga_ - gb
        if abs(step) != 1:
            continue
        table = "UP" if step > 0 else "DOWN"
        row = f"{int(gb)} -> {int(ga_)} Shift"
        j = i + 1
        event = {
            "file": os.path.basename(path),
            "table": table,
            "row": row,
            "time_s": float(t[j]),
            "speed_mph": float(v[j]),
            "tps": float(tps[j]),
            "gear_int_before": float(gb),
            "gear_int_after": float(ga_),
            "gear_actual_before": float(actual[i]),
            "gear_actual_after": float(actual[j]),
            "gear_cmd": float(cmd[j]) if cmd is not None else np.nan,
            "oncoming_clutch": float(clutch[j]) if clutch is not None else np.nan,
        }
        events.append(event)
    return events


def nearest_tps_bin(val):
    if np.isnan(val):
        return int(TPS_AXIS[0])
    v = min(100.0, max(0.0, float(val)))
    idx = int(np.argmin(np.abs(np.array(TPS_AXIS) - v)))
    return int(TPS_AXIS[idx])


def scatter_events_df(events_df):
    if events_df.empty:
        return events_df.assign(tps_bin=pd.Series(dtype=int))
    df = events_df.copy()
    df["tps_bin"] = df["tps"].apply(nearest_tps_bin)
    return df


def build_summary(df_events):
    if df_events.empty:
        return pd.DataFrame(columns=["table","row","tps_bin","count","median_mph","std_mph"])
    grouped = df_events.groupby(["table","row","tps_bin"]) ["speed_mph"].agg(
        count="count", median_mph="median", std_mph="std"
    ).reset_index()
    return grouped


def blank_numeric(rows, fill=np.nan):
    df = pd.DataFrame(fill, index=rows, columns=TPS_COLS, dtype=float)
    return df


def write_table(table, rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["mph"] + TPS_COLS + ["%"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            vals = []
            for col in TPS_COLS:
                val = table.loc[row, col]
                if np.isnan(val):
                    vals.append("")
                else:
                    vals.append(f"{round(val,1):.1f}")
            f.write("\t".join([row] + vals + [""]) + "\n")


def write_delta(table, rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["mph"] + TPS_COLS + ["%"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            vals = [f"{round(table.loc[row, col],1):.1f}" for col in TPS_COLS]
            f.write("\t".join([row] + vals + [""]) + "\n")


def load_baseline(path):
    df = pd.read_csv(path, sep="\t")
    rename = {}
    for col in df.columns:
        try:
            rename[col] = str(int(float(col)))
        except ValueError:
            rename[col] = col
    df = df.rename(columns=rename)
    return df


def build_suggested(summary, table_name, rows):
    table = blank_numeric(rows)
    subset = summary[summary["table"] == table_name]
    for _, r in subset.iterrows():
        row_label = r["row"]
        tps_col = str(int(r["tps_bin"]))
        if row_label in table.index and tps_col in table.columns:
            table.loc[row_label, tps_col] = r["median_mph"]
    return table


def build_delta(summary, baseline, table_name, rows):
    table = blank_numeric(rows, fill=0.0)
    subset = summary[summary["table"] == table_name]
    for _, r in subset.iterrows():
        row_label = r["row"]
        tps_col = str(int(r["tps_bin"]))
        if row_label not in table.index or tps_col not in table.columns:
            continue
        base_row = baseline[baseline["mph"] == row_label]
        if base_row.empty or tps_col not in base_row.columns:
            continue
        count = int(r["count"])
        std = r["std_mph"]
        std = 0.0 if np.isnan(std) else float(std)
        median = float(r["median_mph"])
        min_count = MIN_COUNT_BY_ROW.get(row_label, 4)
        if count < min_count or std > STD_MAX:
            continue
        base_mph = float(base_row.iloc[0][tps_col])
        gap = median - base_mph
        if abs(gap) < BASE_GAP_MIN:
            continue
        delta = max(-DELTA_CAP, min(DELTA_CAP, gap))
        table.loc[row_label, tps_col] = delta
    return table


def main():
    parser = argparse.ArgumentParser(description="Tahoe BESTINTERP CONSIST")
    parser.add_argument("--logs-glob", required=True)
    parser.add_argument("--baseline-up", default=DEFAULT_BASELINE_UP)
    parser.add_argument("--baseline-down", default=DEFAULT_BASELINE_DOWN)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    logs = sorted(glob.glob(args.logs_glob))
    if not logs:
        raise RuntimeError(f"No logs matched {args.logs_glob}")

    os.makedirs(args.out_dir, exist_ok=True)

    all_events = []
    for path in logs:
        print(f"[CONSIST_TAHOE] scanning {path}")
        df_raw = pd.read_csv(path, low_memory=False)
        try:
            df = ensure_canon(df_raw, path)
        except RuntimeError as exc:
            print(f"[WARN] {exc}; skipping")
            continue
        events = detect_shift_events(df, path)
        print(f"[CONSIST_TAHOE]   found {len(events)} shift events")
        all_events.extend(events)

    events_cols = [
        "file","table","row","time_s","speed_mph","tps",
        "gear_int_before","gear_int_after","gear_actual_before",
        "gear_actual_after","gear_cmd","oncoming_clutch"
    ]
    events_df = pd.DataFrame(all_events, columns=events_cols)
    events_path = os.path.join(args.out_dir, "CONSIST__SHIFT_EVENTS_DEBUG__TAHOE.csv")
    events_df.to_csv(events_path, index=False)
    print(f"[CONSIST_TAHOE] events -> {events_path}")

    bins_df = scatter_events_df(events_df)
    summary_df = build_summary(bins_df)
    summary_path = os.path.join(args.out_dir, "CONSIST__DEBUG_SUMMARY__TAHOE.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[CONSIST_TAHOE] summary -> {summary_path}")

    baseline_up = load_baseline(args.baseline_up)
    baseline_down = load_baseline(args.baseline_down)

    up_sugg = build_suggested(summary_df, "UP", ROW_UP)
    down_sugg = build_suggested(summary_df, "DOWN", ROW_DN)
    write_table(up_sugg, ROW_UP, os.path.join(args.out_dir, "CONSIST__SHIFT_UP__SUGGESTED__TAHOE.tsv"))
    write_table(down_sugg, ROW_DN, os.path.join(args.out_dir, "CONSIST__SHIFT_DOWN__SUGGESTED__TAHOE.tsv"))

    up_delta = build_delta(summary_df, baseline_up, "UP", ROW_UP)
    down_delta = build_delta(summary_df, baseline_down, "DOWN", ROW_DN)
    write_delta(up_delta, ROW_UP, os.path.join(args.out_dir, "CONSIST__SHIFT_UP__DELTA__TAHOE.tsv"))
    write_delta(down_delta, ROW_DN, os.path.join(args.out_dir, "CONSIST__SHIFT_DOWN__DELTA__TAHOE.tsv"))

    nz_up = int((up_delta.to_numpy() != 0).sum())
    nz_dn = int((down_delta.to_numpy() != 0).sum())
    total_up = up_delta.size
    total_dn = down_delta.size
    print(f"[CONSIST_TAHOE] UP delta nonzero {nz_up}/{total_up}")
    print(f"[CONSIST_TAHOE] DOWN delta nonzero {nz_dn}/{total_dn}")

if __name__ == "__main__":
    main()
