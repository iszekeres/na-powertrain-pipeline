import argparse
import glob
import os

import numpy as np
import pandas as pd

TPS_AXIS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
DOWN_ROWS = ["2 -> 1 Shift","3 -> 2 Shift"]
DEFAULT_BASELINE = os.path.join(
    "newlogs","output","01_tables","shift","SHIFT_TABLES__DOWN__Throttle17__COMFORT.tsv"
)

MIN_SPEED = 1.0
MAX_SPEED = 22.0
TPS_PEAK_MAX = 35.0
PEDAL_MAX = 40.0
DTHR_MIN = 3.0
WINDOW_BACK = 0.2
WINDOW_FWD = 0.8
BRAKE_MAX = 0.0

MIN_COUNT = 4
STD_MAX = 4.0
BASE_GAP_MIN = 0.3
DELTA_MPH = 0.3

CANON_MAP = {
    "time": ["time_s__canon","time_s","Time","Time (s)","offset","Offset"],
    "speed": ["speed_mph__canon","Vehicle Speed (SAE)","Vehicle Speed","Speed (mph)"],
    "tps": ["throttle_pct__canon","Throttle Position","Throttle Position (SAE)","Throttle Position (%)"],
    "pedal": ["pedal_pct__canon","Accelerator Pedal Position","Accelerator Pedal Position (SAE)"],
    "gear": ["gear_actual__canon","Trans Current Gear","Trans Current Gear (SAE)"],
    "brake": ["brake__canon","brake","Brake","Brake (on/off)"]
}


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_canon(df, path):
    cols = {}
    for key, candidates in CANON_MAP.items():
        col = pick_col(df, candidates)
        cols[key] = col
    missing = [key for key, col in cols.items() if col is None and key in ("time","speed","tps","gear")]
    if missing:
        raise RuntimeError(f"{os.path.basename(path)}: missing required column(s): {missing}")

    out = df.copy()
    out["time"] = pd.to_numeric(out[cols["time"]], errors="coerce") if cols["time"] else np.nan
    out["speed_mph"] = pd.to_numeric(out[cols["speed"]], errors="coerce") if cols["speed"] else np.nan
    out["tps"] = pd.to_numeric(out[cols["tps"]], errors="coerce") if cols["tps"] else np.nan
    out["gear"] = pd.to_numeric(out[cols["gear"]], errors="coerce") if cols["gear"] else np.nan

    if cols["pedal"]:
        out["pedal"] = pd.to_numeric(out[cols["pedal"]], errors="coerce")
    else:
        out["pedal"] = np.nan

    if cols["brake"]:
        out["brake"] = pd.to_numeric(out[cols["brake"]], errors="coerce").fillna(0.0)
    else:
        out["brake"] = 0.0

    out = out.dropna(subset=["time","speed_mph","tps","gear"]).copy()
    gear_int = out["gear"].round().astype("Int64")
    out = out.assign(gear_int=gear_int)
    out = out.dropna(subset=["gear_int"]).copy()

    out["dt"] = out["time"].diff()
    out["dthr"] = out["tps"].diff()
    out["dthr_dt"] = 0.0
    mask_dt = out["dt"] > 0
    out.loc[mask_dt, "dthr_dt"] = out.loc[mask_dt, "dthr"] / out.loc[mask_dt, "dt"]

    return out


def snap_tps_bin(val):
    if not np.isfinite(val):
        return TPS_AXIS[0]
    for b in reversed(TPS_AXIS):
        if val >= b:
            return b
    return TPS_AXIS[0]


def detect_stopgo_events(df, filename):
    events = []
    df = df.copy()
    df["gear_prev"] = df["gear_int"].shift(1)
    transitions = df[
        df["gear_prev"].notna()
        & df["gear_int"].notna()
        & (df["gear_prev"] != df["gear_int"])
    ]

    for idx, row in transitions.iterrows():
        g_prev = int(row["gear_prev"])
        g_curr = int(row["gear_int"])
        if g_prev <= g_curr:
            continue
        row_label = f"{g_prev} -> {g_curr} Shift"
        if row_label not in DOWN_ROWS:
            continue

        speed = float(row["speed_mph"])
        if speed < MIN_SPEED or speed > MAX_SPEED:
            continue

        time_event = float(row["time"])
        window = df[(df["time"] >= time_event - WINDOW_BACK) & (df["time"] <= time_event + WINDOW_FWD)]
        if window.empty:
            continue

        tps_peak = float(window["tps"].max())
        if tps_peak > TPS_PEAK_MAX:
            continue

        pedal_peak = float(window["pedal"].max()) if "pedal" in window.columns else np.nan
        if np.isfinite(pedal_peak) and pedal_peak > PEDAL_MAX:
            continue

        dthr_peak = float(window["dthr_dt"].max())
        if dthr_peak < DTHR_MIN:
            continue

        brake_curr = float(row.get("brake", 0.0))
        if brake_curr > BRAKE_MAX:
            continue

        tps_bin = snap_tps_bin(tps_peak)
        events.append({
            "file": os.path.basename(filename),
            "time_s": time_event,
            "speed_mph": speed,
            "tps": float(row["tps"]),
            "tps_peak": tps_peak,
            "pedal": pedal_peak,
            "gear_before": g_prev,
            "gear_after": g_curr,
            "row": row_label,
            "tps_bin": tps_bin,
            "dthr_dt_peak": dthr_peak,
        })
    return events


def blank_delta_table():
    cols = ["mph"] + [str(v) for v in TPS_AXIS] + ["%"]
    df = pd.DataFrame(index=DOWN_ROWS, columns=cols, dtype=object)
    df.iloc[:,1:-1] = 0.0
    df.iloc[:,0] = df.index.astype(object)
    df.iloc[:,-1] = "%"
    return df


def load_baseline(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline DOWN table missing: {path}")
    return pd.read_csv(path, sep="\t")


def build_delta(events_df, baseline_df):
    delta = blank_delta_table()
    if events_df.empty:
        return delta

    grouped = events_df.groupby(["row","tps_bin"])["speed_mph"].agg(count="count", median="median", std="std").reset_index()
    for _, row in grouped.iterrows():
        row_label = row["row"]
        tps_bin = int(row["tps_bin"])
        if row_label not in DOWN_ROWS:
            continue
        if str(tps_bin) not in delta.columns:
            continue

        count = int(row["count"])
        std = float(row["std"]) if not np.isnan(row["std"]) else float("inf")
        median = float(row["median"])

        if count < MIN_COUNT:
            continue
        if std > STD_MAX:
            continue

        row_mask = baseline_df["mph"] == row_label
        base_vals = baseline_df.loc[row_mask, str(tps_bin)]
        if base_vals.empty:
            continue
        base_mph = float(base_vals.iloc[0])
        gap = base_mph - median
        if gap < BASE_GAP_MIN:
            continue

        delta_val = min(DELTA_MPH, max(0.0, gap))
        delta.loc[delta["mph"] == row_label, str(tps_bin)] = round(delta_val,1)

    return delta


def main():
    parser = argparse.ArgumentParser(description="Tahoe STOPGO pass (BESTINTERP)")
    parser.add_argument("--logs-glob", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline-down", default=DEFAULT_BASELINE)
    args = parser.parse_args()

    logs = glob.glob(args.logs_glob)
    if not logs:
        raise RuntimeError(f"No logs matched {args.logs_glob}")

    os.makedirs(args.out_dir, exist_ok=True)

    all_events = []
    for path in logs:
        print(f"[STOPGO_TAHOE] scanning {path}")
        df_raw = pd.read_csv(path, low_memory=False)
        df = ensure_canon(df_raw, path)
        events = detect_stopgo_events(df, path)
        print(f"[STOPGO_TAHOE]   found {len(events)} events")
        all_events.extend(events)

    events_cols = [
        "file","time_s","speed_mph","tps","tps_peak","pedal",
        "gear_before","gear_after","row","tps_bin","dthr_dt_peak"
    ]
    events_df = pd.DataFrame(all_events, columns=events_cols)
    events_path = os.path.join(args.out_dir, "STOPGO__EVENTS_RAW_DEBUG__TAHOE.csv")
    events_df.to_csv(events_path, index=False)
    print(f"[STOPGO_TAHOE] events debug -> {events_path}")

    baseline_df = load_baseline(args.baseline_down)
    delta_df = build_delta(events_df, baseline_df)
    delta_path = os.path.join(args.out_dir, "STOPGO__SHIFT_DOWN__DELTA.tsv")
    delta_df.to_csv(delta_path, sep="\t", index=False)
    nonzero = (delta_df.drop(columns=["mph","%"], errors="ignore") != 0).sum().sum()
    total = delta_df.drop(columns=["mph","%"], errors="ignore").size
    print(f"[STOPGO_TAHOE] delta -> {delta_path} | nonzero {nonzero}/{total}")

if __name__ == "__main__":
    main()
