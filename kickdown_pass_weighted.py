# kickdown_pass_weighted.py  (BESTINTERP-friendly)
import argparse, glob, os, sys
import numpy as np
import pandas as pd
from tps_phases import is_kickdown_band

TPS_AXIS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
DOWN_ROWS = ["2 -> 1 Shift","3 -> 2 Shift","4 -> 3 Shift","5 -> 4 Shift","6 -> 5 Shift"]
TPS_AXIS_ARR = np.array(TPS_AXIS, dtype=float)
DEFAULT_OUT_DIR = os.path.join("newlogs", "output", "02_passes", "KICKDOWN_DEBUG")

CANON_FALLBACKS = {
    "time_s__canon": [
        "time_s__canon","time_s","Time_s","Time","Time (s)","Time(s)","offset","Offset","offset__canon"
    ],
    "speed_mph__canon": [
        "speed_mph__canon","speed_mph","Vehicle Speed (SAE)","Vehicle Speed","vehicle_speed_mph","Speed"
    ],
    # TPS: require canonical throttle_pct__canon (no fallback)
    "throttle_pct__canon": [
        "throttle_pct__canon",
    ],
    "gear_actual__canon": [
        "gear_actual__canon","gear_actual","Trans Current Gear","Transmission Gear",
        "Trans Gear","Gear Actual","gear_cmd","gear_cmd__canon"
    ],
    "pedal_pct__canon": [
        "pedal_pct__canon","pedal_pct","Accelerator Pedal Position",
        "Accelerator Pedal Position (%)",
        "Accelerator Pedal (%)"
    ],
    "brake__canon": [
        "brake__canon","brake","Brake","Brake Pressure","Brake Pedal Position"
    ],
}
CORE_CANON = ["time_s__canon","speed_mph__canon","throttle_pct__canon","gear_actual__canon"]


def ensure_canon(df, path):
    df = df.copy()
    for canon, candidates in CANON_FALLBACKS.items():
        need = canon not in df.columns or df[canon].dropna().empty
        if not need:
            continue
        for cand in candidates:
            if cand in df.columns and not df[cand].dropna().empty:
                df[canon] = df[cand]
                need = False
                break
        if need and canon in CORE_CANON:
            raise RuntimeError(f"{path}: missing required column {canon}")
    missing = [c for c in CORE_CANON if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path}: missing required canonical columns {missing}")
    df = df.dropna(subset=CORE_CANON, how="any").copy()
    return df


def nanmax_safe(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return arr.max()


def nearest_tps_label(value):
    if value is None or not np.isfinite(value):
        return str(TPS_AXIS[0])
    idx = int(np.argmin(np.abs(TPS_AXIS_ARR - float(value))))
    return str(TPS_AXIS[idx])


def blank_down_table():
    cols = ["mph"] + [str(x) for x in TPS_AXIS] + ["%"]
    df = pd.DataFrame(index=DOWN_ROWS, columns=cols, dtype=object)
    df.iloc[:, 1:-1] = 0.0
    df.iloc[:, 0] = df.index.astype(object)
    df.iloc[:, -1] = "%"
    return df


def main():
    ap = argparse.ArgumentParser(description="Loose kickdown detector for BESTINTERP logs.")
    ap.add_argument("--logs-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-speed", type=float, default=20.0, help="Minimum speed for candidate events (mph)")
    ap.add_argument("--thr-rate", type=float, default=2.0, help="Minimum dTPS/dt within the evaluation window (percent per second)")
    ap.add_argument("--pedal-min", type=float, default=25.0, help="Minimum TPS/Pedal level within the window (percent)")
    ap.add_argument("--min-hits", type=int, default=6, help="Minimum samples per TPS bin for a delta")
    ap.add_argument("--delta-mph", type=float, default=0.3, help="Delta magnitude when a bin qualifies (mph)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.logs_glob))
    if not files:
        print(f"[KICKDOWN] no files for {args.logs_glob}", file=sys.stderr)
        sys.exit(2)

    counts = {row: pd.Series(0, index=[str(x) for x in TPS_AXIS], dtype=int) for row in DOWN_ROWS}
    events = []

    for path in files:
        df = pd.read_csv(path, low_memory=False)
        print(f"[KICKDOWN] scanning {os.path.basename(path)}")
        try:
            df = ensure_canon(df, os.path.basename(path))
        except RuntimeError as exc:
            print(f"[WARN] {exc}; skipping")
            continue

        gear_int = df["gear_actual__canon"].astype(float).round().astype("Int64")
        df = df.assign(gear_int=gear_int)

        time = pd.to_numeric(df["time_s__canon"], errors="coerce")
        speed = pd.to_numeric(df["speed_mph__canon"], errors="coerce")
        thr = pd.to_numeric(df["throttle_pct__canon"], errors="coerce").clip(0, 100)
        pedal = pd.to_numeric(df.get("pedal_pct__canon"), errors="coerce") if "pedal_pct__canon" in df.columns else pd.Series(np.nan, index=df.index)
        brake = pd.to_numeric(df.get("brake__canon"), errors="coerce") if "brake__canon" in df.columns else pd.Series(0.0, index=df.index)

        dt = time.diff().replace(0, np.nan)
        dthr = thr.diff() / dt
        dthr = dthr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        time_vals = time.to_numpy()
        speed_vals = speed.to_numpy()
        tps_vals = thr.to_numpy()
        pedal_vals = pedal.to_numpy()
        brake_vals = brake.to_numpy()
        dthr_vals = dthr.to_numpy()
        gear_vals = df["gear_int"].to_numpy(dtype=float)

        dgear = np.diff(gear_vals)
        down_idx = np.where(dgear < 0)[0]

        for idx in down_idx:
            before = gear_vals[idx]
            after = gear_vals[idx + 1]
            if not (np.isfinite(before) and np.isfinite(after)):
                continue
            if not (1 <= before <= 6 and 1 <= after <= 6):
                continue
            j = idx + 1
            time_event = time_vals[j]
            speed_event = speed_vals[j]
            if not (np.isfinite(time_event) and np.isfinite(speed_event)):
                continue
            if speed_event < args.min_speed:
                continue
            brake_val = brake_vals[j] if j < len(brake_vals) else 0.0
            # Skip if brake is ON (brake > 0.5)
            if np.isfinite(brake_val) and brake_val > 0.5:
                continue

            # TPS phase gating: only treat >=25% TPS as true kickdown intent
            tps_event = tps_vals[j]
            if not is_kickdown_band(tps_event):
                continue

            window_mask = (time_vals >= time_event - 0.2) & (time_vals <= time_event + 1.0)
            if not np.any(window_mask):
                continue

            peak_tps = nanmax_safe(tps_vals[window_mask])
            peak_pedal = nanmax_safe(pedal_vals[window_mask])
            peaks = [p for p in (peak_tps, peak_pedal) if np.isfinite(p)]
            combined_peak = max(peaks) if peaks else np.nan
            if not np.isfinite(combined_peak) or combined_peak < args.pedal_min:
                continue

            peak_rate = nanmax_safe(dthr_vals[window_mask])
            if not np.isfinite(peak_rate) or peak_rate < args.thr_rate:
                continue

            tps_event = tps_vals[j]
            pedal_event = pedal_vals[j]
            dthr_event = dthr_vals[j]

            row_label = f"{int(before)} -> {int(after)} Shift"
            bin_label = nearest_tps_label(tps_event)
            if row_label in counts:
                counts[row_label].loc[bin_label] = int(counts[row_label].loc[bin_label]) + 1

            events.append({
                "file": os.path.basename(path),
                "time_s": float(time_event),
                "speed_mph": float(speed_event),
                "tps": float(tps_event) if np.isfinite(tps_event) else np.nan,
                "pedal": float(pedal_event) if np.isfinite(pedal_event) else np.nan,
                "gear_before": float(before),
                "gear_after": float(after),
                "dthr_dt": float(dthr_event) if np.isfinite(dthr_event) else np.nan,
            })

    out = blank_down_table()
    for row, series in counts.items():
        for col in series.index:
            if int(series.loc[col]) >= args.min_hits:
                out.loc[row, col] = max(0.0, args.delta_mph)

    out_dir = os.path.dirname(args.out) or DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)

    event_cols = ["file","time_s","speed_mph","tps","pedal","gear_before","gear_after","dthr_dt"]
    events_df = pd.DataFrame(events, columns=event_cols)
    events_path = os.path.join(out_dir, "KICKDOWN__EVENTS_RAW_DEBUG.csv")
    events_df.to_csv(events_path, index=False)

    nz = int(np.count_nonzero(out.iloc[:,1:-1].to_numpy(dtype=float)))
    total = len(DOWN_ROWS) * len(TPS_AXIS)
    print(f"[KICKDOWN] wrote {args.out} | nonzero {nz}/{total}")

if __name__ == "__main__":
    main()
