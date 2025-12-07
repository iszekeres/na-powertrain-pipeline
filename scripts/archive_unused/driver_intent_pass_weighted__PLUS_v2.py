
#!/usr/bin/env python3
# driver_intent_pass_weighted__PLUS_v2.py
# Same as _PLUS, but adds explicit FALLBACK/INFO logging and writes a run summary.
#
# Key additions:
#  - Prints which columns were used and where fallbacks occurred (e.g., TPS from pedal instead of throttle).
#  - Reports if chassis gating (lat/yaw/steer) was skipped due to missing columns.
#  - Reports if TCC lock inference fallback (gear>=3) was used.
#  - Emits INTENT__RUN_SUMMARY.txt with hit counts and gating effects.

import argparse, glob, os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path

TPS_AXIS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
UP_ROWS = ["1 -> 2 Shift","2 -> 3 Shift","3 -> 4 Shift","4 -> 5 Shift","5 -> 6 Shift"]
TCC_REL_ROWS = ["3rd Release","4th Release","5th Release","6th Release"]

def tps_bin_idx(val):
    if pd.isna(val): return None
    v = float(val); v = min(max(v, 0.0), 100.0)
    idx = 0
    for i,ax in enumerate(TPS_AXIS):
        if v >= ax: idx = i
        else: break
    return idx

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def diff_rate(series, time_s, clip=None):
    s = safe_num(series).values.astype(float)
    t = safe_num(time_s).values.astype(float)
    ds = np.diff(s, prepend=s[:1])
    dt = np.diff(t, prepend=t[:1])
    dt[dt<=0] = np.nan
    rate = ds / dt
    if clip:
        rate = np.clip(rate, -clip, clip)
    return pd.Series(rate)

def first_present(df, ordered_candidates):
    cols = df.columns.tolist()
    # exact first
    for c in ordered_candidates:
        if c in cols:
            return c, f"exact:{c}"
    # contains
    lc = [x.lower() for x in cols]
    for c in ordered_candidates:
        cl = c.lower()
        for i,name in enumerate(lc):
            if cl in name:
                return cols[i], f"contains:{c}->{cols[i]}"
    return None, "missing"

def main():
    ap = argparse.ArgumentParser(description="Enhanced driver intent pass with fallback logging")
    ap.add_argument("--logs-glob", required=True, help=r"Glob of CLEAN_FULL CSVs (e.g., .\newlogs\cleaned\__trans_focus__clean_FULL__*withbrake*.csv)")
    ap.add_argument("--out-dir", required=True, help=r"Directory to write outputs")
    # Columns
    ap.add_argument("--pedal-column",    default="pedal_pct")
    ap.add_argument("--throttle-column", default="throttle_pct")
    ap.add_argument("--speed-column",    default="speed_mph")
    ap.add_argument("--gear-column",     default="gear_actual")
    ap.add_argument("--time-column",     default="time_s")
    ap.add_argument("--brake-column",    default="brake")
    ap.add_argument("--tcc-column",      default="tcc_locked_built")
    ap.add_argument("--latg-column",     default=None, help="Optional lateral acceleration column (|g|)")
    ap.add_argument("--yaw-column",      default=None, help="Optional yaw rate column (deg/s)")
    ap.add_argument("--steer-column",    default=None, help="Optional steering angle column (deg)")
    # Thresholds / gating
    ap.add_argument("--thr-rate-pedal",    type=float, default=20.0, help="Pedal surge threshold (%/s)")
    ap.add_argument("--thr-rate-throttle", type=float, default=15.0, help="Throttle surge threshold (%/s)")
    ap.add_argument("--min-speed", type=float, default=8.0)
    ap.add_argument("--max-speed", type=float, default=80.0)
    ap.add_argument("--lat-g",     type=float, default=0.12, help="Max |lat g| to count as straight (skip if absent)")
    ap.add_argument("--yaw-rate",  type=float, default=20.0, help="Max |yaw deg/s| to count as straight (skip if absent)")
    ap.add_argument("--steer-abs", type=float, default=60.0, help="Max |steer deg| to count as straight (skip if absent)")
    ap.add_argument("--brake-release-window", type=float, default=0.6, help="Seconds after brake release to accept intent")
    # Deltas
    ap.add_argument("--delta-up",  type=float, default=0.2, help="mph added to SHIFT UP thresholds per normalized hit")
    ap.add_argument("--delta-tcc", type=float, default=-0.3, help="mph added to TCC RELEASE per normalized hit (negative => earlier release)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.logs_glob))
    if not files:
        print(f"[INTENT_PLUS_V2] No files matched {args.logs_glob}")
        sys.exit(0)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    up_hits  = {row: np.zeros(len(TPS_AXIS), dtype=float) for row in UP_ROWS}
    tcc_hits = {row: np.zeros(len(TPS_AXIS), dtype=float) for row in TCC_REL_ROWS}
    run_summary = {"files": [], "totals": {"rows":0, "surge":0, "good":0, "tps_from_pedal_ratio":0.0, "fallbacks":[]}}

    for f in files:
        info = {"file": os.path.basename(f), "rows":0, "surge":0, "good":0, "fallbacks":[]}
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[INTENT_PLUS_V2] READ FAIL {f}: {e}")
            continue

        def pick(name, ordered):
            col, how = first_present(df, ordered)
            if how.startswith("contains:"):
                print(f"[INFO] {info['file']}: using column {col} (matched by contains from {name})")
                info["fallbacks"].append(f"{name}->{col}[contains]")
            elif how == "missing":
                print(f"[FALLBACK] {info['file']}: missing {name}; will attempt alternate logic")
                info["fallbacks"].append(f"{name}[missing]")
            return col

        col_ped = pick("pedal",    [args.pedal_column, "Accelerator Pedal", "Accelerator Pedal Position"])
        col_thr = pick("throttle", [args.throttle-column if hasattr(args,'throttle-column') else args.throttle_column, "Throttle Position", "Throttle Angle"])
        col_spd = pick("speed",    [args.speed_column, "Vehicle Speed", "VSS", "speed_mph"])
        col_gear= pick("gear",     [args.gear_column, "Trans Current Gear", "Transmission Current Gear", "GEAR_ACTUAL"])
        col_time= pick("time",     [args.time_column, "offset"])
        col_brk = pick("brake",    [args.brake_column, "Brake Pressure", "Brake Switch", "Brake Applied"])
        col_tcc = pick("tcc",      [args.tcc_column, "tcc_locked_built"])

        # chassis (optional)
        col_lat = pick("lat_g",    [x for x in [args.latg_column] if x] + ["lateral acceleration", "Lateral Acceleration", "Lat Accel"])
        col_yaw = pick("yaw_rate", [x for x in [args.yaw_column] if x] + ["yaw rate", "Yaw Rate"])
        col_str = pick("steer",    [x for x in [args.steer_column] if x] + ["Steering Wheel Position", "Steering Angle"])

        need = [col_ped, col_thr, col_spd, col_gear, col_time]
        if any(c is None for c in need):
            print(f"[SKIP] {info['file']}: missing a required column among {need}")
            run_summary["files"].append(info); continue

        info["rows"] = len(df)
        run_summary["totals"]["rows"] += info["rows"]

        pedal   = safe_num(df[col_ped])
        thr     = safe_num(df[col_thr])
        speed   = safe_num(df[col_spd])
        gear    = safe_num(df[col_gear]).astype("Int64")
        time_s  = safe_num(df[col_time])

        dpdt = diff_rate(pedal, time_s, clip=200)
        dtdt = diff_rate(thr,   time_s, clip=200)

        # chassis gating
        straight = pd.Series(True, index=df.index)
        if col_lat is None:
            print(f"[SKIP] {info['file']}: no lat_g column -> lat gating off")
        else:
            straight &= safe_num(df[col_lat]).abs() <= args.lat_g

        if col_yaw is None:
            print(f"[SKIP] {info['file']}: no yaw_rate column -> yaw gating off")
        else:
            straight &= safe_num(df[col_yaw]).abs() <= args.yaw_rate

        if col_str is None:
            print(f"[SKIP] {info['file']}: no steer column -> steer gating off")
        else:
            straight &= safe_num(df[col_str]).abs() <= args.steer_abs

        spdbg = (speed >= args.min_speed) & (speed <= args.max_speed)

        # brake release window
        br_ok = pd.Series(True, index=df.index)
        if col_brk is not None:
            br = safe_num(df[col_brk]).fillna(0)
            br_state = (br > 0).astype(int)
            prev = br_state.shift(1).fillna(br_state.iloc[0])
            release = (prev == 1) & (br_state == 0)
            rel_t = pd.Series(np.nan, index=df.index, dtype=float)
            last_t = np.nan
            ts = time_s.values
            for i, r in enumerate(release.values):
                if r:
                    last_t = ts[i]
                rel_t.iloc[i] = np.inf if np.isnan(last_t) else (ts[i]-last_t)
            br_ok = (br_state == 0) | (rel_t <= args.brake_release_window)
        else:
            print(f"[FALLBACK] {info['file']}: no brake column -> no brake gating")

        surge = (dpdt >= args.thr_rate_pedal) | (dtdt >= args.thr_rate_throttle)
        info["surge"] = int(surge.sum())

        # TPS source & fallback ratio
        tps = thr.copy()
        tps_isna = tps.isna()
        used_pedal_mask = tps_isna & (~pedal.isna())
        tps[used_pedal_mask] = pedal[used_pedal_mask]
        pedal_fallback_ratio = float(used_pedal_mask.sum())/len(tps)
        info["tps_from_pedal_ratio"] = round(pedal_fallback_ratio, 4)
        if pedal_fallback_ratio > 0:
            print(f"[FALLBACK] {info['file']}: TPS from pedal for {pedal_fallback_ratio*100:.2f}% of rows (throttle NaN)")        
        if col_thr is None and col_ped is not None:
            print(f"[FALLBACK] {info['file']}: throttle column missing entirely; using pedal for TPS bins")

        good = surge & straight & spdbg & br_ok & gear.notna()
        info["good"] = int(good.sum())
        run_summary["totals"]["surge"] += info["surge"]
        run_summary["totals"]["good"]  += info["good"]

        # TCC lock fallback
        if col_tcc is not None:
            tcc_lock = safe_num(df[col_tcc]).fillna(0) > 0.5
        else:
            print(f"[FALLBACK] {info['file']}: no TCC lock signal; assume lockable when gear >= 3")
            tcc_lock = (gear >= 3)

        # UP hits
        for g in range(1,6):
            mask = good & (gear == g)
            if mask.any():
                idxs = np.where(mask.values)[0]
                for idx in idxs:
                    b = tps_bin_idx(tps.iloc[idx])
                    if b is None: 
                        continue
                    up_hits[UP_ROWS[g-1]][b] += 1.0

        # TCC release hits
        for g,rowname in zip([3,4,5,6], TCC_REL_ROWS):
            mask = good & (gear == g) & tcc_lock
            if mask.any():
                idxs = np.where(mask.values)[0]
                for idx in idxs:
                    b = tps_bin_idx(tps.iloc[idx])
                    if b is None: 
                        continue
                    tcc_hits[rowname][b] += 1.0

        run_summary["files"].append(info)

    # Normalize and convert to mph deltas
    def hits_to_delta(hdict, step):
        m = np.array(list(hdict.values()))
        colsum = m.sum(axis=0)
        nz = colsum[colsum>0]
        scale = np.percentile(nz, 95) if nz.size else 1.0
        if scale <= 0: scale = 1.0
        out = {k: (v/scale)*step for (k,v) in hdict.items()}
        return out

    up_delta  = hits_to_delta(up_hits,  args.delta_up)
    tcc_delta = hits_to_delta(tcc_hits, args.delta_tcc)

    up_path  = Path(args.out_dir) / "INTENT__SHIFT_UP__DELTA.tsv"
    tcc_path = Path(args.out_dir) / "INTENT__TCC_RELEASE__DELTA.tsv"

    # Write UP delta
    df_up = pd.DataFrame([up_delta[row] for row in UP_ROWS], columns=[str(x) for x in TPS_AXIS])
    df_up.insert(0, "mph", UP_ROWS); df_up["%"] = np.nan
    df_up.to_csv(up_path, sep="\t", index=False, float_format="%.1f")

    # Write TCC RELEASE delta
    df_tc = pd.DataFrame([tcc_delta[row] for row in TCC_REL_ROWS], columns=[str(x) for x in TPS_AXIS])
    df_tc.insert(0, "mph", TCC_REL_ROWS); df_tc["%"] = np.nan
    df_tc.to_csv(tcc_path, sep="\t", index=False, float_format="%.1f")

    # Summary file
    run_summary["totals"]["tps_from_pedal_ratio"] = round(
        np.mean([f.get("tps_from_pedal_ratio",0.0) for f in run_summary["files"]]) if run_summary["files"] else 0.0, 4
    )
    summary_txt = Path(args.out_dir) / "INTENT__RUN_SUMMARY.txt"
    with open(summary_txt, "w", encoding="utf-8") as fh:
        fh.write(f"Files: {len(run_summary['files'])}\n")
        fh.write(f"Rows total: {run_summary['totals']['rows']}\n")
        fh.write(f"Surge hits total: {run_summary['totals']['surge']}\n")
        fh.write(f"Good (post-gating) total: {run_summary['totals']['good']}\n")
        fh.write(f"Mean TPS-from-pedal ratio: {run_summary['totals']['tps_from_pedal_ratio']}\n")
        fh.write("\nPer-file:\n")
        for f in run_summary["files"]:
            fh.write(f"- {f['file']}: rows={f['rows']} surge={f['surge']} good={f['good']} tps_from_pedal={f.get('tps_from_pedal_ratio',0.0)} fallbacks={';'.join(f['fallbacks'])}\n")

    print(f"[INTENT_PLUS_V2] WROTE {up_path}")
    print(f"[INTENT_PLUS_V2] WROTE {tcc_path}")
    print(f"[INTENT_PLUS_V2] WROTE {summary_txt}")

if __name__ == "__main__":
    main()
