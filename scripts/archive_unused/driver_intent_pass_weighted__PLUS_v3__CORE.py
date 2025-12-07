#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
driver_intent_pass_weighted__PLUS_v3__STANDALONE__fixB.py
--------------------------------------------------------
Patched INTENT pass (supersedes fixA):
- FIX: Build speed window mask first, then filter df, then recompute all Series
       so boolean masks always align with df after filtering (prevents IndexError).
- Keep previous fixes:
  * Correct dt usage (no dt.diff()), robust fill for zeros/NaNs.
  * Correct nonzero counting logic.
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tps_phases import is_cruise_band

TPS_AXIS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
UP_ROWS = ["1 -> 2 Shift","2 -> 3 Shift","3 -> 4 Shift","4 -> 5 Shift","5 -> 6 Shift"]
TCC_REL_ROWS = ["3rd Release","4th Release","5th Release","6th Release"]

ALT_NAMES = {
    "pedal": ["Accelerator Pedal Position","Accelerator Pedal (%)","Pedal Position","pedal_pct","pedal"],
    # TPS: require canonical throttle_pct__canon (no fallback to raw TPS)
    "throttle": ["throttle_pct__canon"],
    "speed": ["speed_mph__canon","speed_mph","Vehicle Speed (SAE)","VSS mph","mph","vss_mph"],
    "gear": ["gear_actual","Trans Current Gear","Transmission Current Gear","Trans Current Gear (SAE)","Current Gear","Gear Current","Trans_Gear_Current","G_Cur","Gear_Actual","GEAR_ACTUAL"],
    "time": ["time_s","Offset","Time (s)","time","elapsed_s"],
    "brake": ["brake","Brake","Brake Applied","Brake Switch","Brake Pressure","Brake Pressure (kPa)"],
    "tcc_lock": ["tcc_locked_built__canon","tcc_locked_built","TCC Locked Built","TCC Lock","TCC Lock Flag"],
    "latg": ["Lateral Acceleration","lateral acceleration","Lat Accel","lat_g"],
    "yaw": ["Yaw Rate","yaw rate","Yaw Rate (deg/s)","yaw_rate_deg_s"],
    "steer": ["Steering Wheel Position","Steer Angle","steering_angle_deg"],
    "ect_f": ["Engine Coolant Temp (SAE)","ECT (F)","ECT_F"],
    "tft_f": ["Trans Fluid Temp","TFT (F)","TFT_F"],
    "tcc_slip": ["tcc_slip_fused","TCC Slip","TCC Slip (RPM)"],
}

def find_col(df, pref, alts):
    if pref and pref in df.columns: return pref, pref
    for a in alts:
        if a in df.columns: return a, a
    lclook = {c.lower(): c for c in df.columns}
    for a in alts:
        if a.lower() in lclook: return lclook[a.lower()], lclook[a.lower()]
    return None, None

def tps_bin(v):
    v = float(np.clip(v, 0, 100))
    return min(TPS_AXIS, key=lambda x: abs(x - v))

def make_blank(rows):
    cols = ["mph"] + [str(x) for x in TPS_AXIS] + ["%"]
    df = pd.DataFrame(index=rows, columns=cols)
    df.iloc[:, :] = 0.0
    df["mph"] = rows
    return df

def main():
    ap = argparse.ArgumentParser(description="Standalone INTENT pass (driver intent deltas) — fixB.")
    ap.add_argument("--logs-glob", required=True)
    ap.add_argument("--out-dir", required=True)
    # Preferred column names (project-canonical)
    ap.add_argument("--pedal-column", default="Accelerator Pedal Position")
    # TPS: prefer canonical throttle_pct__canon
    ap.add_argument("--throttle-column", default="throttle_pct__canon")
    ap.add_argument("--speed-column", default="speed_mph__canon")
    ap.add_argument("--gear-column", default="gear_actual")
    ap.add_argument("--time-column", default="Offset")
    ap.add_argument("--brake-column", default="brake")
    ap.add_argument("--tcc-column", default="tcc_locked_built__canon")
    ap.add_argument("--latg-column", default="Lateral Acceleration")
    ap.add_argument("--yaw-column", default="Yaw Rate")
    ap.add_argument("--steer-column", default="Steering Wheel Position")
    ap.add_argument("--ect-column", default="Engine Coolant Temp (SAE)")
    ap.add_argument("--tft-column", default="Trans Fluid Temp")
    ap.add_argument("--tcc-slip-column", default="tcc_slip_fused")
    # Feature toggles
    ap.add_argument("--nogates", action="store_true")
    ap.add_argument("--no-chassis", action="store_true")
    ap.add_argument("--no-tcc-lock", action="store_true")
    ap.add_argument("--require-warm", action="store_true")
    # Thresholds
    ap.add_argument("--thr-rate-pedal", type=float, default=12.0)
    ap.add_argument("--thr-rate-throttle", type=float, default=9.0)
    ap.add_argument("--min-speed", type=float, default=5.0)
    ap.add_argument("--max-speed", type=float, default=85.0)
    ap.add_argument("--pass-win-min", type=float, default=35.0)
    ap.add_argument("--pass-win-max", type=float, default=55.0)
    ap.add_argument("--brake-release-window", type=float, default=1.2)
    ap.add_argument("--delta-up", type=float, default=0.2)
    ap.add_argument("--delta-tcc", type=float, default=-0.3)
    ap.add_argument("--ect-warm-f", type=float, default=100.0)
    ap.add_argument("--tft-warm-f", type=float, default=100.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    up_df = make_blank(UP_ROWS)
    tcc_rel_df = make_blank(TCC_REL_ROWS)

    files = sorted(glob.glob(args.logs_glob))
    if not files:
        print("No files matched:", args.logs_glob)
        sys.exit(2)

    drop_tcc = args.nogates or args.no_tcc_lock
    drop_chassis = args.nogates or args.no_chassis

    total_rows = 0
    summary_lines = []

    for fp in files:
        df = pd.read_csv(fp, low_memory=False)
        total_rows += len(df)

        # Column resolution helper
        def getcol(key, pref, altkey):
            col,_ = find_col(df, pref, ALT_NAMES[altkey])
            summary_lines.append(f"{'OK' if col else 'FALLBACK'} {key}: " + (f"contains->{col}" if col else "missing"))
            return col

        ped_pref = getcol("pedal",    args.pedal_column,    "pedal")
        thr_pref = getcol("throttle", args.throttle_column, "throttle")
        if thr_pref is None:
            raise RuntimeError("INTENT: required column 'throttle_pct__canon' missing in CLEAN_FULL")
        spd_pref = getcol("speed",    args.speed_column,    "speed")
        gear_pref= getcol("gear",     args.gear_column,     "gear")
        time_pref= getcol("time",     args.time_column,     "time")
        brk_pref = getcol("brake",    args.brake_column,    "brake")
        ect_pref = getcol("ECT_F",    args.ect_column,      "ect_f")
        tft_pref = getcol("TFT_F",    args.tft_column,      "tft_f")
        tcc_pref = None if drop_tcc else getcol("tcc", args.tcc_column, "tcc_lock")
        latg_pref= None if drop_chassis else getcol("lat_g", args.latg_column, "latg")
        yaw_pref = None if drop_chassis else getcol("yaw_rate", args.yaw_column, "yaw")
        steer_pref=None if drop_chassis else getcol("steer", args.steer_column, "steer")
        slip_pref= getcol("tcc_slip", args.tcc-slip-column if False else args.tcc_slip_column, "tcc_slip")  # keep arg name

        # Warm gate
        if args.require_warm and ect_pref and tft_pref:
            ect_all = pd.to_numeric(df[ect_pref], errors="coerce")
            tft_all = pd.to_numeric(df[tft_pref], errors="coerce")
            warm_mask = (ect_all >= args.ect_warm_f) & (tft_all >= args.tft_warm_f)
        else:
            warm_mask = pd.Series(True, index=df.index)

        # Speed window gate — build mask BEFORE slicing df
        if spd_pref:
            spd_all = pd.to_numeric(df[spd_pref], errors="coerce")
            speed_keep = (spd_all >= args.min_speed) & (spd_all <= args.max_speed)
        else:
            speed_keep = pd.Series(True, index=df.index)

        keep = warm_mask & speed_keep
        if not keep.any():
            continue

        # Slice once, then recompute all Series on the filtered df so shapes align
        df = df.loc[keep].copy()

        # Recompute series from filtered df
        spd = pd.to_numeric(df[spd_pref], errors="coerce") if spd_pref else pd.Series(np.nan, index=df.index)
        ped = pd.to_numeric(df[ped_pref], errors="coerce") if ped_pref else pd.Series(0.0, index=df.index)
        thr = pd.to_numeric(df[thr_pref], errors="coerce") if thr_pref else pd.Series(0.0, index=df.index)
        brk = pd.to_numeric(df[brk_pref], errors="coerce") if brk_pref else pd.Series(0.0, index=df.index)
        gear= pd.to_numeric(df[gear_pref],errors="coerce") if gear_pref else pd.Series(np.nan, index=df.index)
        if time_pref and time_pref in df.columns:
            t = pd.to_numeric(df[time_pref], errors="coerce").astype("float64")
            dt = t.diff().replace(0, np.nan).bfill().ffill()
        else:
            dt = pd.Series(0.01, index=df.index, dtype="float64")

        # Rates
        rate_ped = ped.diff() / dt
        rate_thr = thr.diff() / dt

        # Intent & pass window masks — ALWAYS build off filtered df
        intent_mask = (rate_ped >= args.thr_rate_pedal) | (rate_thr >= args.thr_rate_throttle)
        pass_mask   = (spd >= args.pass_win_min) & (spd <= args.pass_win_max)
        brake_ok    = (brk <= 0.5) | brk.isna()

        sel = intent_mask & pass_mask & brake_ok
        if not sel.any():
            continue

        # Group nearby hits (0.5s buckets) and pick within-bucket points
        ctime = dt.fillna(0).cumsum()
        grp = (ctime / 0.5).astype("int64")
        # Ensure grp index aligns to df index for transform
        grp.index = df.index
        pick = sel.groupby(grp).transform("any") & sel
        idxs = df.index[pick]

        # Precompute baseline TPS as previous-sample throttle
        thr_prev = thr.shift(1)

        for i in idxs:
            g = int(gear.loc[i]) if pd.notna(gear.loc[i]) else None
            if g is None or g < 1 or g > 6: 
                continue
            # Baseline TPS before ramp: previous sample's throttle
            base_tps = thr_prev.loc[i] if pd.notna(thr_prev.loc[i]) else np.nan
            if not is_cruise_band(base_tps):
                # Only treat intent when coming from real cruise (8–18% TPS)
                continue

            thr_i = thr.loc[i] if pd.notna(thr.loc[i]) else 0.0
            tps_bin_str = str(tps_bin(thr_i))

            if 1 <= g <= 5:
                up_df.loc[f"{g} -> {g+1} Shift", tps_bin_str] = float(args.delta_up)
            if 3 <= g <= 6:
                suffix = {1:"st",2:"nd",3:"rd"}.get(g,"th")
                rowr = f"{g}{suffix} Release"
                if rowr in tcc_rel_df.index:
                    tcc_rel_df.loc[rowr, tps_bin_str] = float(args.delta_tcc)

    # Write outputs
    out_up = Path(args.out_dir) / "INTENT__SHIFT_UP__DELTA.tsv"
    out_tcc = Path(args.out_dir) / "INTENT__TCC_RELEASE__DELTA.tsv"
    up_df.to_csv(out_up, sep="\t", index=False, float_format="%.1f")
    tcc_rel_df.to_csv(out_tcc, sep="\t", index=False, float_format="%.1f")

    # Nonzero counts
    up_num  = up_df.iloc[:, 1:-1].apply(pd.to_numeric, errors="coerce").fillna(0)
    tcc_num = tcc_rel_df.iloc[:, 1:-1].apply(pd.to_numeric, errors="coerce").fillna(0)
    nz_up   = int((up_num != 0).to_numpy().sum())
    nz_tcc  = int((tcc_num != 0).to_numpy().sum())

    with open(Path(args.out_dir) / "INTENT__RUN_SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write(f"INTENT__SHIFT_UP__DELTA.tsv: nonzero {nz_up}/{len(UP_ROWS)*len(TPS_AXIS)}\n")
        f.write(f"INTENT__TCC_RELEASE__DELTA.tsv: nonzero {nz_tcc}/{len(TCC_REL_ROWS)*len(TPS_AXIS)}\n")
        f.write(f"Total rows: {total_rows}\n")

if __name__ == "__main__":
    main()
