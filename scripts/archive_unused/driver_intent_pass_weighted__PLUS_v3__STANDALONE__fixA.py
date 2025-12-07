#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
driver_intent_pass_weighted__PLUS_v3__STANDALONE__fixA.py
--------------------------------------------------------
Patched INTENT pass:
- Fix dt handling (use dt directly, not dt.diff())
- Silence deprecated fillna(method=...) by using .bfill()/.ffill()
- Fix nonzero counting (apply(pd.to_numeric) on DataFrame slice)
"""
import argparse, glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd

TPS_AXIS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
UP_ROWS = ["1 -> 2 Shift","2 -> 3 Shift","3 -> 4 Shift","4 -> 5 Shift","5 -> 6 Shift"]
TCC_REL_ROWS = ["3rd Release","4th Release","5th Release","6th Release"]

ALT_NAMES = {
    "pedal": ["Accelerator Pedal Position","Accelerator Pedal (%)","Pedal Position","pedal_pct","pedal"],
    "throttle": ["Throttle Position","Throttle (%)","Throttle (SAE)","throttle_pct","throttle"],
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

def make_blank_up():
    cols = ["mph"] + [str(x) for x in TPS_AXIS] + ["%"]
    df = pd.DataFrame(index=UP_ROWS, columns=cols)
    df.iloc[:, :] = 0.0
    df["mph"] = UP_ROWS
    return df

def make_blank_tcc_rel():
    cols = ["mph"] + [str(x) for x in TPS_AXIS] + ["%"]
    df = pd.DataFrame(index=TCC_REL_ROWS, columns=cols)
    df.iloc[:, :] = 0.0
    df["mph"] = TCC_REL_ROWS
    return df

def main():
    ap = argparse.ArgumentParser(description="Standalone INTENT pass (driver intent deltas) — fixA.")
    ap.add_argument("--logs-glob", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pedal-column", default="Accelerator Pedal Position")
    ap.add_argument("--throttle-column", default="Throttle Position")
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
    ap.add_argument("--nogates", action="store_true")
    ap.add_argument("--no-chassis", action="store_true")
    ap.add_argument("--no-tcc-lock", action="store_true")
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
    ap.add_argument("--require-warm", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    up_df = make_blank_up()
    tcc_rel_df = make_blank_tcc_rel()

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

        # Column resolution
        def getcol(key, pref, altkey):
            col,_ = find_col(df, pref, ALT_NAMES[altkey])
            summary_lines.append(f"{'OK' if col else 'FALLBACK'} {key}: " + (f"contains->{col}" if col else "missing"))
            return col

        ped_col = getcol("pedal",    args.pedal_column,    "pedal")
        thr_col = getcol("throttle", args.throttle_column, "throttle")
        spd_col = getcol("speed",    args.speed_column,    "speed")
        gear_col= getcol("gear",     args.gear_column,     "gear")
        time_col= getcol("time",     args.time_column,     "time")
        brk_col = getcol("brake",    args.brake_column,    "brake")
        ect_col = getcol("ECT_F",    args.ect_column,      "ect_f")
        tft_col = getcol("TFT_F",    args.tft_column,      "tft_f")
        tcc_col = None if drop_tcc else getcol("tcc", args.tcc_column, "tcc_lock")
        latg_col= None if drop_chassis else getcol("lat_g", args.latg_column, "latg")
        yaw_col = None if drop_chassis else getcol("yaw_rate", args.yaw_column, "yaw")
        steer_col=None if drop_chassis else getcol("steer", args.steer_column, "steer")
        slip_col= getcol("tcc_slip", args.tcc_slip_column, "tcc_slip")

        # Warm filter
        if args.require_warm and ect_col and tft_col:
            ect = pd.to_numeric(df[ect_col], errors="coerce")
            tft = pd.to_numeric(df[tft_col], errors="coerce")
            df = df[(ect>=args.ect_warm_f) & (tft>=args.tft_warm_f)]
            if df.empty: continue

        # Speed window for pass detection
        if spd_col and spd_col in df.columns:
            spd = pd.to_numeric(df[spd_col], errors="coerce")
            df = df[(spd>=args.min_speed) & (spd<=args.max_speed)]
            if df.empty: continue

        # Time delta
        if time_col and time_col in df.columns:
            t = pd.to_numeric(df[time_col], errors="coerce").astype("float64")
            dt = t.diff()
            dt = dt.replace(0, np.nan).bfill().ffill()
        else:
            dt = pd.Series(0.01, index=df.index, dtype="float64")

        # Rates (per second)
        ped = pd.to_numeric(df[ped_col], errors="coerce") if ped_col else pd.Series(0, index=df.index)
        thr = pd.to_numeric(df[thr_col], errors="coerce") if thr_col else pd.Series(0, index=df.index)
        rate_ped = ped.diff() / dt
        rate_thr = thr.diff() / dt

        intent_mask = (rate_ped >= args.thr_rate_pedal) | (rate_thr >= args.thr_rate_throttle)

        # "Passing" window mask (speed subset)
        if spd_col and spd_col in df.columns:
            pass_mask = (spd >= args.pass_win_min) & (spd <= args.pass_win_max)
        else:
            pass_mask = pd.Series(True, index=df.index)

        # Brake mask (avoid braking frames)
        if brk_col and brk_col in df.columns:
            brk = pd.to_numeric(df[brk_col], errors="coerce")
            brake_ok = (brk <= 0.5) | brk.isna()
        else:
            brake_ok = pd.Series(True, index=df.index)

        sel = intent_mask & pass_mask & brake_ok
        if not sel.any():
            continue

        # Group nearby hits (0.5s buckets) and pick within-bucket points
        ctime = dt.fillna(0).cumsum()
        grp = (ctime / 0.5).astype(int)
        pick = sel.groupby(grp).transform("any") & sel
        idxs = df.index[pick]

        # Gear / throttle snapshots
        gear = pd.to_numeric(df[gear_col], errors="coerce") if gear_col else pd.Series(np.nan, index=df.index)

        for i in idxs:
            g = int(gear.iloc[i]) if pd.notna(gear.iloc[i]) else None
            if g is None or g < 1 or g > 6: continue
            thr_i = thr.iloc[i] if pd.notna(thr.iloc[i]) else 0.0
            tps_bin_str = str(tps_bin(thr_i))

            if 1 <= g <= 5:
                up_df.loc[f"{g} -> {g+1} Shift", tps_bin_str] = float(args.delta_up)
            if 3 <= g <= 6:
                suffix = {1:"st",2:"nd",3:"rd"}.get(g,"th")
                rowr = f"{g}{suffix} Release"
                if rowr in tcc_rel_df.index:
                    tcc_rel_df.loc[rowr, tps_bin_str] = float(args.delta_tcc)

    # Write outputs
    out_up = out_dir / "INTENT__SHIFT_UP__DELTA.tsv"
    out_tcc = out_dir / "INTENT__TCC_RELEASE__DELTA.tsv"
    up_df.to_csv(out_up, sep="\t", index=False, float_format="%.1f")
    tcc_rel_df.to_csv(out_tcc, sep="\t", index=False, float_format="%.1f")

    # Nonzero counts (fixed)
    up_num  = up_df.iloc[:, 1:-1].apply(pd.to_numeric, errors="coerce").fillna(0)
    tcc_num = tcc_rel_df.iloc[:, 1:-1].apply(pd.to_numeric, errors="coerce").fillna(0)
    nz_up   = int((up_num != 0).to_numpy().sum())
    nz_tcc  = int((tcc_num != 0).to_numpy().sum())

    with open(out_dir / "INTENT__RUN_SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write(f"INTENT__SHIFT_UP__DELTA.tsv: nonzero {nz_up}/{len(UP_ROWS)*len(TPS_AXIS)}\n")
        f.write(f"INTENT__TCC_RELEASE__DELTA.tsv: nonzero {nz_tcc}/{len(TCC_REL_ROWS)*len(TPS_AXIS)}\n")
        f.write(f"Total rows: {total_rows}\n")

if __name__ == "__main__":
    main()
