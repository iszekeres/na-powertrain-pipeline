#!/usr/bin/env python3
# corner_exit_pass_weighted__chassis_shim.py
# Generate CORNER__SHIFT_DOWN__DELTA.tsv using throttle pickup + optional chassis cues.
#
# If no chassis channels are present (steer/latg/yaw), chassis gating is BYPASSED by default.
# Use --require-chassis to force chassis cues; otherwise gating falls back to True when none exist.
#
# Usage example:
#   python corner_exit_pass_weighted__chassis_shim.py \
#     --logs-glob ".\06_Logs\Trans_Review\__trans_focus__clean__*withtime*.csv" \
#     --out "C:\tuning\logs\06_Logs\Trans_Review\CORNER__SHIFT_DOWN__DELTA.tsv" \
#     --min-speed 8 --max-speed 30 --thr-rate 18 \
#     --lat-g 0.08 --yaw-rate 12 --steer-abs 45 --steer-rate 30 \
#     --min-score 60 --delta-mph 0.3 \
#     --steer-column "Steering Wheel Position" --latg-column "lateral acceleration" --yaw-column "yaw rate" \
#     --latg-units g --yaw-units deg
#
import os, glob, argparse, numpy as np, pandas as pd
from tps_phases import is_corner_exit_band

TPS_AXIS17 = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
DOWN_ROWS = ["2 -> 1 Shift","3 -> 2 Shift","4 -> 3 Shift","5 -> 4 Shift","6 -> 5 Shift"]
HEADER = ["mph"] + [str(x) for x in TPS_AXIS17] + ["%"]

ALIASES = {
  "steer": [
      "steering_angle_deg","Steering Angle","SAS_Angle","Steering Wheel Angle",
      "Steer_Angle","steer_angle_deg","SteerAngle","Steering Wheel Position"
  ],
  "latg":  [
      "lat_g","LatAcc","lateral_g","Lateral G","Lat Accel (g)","lat_accel_g","lateral acceleration"
  ],
  "yaw":   [
      "yaw_rate_deg_s","YawRate","yaw_dps","Yaw Rate","Yaw Rate (deg/s)","yaw rate"
  ]
}

def ensure_corner_canon(df, src_name):
    """
    Ensure CORNER has gear_actual, speed_mph, throttle_pct columns.
    If the raw names are missing or all-NaN, alias them from canonical fields.
    """
    alias_map = {
        "gear_actual": [
            "gear_actual__canon",
            "Gear Actual",
            "Trans Current Gear",
        ],
        "speed_mph": [
            "speed_mph__canon",
            "Vehicle Speed (SAE)",
            "Speed MPH",
        ],
        # TPS: require canonical throttle_pct__canon (no fallback to raw TPS)
        "throttle_pct": [
            "throttle_pct__canon",
        ],
    }

    for target, candidates in alias_map.items():
        need_alias = (target not in df.columns) or df[target].isna().all()
        if need_alias:
            for cand in candidates:
                if cand in df.columns and not df[cand].isna().all():
                    df[target] = df[cand]
                    break

    key_cols = ["gear_actual","speed_mph","throttle_pct"]
    missing = [c for c in key_cols if c not in df.columns or df[c].isna().all()]
    if missing:
        raise RuntimeError(f"{src_name}: missing required column(s) for CORNER after alias: {missing}")

    df = df.dropna(subset=key_cols, how="any").copy()
    return df

def find_col(cols, keys):
    for k in keys:
        if k in cols: return k
    lower = {c.lower(): c for c in cols}
    for k in keys:
        kl = k.lower()
        if kl in lower:
            return lower[kl]
    # fuzzy contains
    for c in cols:
        cl = c.lower()
        for k in keys:
            if k.lower() in cl:
                return c
    return None

def tps_bin_value(x):
    if pd.isna(x): return TPS_AXIS17[0]
    idx = np.searchsorted(TPS_AXIS17, x, side='right') - 1
    return TPS_AXIS17[max(0, min(idx, len(TPS_AXIS17)-1))]

def main():
    ap = argparse.ArgumentParser(description="Corner-exit (with optional chassis cues) shim.")
    ap.add_argument("--logs-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-speed", type=float, default=8.0)
    ap.add_argument("--max-speed", type=float, default=30.0)
    ap.add_argument("--thr-rate", type=float, default=18.0, help="min d(throttle)/dt in %%/s")
    ap.add_argument("--lat-g", type=float, default=0.08, dest="lat_g", help="threshold in 'g' units unless --latg-units mps2")
    ap.add_argument("--yaw-rate", type=float, default=12.0, dest="yaw_rate", help="threshold in deg/s unless --yaw-units rad")
    ap.add_argument("--steer-abs", type=float, default=45.0, dest="steer_abs")
    ap.add_argument("--steer-rate", type=float, default=30.0, dest="steer_rate")
    ap.add_argument("--min-score", type=float, default=80.0, help="hits * weights must exceed this to emit a delta")
    ap.add_argument("--delta-mph", type=float, default=0.3)

    ap.add_argument("--steer-column", dest="steer_col", default=None, help="Explicit steering angle column name")
    ap.add_argument("--latg-column", dest="latg_col", default=None, help="Explicit lateral accel column name")
    ap.add_argument("--yaw-column", dest="yaw_col", default=None, help="Explicit yaw-rate column name")

    ap.add_argument("--latg-units", choices=["g","mps2"], default="g", help="units for lateral accel column (default g)")
    ap.add_argument("--yaw-units", choices=["deg","rad"], default="deg", help="units for yaw-rate column (default deg/s)")
    ap.add_argument("--require-chassis", action="store_true", help="if set, require at least one chassis cue to be present; else bypass gating when none present")
    args = ap.parse_args()

    files = sorted(glob.glob(args.logs_glob))
    if not files:
        raise SystemExit("No files matched: " + args.logs_glob)

    scores = {(row, tps): 0.0 for row in DOWN_ROWS for tps in TPS_AXIS17}

    for fp in files:
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception:
            df = pd.read_csv(fp, low_memory=False, engine="python", on_bad_lines="skip")

        try:
            df = ensure_corner_canon(df, os.path.basename(fp))
        except RuntimeError as exc:
            print(f"[WARN] {exc}; skipping")
            continue

        t   = pd.to_numeric(df["time_s"], errors="coerce")
        thr = pd.to_numeric(df["throttle_pct"], errors="coerce").clip(0,100)
        v   = pd.to_numeric(df["speed_mph"], errors="coerce")
        g   = pd.to_numeric(df["gear_actual"], errors="coerce").round().astype("Int64")

        # explicit names or aliases
        steer_col = args.steer_col if args.steer_col in df.columns else find_col(df.columns, ALIASES["steer"])
        latg_col  = args.latg_col  if args.latg_col  in df.columns else find_col(df.columns, ALIASES["latg"])
        yaw_col   = args.yaw_col   if args.yaw_col   in df.columns else find_col(df.columns, ALIASES["yaw"])

        have_chassis = any([steer_col, latg_col, yaw_col])

        steer = pd.to_numeric(df[steer_col], errors="coerce") if steer_col else pd.Series(np.nan, index=df.index)
        latg  = pd.to_numeric(df[latg_col], errors="coerce")  if latg_col  else pd.Series(np.nan, index=df.index)
        yaw   = pd.to_numeric(df[yaw_col], errors="coerce")   if yaw_col   else pd.Series(np.nan, index=df.index)

        # Unit normalization
        # convert lateral accel column to 'g' if given in m/s^2
        if latg_col and args.latg_units == "mps2":
            latg = latg / 9.80665
        # convert yaw to deg/s if given in rad/s
        if yaw_col and args.yaw_units == "rad":
            yaw = yaw * (180.0 / np.pi)

        dt = t.diff().replace(0, np.nan)
        dthr = (thr.diff() / dt).replace([np.inf,-np.inf], np.nan)
        dsteer = (steer.diff() / dt).replace([np.inf,-np.inf], np.nan)

        window = v.between(args.min_speed, args.max_speed) & g.between(2,6)
        pickup = dthr >= args.thr_rate

        # TPS phase gating: require TPS in corner-exit band (>= ~12%)
        tps_ok = thr.apply(is_corner_exit_band)

        if have_chassis:
            chassis = (
                (latg.abs() >= args.lat_g) |
                (yaw.abs()  >= args.yaw_rate) |
                (steer.abs()>= args.steer_abs) |
                (dsteer.abs()>= args.steer_rate)
            )
        else:
            # BYPASS gating if no chassis channels present (unless require-chassis)
            chassis = pd.Series(True, index=df.index) if not args.require_chassis else pd.Series(False, index=df.index)

        hit = window & pickup & tps_ok & chassis

        if not hit.any():
            continue

        # weighting from cues (robust caps); treat missing as 0 contribution
        latn = (latg.abs().clip(0, 0.5) / 0.5).fillna(0.0)    # 0..1 at 0.5 g
        yawn = (yaw.abs().clip(0, 60.0) / 60.0).fillna(0.0)   # 0..1 at 60 deg/s
        steern = (steer.abs().clip(0, 180.0) / 180.0).fillna(0.0)  # 0..1 at 180 deg

        strength = (1.0 + 0.3*latn + 0.2*yawn + 0.2*steern)

        sel = pd.DataFrame({
            "gear": g.where(hit),
            "tpsbin": thr.where(hit).apply(tps_bin_value),
            "strength": strength.where(hit)
        }).dropna()

        sel["row"] = sel["gear"].map({2:"2 -> 1 Shift",3:"3 -> 2 Shift",4:"4 -> 3 Shift",5:"5 -> 4 Shift",6:"6 -> 5 Shift"})
        grp = sel.groupby(["row","tpsbin"])["strength"].sum()
        for (row, tpsv), s in grp.items():
            scores[(row, int(tpsv))] += float(s)

    # Build delta matrix
    data = []
    for row in DOWN_ROWS:
        vals = []
        for tpsv in TPS_AXIS17:
            s = scores[(row, tpsv)]
            vals.append(args.delta_mph if s >= args.min_score else 0.0)
        data.append([row] + vals + [""])

    out_df = pd.DataFrame(data, columns=HEADER)
    # 0.1 mph formatting
    for c in out_df.columns[1:-1]:
        out_df[c] = out_df[c].map(lambda x: f"{float(x):.1f}" if isinstance(x, (int,float)) else x)
    out_df.to_csv(args.out, sep="\t", index=False)
    print("[CORNER_CHASSIS_SHIM] wrote", args.out)

if __name__ == "__main__":
    main()
