import argparse, glob, os, pandas as pd, numpy as np

def pick_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    # case-insensitive fallback
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low: return low[c.lower()]
    return None

def load_clean(dirpath):
    pats = ["__trans_focus__clean_FULL__*withbrake*.csv", "*.csv"]
    files = []
    for pat in pats:
        files = sorted(glob.glob(os.path.join(dirpath, pat)))
        if files: break
    if not files:
        raise SystemExit(f"[ERR] no CSV files under {dirpath}")
    dfs = []
    for p in files:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print("[WARN] could not read", p, "->", e)
    if not dfs: raise SystemExit("[ERR] no readable CSVs")
    return pd.concat(dfs, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, help="dir of CLEAN_FULL CSVs")
    ap.add_argument("--out",   required=True, help="output TSV")
    ap.add_argument("--min-speed", type=float, default=35.0)
    ap.add_argument("--gear-min",  type=int,   default=3)
    ap.add_argument("--window-n",  type=int,   default=50, help="rolling window (samples) for TPS stability")
    ap.add_argument("--tps-std-max", type=float, default=2.5, help="max rolling std of throttle_pct")
    ap.add_argument("--on-rpm",    type=float, default=40.0, help="‘locked-ish’ slip threshold")
    ap.add_argument("--require-warm", action="store_true")
    ap.add_argument("--warm-f", type=float, default=100.0)
    args = ap.parse_args()

    df = load_clean(args.clean)

    # core signals
    speed = pick_col(df, ["speed_mph"])
    gear  = pick_col(df, ["gear_actual__canon","gear_actual","gear","Gear"])
    thr   = pick_col(df, ["throttle_pct","Throttle Position","Throttle Position (%)"])
    brake = pick_col(df, ["brake","Brake","Brake Applied"])

    if not all([speed, gear, thr]):
        miss = [("speed_mph",speed),("gear_actual",gear),("throttle_pct",thr)]
        raise SystemExit("[ERR] missing core columns: " + ", ".join([k for k,v in miss if v is None]))

    # slip: prefer fused; else derive
    slip = pick_col(df, ["tcc_slip_fused","TCC Slip","TCC_Slip","TCC Slip RPM"])
    used_fallback = False
    if slip is None:
        eng = pick_col(df, ["Engine RPM (SAE)","Engine RPM","RPM"])
        tur = pick_col(df, ["Trans Input Shaft RPM","Trans Input Shaft Speed","ISS","Turbine Speed","Trans Turbine Speed"])
        if not eng or not tur:
            raise SystemExit("[ERR] need tcc_slip_fused OR (Engine RPM and Trans Input Shaft RPM)")
        df["__slip"] = (pd.to_numeric(df[eng], errors="coerce") - pd.to_numeric(df[tur], errors="coerce")).abs()
        slip = "__slip"
        used_fallback = True

    # optional warm gates
    if args.require_warm:
        ect = pick_col(df, ["ECT__canon","Engine Coolant Temperature (F)","Engine Coolant Temp (F)","Engine Coolant Temperature","Coolant Temperature"])
        tft = pick_col(df, ["TFT__canon","Trans Fluid Temp (F)","Transmission Fluid Temperature (F)","Transmission Fluid Temperature"])
        if not ect or not tft:
            raise SystemExit("[ERR] require-warm set but could not find ECT/TFT columns")
        warm_mask = (pd.to_numeric(df[ect], errors="coerce") >= args.warm_f) & \
                    (pd.to_numeric(df[tft], errors="coerce") >= args.warm_f)
    else:
        warm_mask = pd.Series(True, index=df.index)

    # stable throttle via rolling std
    thrv = pd.to_numeric(df[thr], errors="coerce")
    thr_std = thrv.rolling(args.window_n, min_periods=args.window_n).std()
    stable = (thr_std <= args.tps_std_max)

    # brake gate if present (0 = not braking)
    if brake:
        br = pd.to_numeric(df[brake], errors="coerce").fillna(0)
        nobrake = (br == 0)
    else:
        nobrake = pd.Series(True, index=df.index)

    m = (pd.to_numeric(df[speed], errors="coerce") >= args.min_speed) & \
        (pd.to_numeric(df[gear], errors="coerce")  >= args.gear_min)  & \
        stable & nobrake & warm_mask

    good = df.loc[m, [gear, slip]].copy()
    good[gear] = pd.to_numeric(good[gear], errors="coerce")

    if good.empty:
        print("[LOCK_PROBE] 0 rows after relaxed gates — try lower --min-speed or higher --tps-std-max.")
        # still write an empty frame with headers
        out = pd.DataFrame(columns=["gear","samples","median_abs_slip","p90_abs_slip","pct_le_onrpm","lock_indicator"])
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out.to_csv(args.out, sep="\t", index=False)
        return

    g = good.groupby(gear, dropna=True)
    def pct_le_onrpm(s): 
        s = pd.to_numeric(s, errors="coerce").abs()
        return float((s <= args.on_rpm).mean()*100.0)

    out = pd.DataFrame({
        "gear": g.size().index.astype(int),
        "samples": g.size().values,
        "median_abs_slip": g[slip].apply(lambda s: pd.to_numeric(s, errors="coerce").abs().median()).values,
        "p90_abs_slip":    g[slip].apply(lambda s: pd.to_numeric(s, errors="coerce").abs().quantile(0.9)).values,
        "pct_le_onrpm":    g[slip].apply(pct_le_onrpm).values,
    })
    out["lock_indicator"] = np.where(out["pct_le_onrpm"]>=70, "likely-locked",
                              np.where(out["pct_le_onrpm"]>=40, "partial", "unlocked/unknown"))

    out_dir = os.path.dirname(args.out)
    if out_dir == "":
        out_dir = "."
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(args.out))
    out.to_csv(out_path, sep="\t", index=False)

    src = "fused tcc_slip_fused" if not used_fallback else "derived |Engine RPM - Input RPM|"
    print(f"[LOCK_PROBE RELAXED] wrote {out_path} using {src}. rows:{len(out)}")


if __name__ == "__main__":
    main()
