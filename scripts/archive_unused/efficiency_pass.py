#!/usr/bin/env python3
# efficiency_pass.py — Low-TPS efficiency-first placement using an RPM/TE proxy (no shift edges)
import pandas as pd, numpy as np, math, argparse
from common_utils import detect_cols, require, read_logs, TPS_AXIS, empty_shift_tables, empty_tcc_tables, to_tsv, nearest_tps_bin

GEAR = [4.027, 2.364, 1.532, 1.152, 0.852, 0.667]
FD = 3.08
ETA = 0.92
TIRE_DIAM_IN = 32.5
RADIUS_M = (TIRE_DIAM_IN*0.0254)/2.0

def engine_eff_proxy(rpm, te):
    # Lower is better. Proxy for fuel/work ~ rpm / te (bounded)
    if te<=0 or pd.isna(te) or pd.isna(rpm): return np.inf
    return float(rpm) / float(te)

def tractive_effort(tps, gear_idx):
    # normalized TE proxy at low TPS: TPS fraction × gearing × driveline
    overall = GEAR[gear_idx]*FD
    return (max(0.0, tps/100.0))*overall*ETA / RADIUS_M

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-glob", default=".\06_Logs\Trans_Review\__trans_focus__clean__*.csv")
    ap.add_argument("--out-prefix", default=".\EFF")
    args = ap.parse_args()

    print("[EFF] Starting...", flush=True)
    df, files = read_logs(args.logs_glob)
    c = detect_cols(df)
    require(c, ["speed","tps","gear","rpm"], df.columns)

    # Clean selection and copy to avoid chained assignment warnings
    d = df[[c["speed"],c["tps"],c["gear"],c["rpm"]]].copy()
    d = d.dropna().copy()
    d["speed_mph"]=pd.to_numeric(d[c["speed"]], errors="coerce").astype(float).clip(0, 140)
    d["tps"]      =pd.to_numeric(d[c["tps"]], errors="coerce").astype(float).clip(0, 100)
    d["gear_i"]   =pd.to_numeric(d[c["gear"]], errors="coerce").round().astype("Int64").fillna(1).astype(int).clip(1,6)
    d["rpm"]      =pd.to_numeric(d[c["rpm"]], errors="coerce").astype(float).clip(400,6500)

    # steady-state filter (very light accel by speed slope proxy)
    d["ds"] = d["speed_mph"].diff().abs().rolling(10, min_periods=1).mean()
    d = d[d["ds"]<0.3].copy()

    up_rows, dn_rows = empty_shift_tables()
    ap_rows, rl_rows = empty_tcc_tables()

    for (idx,(label_up,label_dn)) in enumerate(zip([r[0] for r in up_rows],[r[0] for r in dn_rows])):
        g_from = idx+1; g_to = idx+2
        up_vals=[]; dn_vals=[]
        # gear pair subset
        pair = d[d["gear_i"].isin([g_from,g_to])].copy()
        if pair.empty:
            up_rows[idx]=(label_up, [np.nan]*17); dn_rows[idx]=(label_dn, [np.nan]*17); continue

        for t in TPS_AXIS:
            if t>44:
                up_vals.append(np.nan); dn_vals.append(np.nan); continue
            z = pair[pair["tps"].between(max(0,t-4), min(100,t+4))].copy()
            if z.empty:
                up_vals.append(np.nan); dn_vals.append(np.nan); continue
            z = z.assign(
                tps_bin = z["tps"].apply(nearest_tps_bin),
                te      = z.apply(lambda r: tractive_effort(r["tps"], int(r["gear_i"])-1), axis=1),
            )
            z = z.assign(
                eff     = z.apply(lambda r: engine_eff_proxy(r["rpm"], r["te"]), axis=1),
                spd_bin = (z["speed_mph"]/0.5).round()*0.5
            )
            g = z.groupby(["spd_bin","gear_i"])["eff"].median().reset_index()
            spds = sorted(set(g["spd_bin"].tolist()))
            up_mph=np.nan
            for s in spds:
                e_from = float(g[(g["spd_bin"]==s)&(g["gear_i"]==g_from)]["eff"].median()) if not g[(g["spd_bin"]==s)&(g["gear_i"]==g_from)].empty else np.inf
                e_to   = float(g[(g["spd_bin"]==s)&(g["gear_i"]==g_to)]["eff"].median()) if not g[(g["spd_bin"]==s)&(g["gear_i"]==g_to)].empty else np.inf
                if e_to < e_from*0.995:
                    up_mph = float(s); break
            dn_mph = (max(0.0, up_mph - 2.2) if not pd.isna(up_mph) else np.nan)
            up_vals.append(up_mph if not pd.isna(up_mph) else np.nan)
            dn_vals.append(dn_mph if not pd.isna(dn_mph) else np.nan)
        up_rows[idx]=(label_up, up_vals)
        dn_rows[idx]=(label_dn, dn_vals)

    # Early TCC at low TPS in higher gears (seed; blender enforces gaps)
    for gi,label in [(2,"3rd Apply"),(3,"4th Apply"),(4,"5th Apply"),(5,"6th Apply")]:
        vals=[ ( {2:28.0,3:40.0,4:48.0,5:56.0}[gi] if t<=44 else np.nan ) for t in TPS_AXIS ]
        ap_rows[gi]= (label, vals)
        rl_rows[gi]= (label.replace("Apply","Release"), [ (v+1.3 if not pd.isna(v) else np.nan) for v in vals ])

    to_tsv(up_rows, f"{args.out_prefix}__SHIFT_UP__Throttle17.tsv")
    to_tsv(dn_rows, f"{args.out_prefix}__SHIFT_DOWN__Throttle17.tsv")
    to_tsv(ap_rows, f"{args.out_prefix}__TCC_APPLY__Throttle17.tsv")
    to_tsv(rl_rows, f"{args.out_prefix}__TCC_RELEASE__Throttle17.tsv")
    print(f"[EFF] Wrote outputs with prefix: {args.out_prefix}", flush=True)

if __name__=="__main__":
    main()
