
#!/usr/bin/env python3
# nvh_pass_weighted.py — NVH pass with recency & route weighting; emits DELTAs if --mode apply
import argparse, glob, os, sys, math, csv
import pandas as pd
from weight_utils import combined_weight

TAG = "NVH_W"

TPS=[0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-glob", default=r".\06_Logs\Trans_Review\__trans_focus__clean__*.csv")
    ap.add_argument("--out-prefix", default=r".\NVH")
    ap.add_argument("--mode", choices=["report","apply"], default="report")
    ap.add_argument("--half-life-days", type=float, default=30.0)
    ap.add_argument("--route-bias", default="neighborhood=1.5,inbound=1.2,outbound=1.2,highway=1.1")
    args = ap.parse_args()

    route_map = dict(kv.split("=") for kv in args.route_bias.split(",") if "=" in kv)

    need = ["time_s","speed_mph","throttle_pct","gear_actual","tcc_slip_fused","__file"]
    paths = glob.glob(args.logs_glob)
    frames=[]
    for p in sorted(paths):
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            continue
        for c in need:
            if c not in df.columns: df[c]=pd.NA
        df["__file"] = df["__file"].fillna(os.path.basename(p))
        frames.append(df[need].copy())
    if not frames:
        print(f"[{TAG}] No data"); return
    d = pd.concat(frames, ignore_index=True)
    d["spd"]=pd.to_numeric(d["speed_mph"], errors="coerce")
    d["tps"]=pd.to_numeric(d["throttle_pct"], errors="coerce")
    d["gear"]=pd.to_numeric(d["gear_actual"], errors="coerce")
    d["slip"]=pd.to_numeric(d["tcc_slip_fused"], errors="coerce")
    d=d.dropna(subset=["spd","tps","gear"])

    d["ds"] = d["spd"].diff().abs().clip(0,5)
    d["w"]  = [combined_weight(fn, spd, args.half_life_days, route_map) for fn,spd in zip(d["__file"], d["spd"])]
    # weighted median-ish via expanding weights: use groupby mean as proxy (simple, fast)
    grp = d.groupby(["gear",d["tps"].round(),"spd"].apply(lambda s:(s/0.5).round()*0.5))
    rough = d.groupby(["gear", d["tps"].round(), (d["spd"]/0.5).round()*0.5]).apply(lambda g: (g["ds"]*g["w"]).mean()).reset_index(name="roughness_w")

    out_prefix = args.out_prefix.rstrip("\\/")
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    rough.to_csv(f"{out_prefix}__roughness_weighted.csv", index=False)
    print(f"[{TAG}] WROTE {out_prefix}__roughness_weighted.csv")

    if args.mode=="apply":
        import numpy as np
        def empty(kind):
            labs=(["1 -> 2 Shift","2 -> 3 Shift","3 -> 4 Shift","4 -> 5 Shift","5 -> 6 Shift"] if kind=="up" else
                  ["2 -> 1 Shift","3 -> 2 Shift","4 -> 3 Shift","5 -> 4 Shift","6 -> 5 Shift"])
            return {lab:[np.nan]*17 for lab in labs}
        up=empty("up"); dn=empty("down")
        for _,row in rough.iterrows():
            gear=int(row["gear"]); t=int(row["tps"]); r=float(row["roughness_w"])
            if gear<1 or gear>5: continue
            i=min(range(17), key=lambda k: abs(TPS[k]-t))
            if r>0.25:
                up[f"{gear} -> {gear+1} Shift"][i] = -0.3 if math.isnan(up[f"{gear} -> {gear+1} Shift"][i]) else (up[f"{gear} -> {gear+1} Shift"][i]-0.3)
                dn[f"{gear+1} -> {gear} Shift"][i] = +0.2 if math.isnan(dn[f"{gear+1} -> {gear} Shift"][i]) else (dn[f"{gear+1} -> {gear} Shift"][i]+0.2)
        # write deltas
        import csv, math
        with open(f"{out_prefix}__SHIFT_UP__DELTA.tsv","w",encoding="utf-8",newline="") as f:
            w=csv.writer(f, delimiter="\t"); w.writerow(["mph"]+[str(x) for x in TPS]+["%"])
            for lab in up:
                w.writerow([lab]+[("" if (v is None or (isinstance(v,float) and math.isnan(v))) else f"{v:.1f}") for v in up[lab]]+[""])
        with open(f"{out_prefix}__SHIFT_DOWN__DELTA.tsv","w",encoding="utf-8",newline="") as f:
            w=csv.writer(f, delimiter="\t"); w.writerow(["mph"]+[str(x) for x in TPS]+["%"])
            for lab in dn:
                w.writerow([lab]+[("" if (v is None or (isinstance(v,float) and math.isnan(v))) else f"{v:.1f}") for v in dn[lab]]+[""])
        print(f"[{TAG}] WROTE {out_prefix}__SHIFT_UP__DELTA.tsv {out_prefix}__SHIFT_DOWN__DELTA.tsv")
