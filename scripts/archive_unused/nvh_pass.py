#!/usr/bin/env python3
# nvh_pass.py — neutral NVH analysis (report by default; optional 'apply' deltas)
# Usage:
#   python .\nvh_pass.py --logs-glob ".\06_Logs\Trans_Review\__trans_focus__clean__*.csv" --out-prefix ".\NVH" [--mode report|apply]
import argparse, glob, os, sys, math, csv
import pandas as pd

TAG = "NVH"

def nearest_bin(v, step=0.5):
    try: return round(float(v)/step)*step
    except: return None

def safe_num(s):
    try: return float(s)
    except: return float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-glob", default=r".\06_Logs\Trans_Review\__trans_focus__clean__*.csv")
    ap.add_argument("--out-prefix", default=r".\NVH")
    ap.add_argument("--mode", choices=["report","apply"], default="report")
    args = ap.parse_args()

    print(f"[{TAG}] Starting...", flush=True)
    paths = glob.glob(args.logs_glob)
    if not paths:
        print(f"[{TAG}] No files matched: {args.logs_glob}", flush=True); sys.exit(0)
    print(f"[{TAG}] Found {len(paths)} file(s)", flush=True)

    # Minimal columns
    need = ["time_s","speed_mph","throttle_pct","gear_actual","engine_rpm","tcc_slip_fused","tcc_locked_built"]
    frames=[]
    for i,p in enumerate(sorted(paths), start=1):
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            print(f"[{TAG}] WARN: cannot read {p}: {e}", flush=True); continue
        for c in need:
            if c not in df.columns: df[c] = pd.NA
        frames.append(df[need].copy())
        if i%4==0 or i==len(paths):
            print(f"[{TAG}] ({i}/{len(paths)})", flush=True)
    if not frames:
        print(f"[{TAG}] No usable data", flush=True); sys.exit(0)
    d = pd.concat(frames, ignore_index=True)

    # Simple NVH proxies
    d["spd"] = pd.to_numeric(d["speed_mph"], errors="coerce")
    d["tps"] = pd.to_numeric(d["throttle_pct"], errors="coerce")
    d["rpm"] = pd.to_numeric(d["engine_rpm"], errors="coerce")
    d["gear"]= pd.to_numeric(d["gear_actual"], errors="coerce")
    d["slip"]= pd.to_numeric(d["tcc_slip_fused"], errors="coerce")
    d["lock"]= pd.to_numeric(d["tcc_locked_built"], errors="coerce")

    d = d.dropna(subset=["spd","tps","gear"])
    d["spd_bin"]= (d["spd"]/0.5).round()*0.5
    d["tps_bin"]= d["tps"].round()

    # Roughness: speed derivative variance during 10%± TPS windows, per (gear,tps_bin,spd_bin)
    d["ds"] = d["spd"].diff().abs().clip(0,5)
    grp = d.groupby(["gear","tps_bin","spd_bin"])
    rough = grp["ds"].median().reset_index(name="roughness")
    heat  = grp["slip"].apply(lambda s: s.fillna(0).abs().median()).reset_index(name="slip_med")

    # Heuristic flags
    rough["nvh_flag"] = (rough["roughness"]>0.25).astype(int)  # tune threshold later
    heat["heat_flag"] = (heat["slip_med"]>120).astype(int)

    out_prefix = args.out_prefix.rstrip("\\/")
    os.makedirs(os.path.dirname(out_prefix) if os.path.dirname(out_prefix) else ".", exist_ok=True)

    # Write reports
    rough.to_csv(f"{out_prefix}__roughness.csv", index=False)
    heat.to_csv(f"{out_prefix}__slip_heat.csv", index=False)
    print(f"[{TAG}] WROTE {out_prefix}__roughness.csv {out_prefix}__slip_heat.csv", flush=True)

    if args.mode=="apply":
        # Produce neutral 'nudges' TSVs (mostly NaNs; small deltas where flagged)
        # These are advisory; overlay_polish_v3.py will still enforce constraints after.
        import numpy as np

        TPS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
        def empty_shift(kind):
            labels = (["1 -> 2 Shift","2 -> 3 Shift","3 -> 4 Shift","4 -> 5 Shift","5 -> 6 Shift"]
                      if kind=="up" else
                      ["2 -> 1 Shift","3 -> 2 Shift","4 -> 3 Shift","5 -> 4 Shift","6 -> 5 Shift"])
            return {lab:[np.nan]*17 for lab in labels}

        up = empty_shift("up"); dn = empty_shift("down")
        # If NVH flagged at (gear,g,tps_bin,spd_bin), gently lower UP by 0.3 mph and raise DOWN gap by 0.2 at nearby TPS bins
        for _,row in rough.iterrows():
            if int(row["nvh_flag"])!=1: continue
            g = int(row["gear"]); tps = int(row["tps_bin"])
            if g<1 or g>5: continue
            try:
                i = min(range(17), key=lambda k: abs(TPS[k]-tps))
            except:
                continue
            # nudge
            up[f"{g} -> {g+1} Shift"][i] = -0.3
            dn[f"{g+1} -> {g} Shift"][i] = +0.2

        def write_delta(path, body):
            with open(path,"w",encoding="utf-8",newline="") as f:
                w=csv.writer(f, delimiter="\t")
                w.writerow(["mph"]+[str(x) for x in TPS]+["%"])
                for lab in body:
                    w.writerow([lab]+[("" if (v is None or (isinstance(v,float) and math.isnan(v))) else f"{v:.1f}") for v in body[lab]]+[""])

        write_delta(f"{out_prefix}__SHIFT_UP__DELTA.tsv", up)
        write_delta(f"{out_prefix}__SHIFT_DOWN__DELTA.tsv", dn)
        print(f"[{TAG}] WROTE {out_prefix}__SHIFT_UP__DELTA.tsv {out_prefix}__SHIFT_DOWN__DELTA.tsv", flush=True)

if __name__=="__main__":
    main()
