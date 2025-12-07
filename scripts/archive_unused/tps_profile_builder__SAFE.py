import os, glob, argparse, numpy as np, pandas as pd, math, sys
MAD_K = 1.4826  # robust sigma ≈ MAD * 1.4826

ap = argparse.ArgumentParser()
ap.add_argument("--clean", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--bins-mph", type=float, default=5.0)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(a.clean, "*.csv")))
rows = []
need = {"speed_mph","throttle_pct","gear_actual"}

# Canonical 5 mph edges from 0 to 120; centers are 2.5, 7.5, ... 117.5
edges = np.arange(0.0, 120.0 + a.bins_mph, a.bins_mph)

for f in files:
    df = pd.read_csv(f, low_memory=False)
    if not need.issubset(df.columns): 
        continue
    d = df[list(need)].copy()
    d["speed_mph"]    = pd.to_numeric(d["speed_mph"],    errors="coerce")
    d["throttle_pct"] = pd.to_numeric(d["throttle_pct"], errors="coerce")
    d["gear_actual"]  = pd.to_numeric(d["gear_actual"],  errors="coerce")
    d.dropna(subset=["speed_mph","throttle_pct","gear_actual"], inplace=True)
    if d.empty:
        continue

    for g in (1,2,3,4,5,6):
        gdf = d.loc[d["gear_actual"]==g]
        if gdf.empty: 
            continue
        v   = gdf["speed_mph"].to_numpy(dtype=float)
        thr = gdf["throttle_pct"].to_numpy(dtype=float)
        idx = np.digitize(v, edges) - 1  # 0..len(edges)-2

        for i in range(len(edges)-1):
            sel = (idx == i)
            if not np.any(sel): 
                continue
            m = thr[sel]
            med = float(np.nanmedian(m))
            mad = float(np.nanmedian(np.abs(m - med))) if m.size>0 else float("nan")
            sig = MAD_K*mad
            if (sig==0.0 or np.isnan(sig)) and m.size>1:
                # std fallback if MAD is degenerate
                sig = float(np.nanstd(m, ddof=1))
            rows.append({
                "gear": g,
                "mph":  (edges[i] + edges[i+1]) / 2.0,
                "tps_base":  med,
                "tps_sigma": sig,
                "tps_n":     int(m.size)
            })

out_cols = ["gear","mph","tps_base","tps_sigma","tps_n"]
if not rows:
    print("[ERR] TPS builder produced 0 rows (check core columns & values).", file=sys.stderr)
    # Write an empty, correctly-headed file so downstream error messages are clearer:
    pd.DataFrame(columns=out_cols).to_csv(os.path.join(a.out_dir,"TPS_PROFILE__pergear_cruise.tsv"), sep="\t", index=False)
    sys.exit(2)

out = pd.DataFrame(rows, columns=out_cols).sort_values(["gear","mph"])
p   = os.path.join(a.out_dir, "TPS_PROFILE__pergear_cruise.tsv")
out.to_csv(p, sep="\t", index=False)
print("[TPS_SAFE] wrote", p, "rows=", len(out), "cols=", list(out.columns))
