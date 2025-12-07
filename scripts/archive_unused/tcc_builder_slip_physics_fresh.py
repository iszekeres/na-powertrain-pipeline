#!/usr/bin/env python3
# tcc_builder_slip_physics_fresh.py
# Neutral TCC Apply/Release from slip physics, NOT from logged TCC commands or existing tables.
# Uses unlocked-only slip to avoid "replicating" OEM lock timing. Predicts where slip would decay
# to a low threshold if it remained unlocked.
#
# Inputs: speed_mph__canon, time_s__canon, throttle_pct__canon, gear_actual__canon, brake__canon,
#         engine_rpm__canon, turbine_rpm__canon (for slip), tftF__canon optional.
# Output tables: TCC_APPLY__Throttle17.tsv, TCC_RELEASE__Throttle17.tsv (17-pt TPS axis, 0.1 mph)

import os, numpy as np, pandas as pd

TPS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
HEADER = "mph\t" + "\t".join(map(str,TPS)) + "\t%"

SLIP_ON  = 40   # rpm — where locking becomes reasonable if still unlocked
SLIP_OFF = 100  # rpm — unlock region when slowing (hysteresis reference)
REL_GAP  = 1.5  # mph — ensure Release >= Apply + gap (project policy)

GEARS = [3,4,5,6]  # neutral seed: 3rd–6th; we can extend if data supports it

def prep(df):
    d = df.copy()
    need = ["speed_mph__canon","time_s__canon","throttle_pct__canon","gear_actual__canon","brake__canon"]
    for c in need:
        if c not in d.columns: d[c]=np.nan
    d = d[d[need].notna().all(axis=1)]
    if "tftF__canon" in d.columns:
        d = d[(d["tftF__canon"]>=100) | (d["tftF__canon"].isna())]
    d = d[(d["brake__canon"]==0) & d["speed_mph__canon"].between(5,120) & d["throttle_pct__canon"].between(0,100)]
    # slip from canon rpms
    if "engine_rpm__canon" in d.columns and "turbine_rpm__canon" in d.columns:
        slip = pd.to_numeric(d["engine_rpm__canon"], errors="coerce") - pd.to_numeric(d["turbine_rpm__canon"], errors="coerce")
    else:
        slip = pd.Series(np.nan, index=d.index)
    d["slip_abs"] = slip.abs()
    # treat "locked-like" windows as OEM lock; exclude those samples from modeling
    # locked-like ≈ slip <= 30 rpm (sustained). We approximate by a short rolling min.
    smin = d["slip_abs"].rolling(6, center=True, min_periods=1).min()
    d["unlocked_mask"] = (smin > 30)  # keep only samples likely UNLOCKED
    return d

def tbin(x):
    arr=np.array(TPS); return int(arr[np.abs(arr-x).argmin()])

def predict_apply_release(d):
    # For each gear & TPS bin: use UNLOCKED samples only to estimate where slip would reach SLIP_ON as speed rises.
    # Implementation: compute median unlocked slip vs mph bin; find first mph where median<=SLIP_ON; if none, leave blank.
    bins_speed = np.arange(0,121,1)
    rowsA, rowsR = [], []
    for g in GEARS:
        nameA = f"{g}rd Apply" if g==3 else (f"{g}th Apply")
        nameR = f"{g}rd Release" if g==3 else (f"{g}th Release")
        ups, rels = [], []
        dg = d[(d["gear_actual__canon"].round()==g)]
        if dg.empty:
            ups = ["" for _ in TPS]; rels=["" for _ in TPS]
            rowsA.append((nameA, ups)); rowsR.append((nameR, rels)); continue
        dg = dg.copy()
        dg["tps_bin"] = dg["throttle_pct__canon"].map(tbin)
        dg["mph_bin"] = dg["speed_mph__canon"].round().astype(int).clip(0,120)

        for tps in TPS:
            dt = dg[(dg["tps_bin"]==tps) & (dg["unlocked_mask"])]
            if dt.empty:
                ups.append(""); rels.append(""); continue
            med = dt.groupby("mph_bin")["slip_abs"].median()
            s = med.reindex(bins_speed).interpolate("linear", limit_area="inside").rolling(3,center=True,min_periods=1).median()

            # APPLY: first mph where predicted unlocked slip <= SLIP_ON
            idx = np.where((s<=SLIP_ON) & s.notna())[0]
            if len(idx):
                ap = float(bins_speed[idx[0]])
                ap = round(ap,1)
                # RELEASE: simple hysteresis above apply
                rel = round(ap + REL_GAP, 1)
                ups.append(ap); rels.append(rel)
            else:
                ups.append(""); rels.append("")

        rowsA.append((nameA, ups))
        rowsR.append((nameR, rels))

    # 1st/2nd rows: neutral seed leaves blank (or sentinel 318 if you prefer lockout later via overlay)
    # We’ll write them as 318 to be paste-friendly and explicit.
    for prefix in [("1st Apply","1st Release"), ("2nd Apply","2nd Release")]:
        rowsA.insert(0,(prefix[0], [318 for _ in TPS]))
        rowsR.insert(0,(prefix[1], [318 for _ in TPS]))

    # enforce monotone TPS within each row and 0.1 formatting; Release ≥ Apply + REL_GAP where both present
    def monotone_tps(rows):
        fixed=[]
        for nm,vals in rows:
            s = pd.Series([v if isinstance(v,(int,float)) else np.nan for v in vals])
            for i in range(1,len(s)):
                if not np.isnan(s[i-1]):
                    if np.isnan(s[i]) or s[i] < s[i-1]: s[i]=s[i-1]
            fixed.append((nm, [("" if np.isnan(x) else round(float(x),1)) for x in s.tolist()]))
        return fixed
    rowsA = monotone_tps(rowsA); rowsR = monotone_tps(rowsR)

    # enforce gap
    # map apply rows by gear name to line up with release
    amap = {nm.replace(" Apply",""):vals for nm,vals in rowsA}
    rfix=[]
    for nm,rv in rowsR:
        base = nm.replace(" Release","")
        av   = amap.get(base, ["" for _ in TPS])
        for j in range(len(TPS)):
            if isinstance(rv[j],(int,float)) and isinstance(av[j],(int,float)) and rv[j] < av[j] + REL_GAP:
                rv[j] = round(av[j] + REL_GAP, 1)
        rfix.append((nm,rv))
    rowsR = rfix

    return rowsA, rowsR

def write_tsv(out_dir, rowsA, rowsR):
    os.makedirs(out_dir, exist_ok=True)
    ap = os.path.join(out_dir,"TCC_APPLY__Throttle17.tsv")
    rp = os.path.join(out_dir,"TCC_RELEASE__Throttle17.tsv")
    def w(path, rows):
        with open(path,"w",encoding="utf-8") as f:
            f.write(HEADER+"\n")
            for nm,vals in rows: f.write(nm+"\t"+"\t".join(map(str,vals))+"\t\n")
    w(ap, rowsA); w(rp, rowsR)
    # label guard (no '3th')
    for p in [ap,rp]:
        txt=open(p,"r",encoding="utf-8").read().replace("3th","3rd")
        open(p,"w",encoding="utf-8").write(txt)
    print("[OK] wrote", ap); print("[OK] wrote", rp)

def main(clean, out_dir):
    df = pd.read_csv(clean, low_memory=False)
    d  = prep(df)
    A,R = predict_apply_release(d)
    write_tsv(out_dir, A, R)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--out-dir", required=True)
    a=ap.parse_args()
    main(a.clean, a.out_dir)
