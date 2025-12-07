#!/usr/bin/env python3
import os, numpy as np, pandas as pd, sys

TPS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
HEADER = "mph\t" + "\t".join(map(str,TPS)) + "\t%"
GEARS = [3,4,5,6]    # neutral seed
SLIP_ON  = 40.0      # rpm threshold for hypothetical lock
REL_GAP  = 1.5       # mph min hysteresis (Release ≥ Apply + gap)

def prep(df):
    d = df.copy()
    need = ["speed_mph__canon","time_s__canon","throttle_pct__canon","gear_actual__canon","brake__canon","engine_rpm__canon","turbine_rpm__canon"]
    for c in need:
        if c not in d.columns: d[c]=np.nan
    d = d[d[need].notna().all(axis=1)]
    if "tftF__canon" in d.columns:
        d = d[(d["tftF__canon"]>=100) | (d["tftF__canon"].isna())]
    d = d[(d["brake__canon"]==0) & d["speed_mph__canon"].between(5,120) & d["throttle_pct__canon"].between(0,100)]
    d["slip_abs"] = (pd.to_numeric(d["engine_rpm__canon"],errors="coerce") - pd.to_numeric(d["turbine_rpm__canon"],errors="coerce")).abs()
    # keep "unlocked" samples only (approx) to avoid copying OEM lock timing
    d["unlocked_mask"] = d["slip_abs"].rolling(6,center=True,min_periods=1).min() > 30.0
    return d

def tbin(x):
    arr=np.array(TPS); return int(arr[np.abs(arr-x).argmin()])

def predict_tables(d):
    bins_speed = np.arange(0,121,1)
    rowsA, rowsR = [], []
    for g in GEARS:
        nameA = f"{g}rd Apply" if g==3 else f"{g}th Apply"
        nameR = f"{g}rd Release" if g==3 else f"{g}th Release"
        ups, rels = [], []
        dg = d[(d["gear_actual__canon"].round()==g)]
        if dg.empty:
            ups = ["" for _ in TPS]; rels = ["" for _ in TPS]
            rowsA.append((nameA, ups)); rowsR.append((nameR, rels)); continue
        dg = dg.copy()
        dg["tps_bin"] = dg["throttle_pct__canon"].map(tbin)
        dg["mph_bin"] = dg["speed_mph__canon"].round().astype(int).clip(0,120)
        for tps in TPS:
            dt = dg[(dg["tps_bin"]==tps) & (dg["unlocked_mask"])]
            if dt.empty: ups.append(""); rels.append(""); continue
            med = dt.groupby("mph_bin")["slip_abs"].median()
            s = med.reindex(bins_speed).interpolate("linear", limit_area="inside").rolling(3,center=True,min_periods=1).median()
            idx = np.where((s<=SLIP_ON) & s.notna())[0]
            if len(idx):
                ap = float(bins_speed[idx[0]])
                ap = round(ap,1)
                rel = round(ap + REL_GAP, 1)
                ups.append(ap); rels.append(rel)
            else:
                ups.append(""); rels.append("")
        rowsA.append((nameA, ups))
        rowsR.append((nameR, rels))

    # 1st/2nd lockout sentinels (project paste-friendly form)
    rowsA.insert(0, ("2nd Apply", [318 for _ in TPS]))
    rowsA.insert(0, ("1st Apply", [318 for _ in TPS]))
    rowsR.insert(0, ("2nd Release", [318 for _ in TPS]))
    rowsR.insert(0, ("1st Release", [318 for _ in TPS]))

    # monotone TPS & release gap
    def mono(rows):
        out=[]
        for nm,vals in rows:
            s = pd.Series([v if isinstance(v,(int,float)) else np.nan for v in vals])
            for i in range(1,len(s)):
                if not np.isnan(s[i-1]):
                    if np.isnan(s[i]) or s[i] < s[i-1]: s[i]=s[i-1]
            out.append((nm,[("" if np.isnan(x) else round(float(x),1)) for x in s.tolist()]))
        return out
    rowsA = mono(rowsA); rowsR = mono(rowsR)

    amap = {nm.replace(" Apply",""):vals for nm,vals in rowsA}
    fixedR=[]
    for nm,rv in rowsR:
        base = nm.replace(" Release","")
        av = amap.get(base, ["" for _ in TPS])
        for j in range(len(TPS)):
            if isinstance(rv[j],(int,float)) and isinstance(av[j],(int,float)) and rv[j] < av[j] + REL_GAP:
                rv[j] = round(av[j] + REL_GAP, 1)
        fixedR.append((nm,rv))
    rowsR = fixedR
    return rowsA, rowsR

def write_tcc(out_dir, A, R):
    os.makedirs(out_dir, exist_ok=True)
    ap = os.path.join(out_dir,"TCC_APPLY__Throttle17.tsv")
    rp = os.path.join(out_dir,"TCC_RELEASE__Throttle17.tsv")
    for path,rows in [(ap,A),(rp,R)]:
        with open(path,"w",encoding="utf-8") as f:
            f.write(HEADER+"\n")
            for nm,vals in rows: f.write(nm+"\t"+"\t".join(map(str,vals))+"\t\n")
        # fix ordinal labels
        txt=open(path,"r",encoding="utf-8").read().replace("3th","3rd")
        open(path,"w",encoding="utf-8").write(txt)
    print("[OK] wrote", ap); print("[OK] wrote", rp)

def main(list_file, out_dir):
    with open(list_file,"r",encoding="utf-8") as f:
        files = [ln.strip() for ln in f if ln.strip()]
    if not files:
        print("[FAIL] no cleaned files listed"); sys.exit(2)
    df = pd.concat([pd.read_csv(p, low_memory=False) for p in files], ignore_index=True)
    d  = prep(df)
    A,R = predict_tables(d)
    write_tcc(out_dir, A, R)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--clean-list", required=True)
    ap.add_argument("--out-dir", required=True)
    a=ap.parse_args()
    main(a.clean_list, a.out_dir)
