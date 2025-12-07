#!/usr/bin/env python3
import sys, os, pandas as pd, numpy as np
TPS=[0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
HDR=["mph"]+[str(x) for x in TPS]+["%"]
def qc(path):
    if not os.path.exists(path):
        print(f"[MISS] {path}"); return 1
    df=pd.read_csv(path, sep="\t")
    ok = (list(df.columns)==HDR)
    if not ok:
        print(f"[HDR] {os.path.basename(path)} header mismatch -> {list(df.columns)}"); return 2
    names=list(df["mph"])
    print(f"[OK] {os.path.basename(path)} rows={names}")
    # simple stats
    vals=pd.to_numeric(df.iloc[:,1:-1].stack(), errors="coerce")
    mn, mx = float(np.nanmin(vals)), float(np.nanmax(vals))
    print(f"     value range: {mn:.3g} .. {mx:.3g}")
    return 0
if __name__=="__main__":
    for p in sys.argv[1:]:
        qc(p)
