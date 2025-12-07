# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd
from passes_common import (ROW_DN, TPS, HDR)

def load(path):
    if not os.path.exists(path): return None
    return pd.read_csv(path, sep="\t")

def main(core=r".\newlogs\output\02_passes\CORNER\CORNER__SHIFT_DOWN__DELTA__CORE.tsv",
         chassis=r".\newlogs\output\02_passes\CORNER\CORNER__SHIFT_DOWN__DELTA__CHASSIS.tsv",
         out=r".\newlogs\output\02_passes\CORNER\CORNER__SHIFT_DOWN__DELTA__COMBINED.tsv"):
    dc = load(core); dh = load(chassis)
    if dc is None or dh is None:
        print("[MISS] need both CORE and CHASSIS for combine")
        return
    outdf = dc.copy()
    for i in range(len(dc)):
        a = pd.to_numeric(dc.iloc[i,1:-1], errors="coerce").to_numpy()
        b = pd.to_numeric(dh.iloc[i,1:-1], errors="coerce").to_numpy()
        outdf.iloc[i,1:-1] = np.maximum(a,b)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    outdf.to_csv(out, sep="\t", index=False)
    print("[OK] wrote", out)
if __name__=="__main__": main()
