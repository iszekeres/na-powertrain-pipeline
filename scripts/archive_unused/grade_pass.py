#!/usr/bin/env python3
# grade_pass.py — hill sense via longitudinal accel proxy
import os, glob, pandas as pd, numpy as np
from passes_common import TPS, UP_ROWS, DN_ROWS, write_table, require_columns
CLEAN = r".\newlogs\cleaned"
OUT   = r".\newlogs\output\02_passes\GRADE"
REQ   = ["Offset","Vehicle Speed (SAE)","Throttle Position","gear_actual","Longitudinal Acceleration"]

def main():
    files=sorted(glob.glob(os.path.join(CLEAN,"__trans_focus__clean_FULL__*.csv")))
    if not files: raise SystemExit("[MISS] no CLEAN_FULL files")
    for p in files: require_columns(p, REQ)

    up  = {r: np.zeros(len(TPS), float) for r in UP_ROWS}
    dn  = {r: np.zeros(len(TPS), float) for r in DN_ROWS}
    cntu= {r: np.zeros(len(TPS), int)   for r in UP_ROWS}
    cntd= {r: np.zeros(len(TPS), int)   for r in DN_ROWS}

    for p in files:
        df  = pd.read_csv(p, usecols=REQ)
        sp  = pd.to_numeric(df["Vehicle Speed (SAE)"], errors="coerce")
        thr = pd.to_numeric(df["Throttle Position"], errors="coerce").fillna(0)
        ga  = parse_gear(df["gear_actual"])
        gx  = pd.to_numeric(df["Longitudinal Acceleration"], errors="coerce").fillna(0)

        steady = sp>10
        downhill = steady & (thr<15) & (gx < -0.02)
        uphill   = steady & (thr>20) & (gx > +0.02)

        for g,row in zip([1,2,3,4,5], UP_ROWS):
            m = uphill & (ga==g)
            if m.any():
                b = max([x for x in TPS if x <= float(thr[m].median())])
                j = TPS.index(b)
                up[row][j] += 0.2; cntu[row][j]+=1

        for g,row in zip([2,3,4,5,6], DN_ROWS):
            m = downhill & (ga==g)
            if m.any():
                b = max([x for x in TPS if x <= float(thr[m].median())])
                j = TPS.index(b)
                dn[row][j] += 0.2; cntd[row][j]+=1

    up_mat = [[(0.0 if cntu[r][i]==0 else min(0.4, up[r][i])) for i in range(len(TPS))] for r in UP_ROWS]
    dn_mat = [[(0.0 if cntd[r][i]==0 else min(0.4, dn[r][i])) for i in range(len(TPS))] for r in DN_ROWS]

    os.makedirs(OUT, exist_ok=True)
    write_table(os.path.join(OUT,"GRADE__SHIFT_UP__DELTA.tsv"),   UP_ROWS, up_mat)
    write_table(os.path.join(OUT,"GRADE__SHIFT_DOWN__DELTA.tsv"), DN_ROWS, dn_mat)
    print("[OK] GRADE written to", OUT)
if __name__=="__main__": main()
