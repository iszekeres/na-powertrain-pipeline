# -*- coding: utf-8 -*-
import os, sys
import numpy as np, pandas as pd
from passes_common import HDR, ROWS_DN

def _read(path):
    if not os.path.exists(path): return None
    df = pd.read_csv(path, sep="\t")
    M = []
    for r in ROWS_DN:
        row = pd.to_numeric(df.loc[df["mph"]==r, df.columns[1:-1]].iloc[0], errors="coerce").to_numpy()
        M.append(row)
    return np.vstack(M)

def main(core_path=None, chas_path=None, out_path=None):
    argv = sys.argv[1:]
    if len(argv) >= 3:
        core_path, chas_path, out_path = argv[:3]
    default_base = r".\newlogs\output\02_passes\CORNER"
    if core_path is None:
        core_path = os.path.join(default_base, "CORNER__SHIFT_DOWN__DELTA__CORE.tsv")
    if chas_path is None:
        chas_path = os.path.join(default_base, "CORNER__SHIFT_DOWN__DELTA__CHASSIS.tsv")
    if out_path is None:
        out_path = os.path.join(default_base, "CORNER__SHIFT_DOWN__DELTA__COMBINED.tsv")
    C = _read(core_path); H = _read(chas_path)
    if C is None or H is None:
        print("[MISS] need both CORE and CHASSIS CORNER outputs"); return
    C = np.nan_to_num(C, nan=0.0)
    H = np.nan_to_num(H, nan=0.0)
    M = np.fmax(C, H)  # elementwise max treating NaN as 0

    # format as strings
    df = pd.DataFrame(columns=HDR, dtype="object"); df["mph"] = ROWS_DN
    for i in range(len(ROWS_DN)):
        vals = [("" if x==0 else f"{x:.1f}") for x in M[i]]
        df.loc[i, df.columns[1:-1]] = vals
        df.loc[i, "%"] = ""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    nz = int((M!=0).sum())
    print(f"[OK] wrote {out_path} | nonzero_cells={nz}")
if __name__=="__main__": main()
