# -*- coding: utf-8 -*-
import os, argparse, numpy as np
from passes_common import ROWS_DN, TPS, write_delta, load_clean_list, load_raw_arrays, tps_bin

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-list", required=True)
    ap.add_argument("--out-dir", default=r".\newlogs\output\02_passes\STOPGO")
    ap.add_argument("--speed-max", type=float, default=12.0)
    ap.add_argument("--thr-max",   type=float, default=22.0)
    ap.add_argument("--delta",     type=float, default=0.3)
    ap.add_argument("--min-hits",  type=int,   default=6)
    args = ap.parse_args()

    counts = np.zeros((len(ROWS_DN), len(TPS)), float)
    files = load_clean_list(args.clean_list)
    for fp in files:
        a = load_raw_arrays(fp, need=["time","speed","thr","ga","brake"])
        t,v,thr,ga,brk = a["time"],a["speed"],a["thr"],a["ga"],a["brake"]
        m = (v <= args.speed_max) & (thr <= args.thr_max)
        m &= np.isfinite(t)&np.isfinite(v)&np.isfinite(thr)&np.isfinite(ga)
        t,v,thr,ga = t[m],v[m],thr[m],ga[m].astype(int)

        dga = np.diff(ga)
        dn = np.where(dga==-1)[0]
        for i in dn:
            g_from = ga[i]; g_to = ga[i+1]
            if not (2<=g_from<=6 and g_to==g_from-1): continue
            row = g_from-2
            col = tps_bin(thr[i])
            counts[row,col]+=1.0

    deltas = np.where(counts>=args.min_hits, args.delta, 0.0)
    out = write_delta(args.out_dir, "STOPGO__SHIFT_DOWN__DELTA.tsv", ROWS_DN, deltas)
    print(f"[OK] STOPGO → {out} | nonzero_cells={int((deltas!=0).sum())}")
if __name__=="__main__": main()
