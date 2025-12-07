# -*- coding: utf-8 -*-
import os, argparse, numpy as np, pandas as pd
from passes_common import RAW, ROWS_UP, TPS, write_delta, load_clean_list, load_raw_arrays, tps_bin

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-list", required=True)
    ap.add_argument("--out-dir", default=r".\newlogs\output\02_passes\LAT")
    ap.add_argument("--min-events", type=int, default=30)
    ap.add_argument("--delta", type=float, default=0.3)   # magnitude
    args = ap.parse_args()

    counts = np.zeros((len(ROWS_UP), len(TPS)), float)
    files = load_clean_list(args.clean_list)

    for fp in files:
        a = load_raw_arrays(fp, need=["time","speed","thr","ga"])
        t,v,thr,ga = a["time"],a["speed"],a["thr"],a["ga"]
        m = np.isfinite(t)&np.isfinite(v)&np.isfinite(thr)&np.isfinite(ga)
        t,v,thr,ga = t[m],v[m],thr[m],ga[m].astype(int)
        # detect upshifts (adjacent +1)
        d = np.diff(ga); up_idx = np.where(d==+1)[0]
        for i in up_idx:
            g_from = ga[i]; g_to = ga[i+1]
            if not (1<=g_from<=5 and g_to==g_from+1): continue
            row = g_from-1  # 1->2 maps to index 0
            col = tps_bin(thr[i])
            counts[row,col]+=1

    # apply constant advance where coverage is good
    deltas = np.zeros_like(counts)
    deltas[counts >= args.min_events] = -args.delta
    out = write_delta(args.out_dir, "LAT__SHIFT_UP__DELTA.tsv", ROWS_UP, deltas)
    dbg = os.path.join(args.out_dir,"LAT__DEBUG_SUMMARY.csv")
    with open(dbg,"w",encoding="utf-8") as f:
        nz=int(np.sum(deltas!=0)); tot=deltas.size
        f.write("files_used,up_events,min_events,nonzero_cells,total_cells\n")
        f.write(f"{len(files)},{int(np.sum(counts))},{args.min_events},{nz},{tot}\n")
    print(f"[OK] LAT → {out}")
if __name__=="__main__": main()
