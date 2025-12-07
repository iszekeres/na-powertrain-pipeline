# -*- coding: utf-8 -*-
import os, argparse, numpy as np
from passes_common import RAW, ROWS_DN, TPS, write_delta, load_clean_list, load_raw_arrays, tps_bin

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-list", required=True)
    ap.add_argument("--out-dir", default=r".\newlogs\output\02_passes\KICKDOWN")
    ap.add_argument("--speed-min", type=float, default=20.0)
    ap.add_argument("--thr-rate", type=float, default=14.0)   # pct/sec
    ap.add_argument("--thr-abs",  type=float, default=25.0)   # %
    ap.add_argument("--window-s", type=float, default=1.4)
    ap.add_argument("--delta",    type=float, default=0.3)
    args = ap.parse_args()

    counts = np.zeros((len(ROWS_DN), len(TPS)), float)
    files = load_clean_list(args.clean_list)
    for fp in files:
        a = load_raw_arrays(fp, need=["time","speed","thr","ga"])
        t,v,thr,ga = a["time"],a["speed"],a["thr"],a["ga"]
        m = np.isfinite(t)&np.isfinite(v)&np.isfinite(thr)&np.isfinite(ga)&(v>=args.speed_min)
        t,v,thr,ga = t[m],v[m],thr[m],ga[m].astype(int)
        dthr = np.gradient(thr, t, edge_order=1) * 1.0  # pct/sec
        hot  = (dthr >= args.thr_rate) & (thr >= args.thr_abs)

        dga = np.diff(ga)
        dn = np.where(dga==-1)[0]
        for i in dn:
            if not hot[i]: continue
            g_from = ga[i]; g_to = ga[i+1]
            if not (2<=g_from<=6 and g_to==g_from-1): continue
            # small lookback window
            j0 = max(0, i - int(args.window_s*10))  # rough 10Hz min
            col = tps_bin(np.nanmean(thr[j0:i+1]))
            row = g_from-2
            counts[row,col]+=1

    # scale by “hits” but cap at --delta
    deltas = np.minimum(args.delta, 0.05*counts)
    out = write_delta(args.out_dir, "KICKDOWN__SHIFT_DOWN__DELTA.tsv", ROWS_DN, deltas)
    print(f"[OK] KICKDOWN → {out} | nonzero_cells={int((deltas!=0).sum())}")
if __name__=="__main__": main()
