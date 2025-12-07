# -*- coding: utf-8 -*-
import os, argparse, numpy as np
from passes_common import RAW, ROWS_DN, TPS, write_delta, load_clean_list, load_raw_arrays, tps_bin, require_columns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-list", required=True)
    ap.add_argument("--out-dir", default=r".\newlogs\output\02_passes\CORNER")
    ap.add_argument("--min-speed", type=float, default=8.0)
    ap.add_argument("--max-speed", type=float, default=30.0)
    ap.add_argument("--latg",      type=float, default=0.12)
    ap.add_argument("--yaw",       type=float, default=18.0)
    ap.add_argument("--steer",     type=float, default=65.0)
    ap.add_argument("--min-hits",  type=int,   default=8)
    ap.add_argument("--delta",     type=float, default=0.3)
    args = ap.parse_args()

    counts = np.zeros((len(ROWS_DN), len(TPS)), float)
    files = load_clean_list(args.clean_list)
    for fp in files:
        # hard require chassis channels
        require_columns(fp, [RAW["time"],RAW["speed"],RAW["thr"],RAW["ga"],RAW["latg"],RAW["yaw"],RAW["steer"]])
        a = load_raw_arrays(fp, need=["time","speed","thr","ga","latg","yaw","steer"])
        t,v,thr,ga,lg,yw,st = a["time"],a["speed"],a["thr"],a["ga"],a["latg"],a["yaw"],a["steer"]
        m = np.isfinite(t)&np.isfinite(v)&np.isfinite(thr)&np.isfinite(ga)&np.isfinite(lg)&np.isfinite(yw)&np.isfinite(st)
        t,v,thr,ga,lg,yw,st = t[m],v[m],thr[m],ga[m].astype(int),lg[m],yw[m],st[m]
        mspd = (v>=args.min_speed)&(v<=args.max_speed)
        mchs = (np.abs(lg)>=args.latg) | (np.abs(yw)>=args.yaw) | (np.abs(st)>=args.steer)
        dn = np.where(np.diff(ga)==-1)[0]
        for i in dn:
            if not (mspd[i] and mchs[i]): continue
            g_from=ga[i]; g_to=ga[i+1]
            if not (2<=g_from<=6 and g_to==g_from-1): continue
            row = g_from-2; col = tps_bin(thr[i])
            counts[row,col]+=1

    deltas = np.where(counts>=args.min_hits, args.delta, 0.0)
    p = write_delta(args.out_dir, "CORNER_CHASSIS__SHIFT_DOWN__DELTA.tsv", ROWS_DN, deltas)
    print(f"[OK] CORNER CHASSIS → {p} | nonzero_cells={int((deltas!=0).sum())}")
if __name__=="__main__": main()
