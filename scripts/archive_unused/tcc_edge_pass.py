# -*- coding: utf-8 -*-
import os, argparse, numpy as np, pandas as pd
from passes_common import RAW, TPS, write_delta, load_clean_list, load_raw_arrays

ROWS_REL = [f"{g} Release" for g in ["1st","2nd","3rd","4th","5th","6th"]]
ROWS_APP = [f"{g} Apply"   for g in ["1st","2nd","3rd","4th","5th","6th"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-list", required=True)
    ap.add_argument("--out-dir", default=r".\newlogs\output\02_passes\TCC_EDGE")
    ap.add_argument("--on-rpm",  type=float, default=30.0)  # lock if slip <= on_rpm for >= on_s
    ap.add_argument("--on-s",    type=float, default=0.6)
    ap.add_argument("--off-rpm", type=float, default=80.0)  # unlock if slip >= off_rpm for >= off_s
    ap.add_argument("--off-s",   type=float, default=0.4)
    args = ap.parse_args()

    files = load_clean_list(args.clean_list)
    counts = np.zeros((6, len(TPS)), float)  # by gear (3..6 effective) x TPS
    files_used = 0

    for fp in files:
        a = load_raw_arrays(fp, need=["time","speed","thr","ga","eng","turb","brake"])
        t,v,thr,ga,eng,turb,brk = a["time"],a["speed"],a["thr"],a["ga"],a["eng"],a["turb"],a["brake"]
        m = np.isfinite(t)&np.isfinite(v)&np.isfinite(thr)&np.isfinite(ga)&np.isfinite(eng)&np.isfinite(turb)
        t,v,thr,ga,eng,turb,brk = t[m],v[m],thr[m],ga[m].astype(int),eng[m],turb[m],brk[m]
        slip = eng - turb  # rpm difference; neutral-first approximation
        if len(slip)==0: continue
        files_used += 1

        # crude state: soft-lock when slip <= on_rpm, unlock when >= off_rpm
        lock = np.zeros_like(slip, dtype=bool)
        dur = 0.0
        for i in range(len(slip)):
            if slip[i] <= args.on_rpm:
                dur += (t[i]-t[i-1]) if i>0 else 0.0
                if dur >= args.on_s: lock[i] = True
            else:
                dur = 0.0

        for i in range(1,len(slip)):
            g = ga[i]
            if g<3 or g>6: continue  # we score 3rd-6th only
            col = int(np.argmin(np.abs(np.array(TPS)-thr[i])))
            counts[g-1, col] += 1.0 if lock[i] else 0.0

    # Neutral (no bias) deltas unless strong evidence exists; we’ll just emit debug CSV
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir,"TCC_EDGE__DEBUG_COUNTS.csv"),"w",encoding="utf-8") as f:
        f.write("gear,tps,count\n")
        for g in range(1,7):
            for j,c in enumerate(counts[g-1]):
                f.write(f"{g},{TPS[j]},{int(c)}\n")
    # Emit zero-delta placeholder tables for safety
    zero_rel = np.zeros((6, len(TPS))); zero_app = np.zeros_like(zero_rel)
    pR = write_delta(args.out_dir, "TCC_EDGE__RELEASE__DELTA.tsv", ROWS_REL, zero_rel)
    pA = write_delta(args.out_dir, "TCC_EDGE__APPLY__DELTA.tsv",   ROWS_APP, zero_app)
    print(f"[OK] TCC EDGE -> {pR} / {pA} (neutral deltas)")
if __name__=="__main__": main()
