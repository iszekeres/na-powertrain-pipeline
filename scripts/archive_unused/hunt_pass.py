# -*- coding: utf-8 -*-
"""
hunt_pass.py — Detect stable-TPS gear hunting and emit DOWN-table deltas.
RAW, strict/no-fallback columns (must exist in CLEAN_FULL):
  "Offset"               → time (s)
  "Vehicle Speed (SAE)"  → speed_mph
  "Throttle Position"    → throttle_pct
  "gear_actual"          → actual gear (1..6)
Outputs: .\newlogs\output\02_passes\HUNT\HUNT__SHIFT_DOWN__DELTA.tsv (+ DEBUG CSV)
"""
import os, argparse, numpy as np, pandas as pd

TPS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
HDR = ["mph"] + [str(x) for x in TPS] + ["%"]
ROWS_DN = ["2 -> 1 Shift","3 -> 2 Shift","4 -> 3 Shift","5 -> 4 Shift","6 -> 5 Shift"]
RAW_REQ = ["Offset","Vehicle Speed (SAE)","Throttle Position","gear_actual"]

def tps_bin(v):
    dif = [abs(v - t) for t in TPS]
    return int(np.argmin(dif))

def load_clean_list(p):
    if not os.path.exists(p): raise RuntimeError(f"[MISS] clean-list: {p}")
    with open(p,"r",encoding="utf-8") as f:
        files = [ln.strip() for ln in f if ln.strip()]
    if not files: raise RuntimeError("[MISS] clean-list is empty.")
    return files

def write_delta(out_dir, delta_mat):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "HUNT__SHIFT_DOWN__DELTA.tsv")
    df = pd.DataFrame(columns=HDR); df["mph"] = ROWS_DN
    for i in range(len(ROWS_DN)):
        vals = delta_mat[i]
        df.loc[i, df.columns[1:-1]] = [("" if (np.isnan(x) or x==0) else f"{x:.1f}") for x in vals]
        df.loc[i, "%"] = ""
    df.to_csv(out_path, sep="\t", index=False)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-list", required=True)
    ap.add_argument("--out-dir", default=r".\newlogs\output\02_passes\HUNT")
    ap.add_argument("--min-cycles", type=int, default=3)
    ap.add_argument("--max-gap-s", type=float, default=12.0)      # max seconds between oscillations
    ap.add_argument("--speed-min", type=float, default=18.0)      # ignore crawling speeds
    ap.add_argument("--tps-stability", type=float, default=6.0)   # TPS must stay within ±6%
    ap.add_argument("--delta", type=float, default=0.2)           # bump size per cell (mph)
    args = ap.parse_args()

    files = load_clean_list(args.clean_list)
    counts = np.zeros((len(ROWS_DN), len(TPS)), dtype=float)
    files_used = 0; events_total = 0; cycles_total = 0

    for fp in files:
        # Strict/no-fallback: require EXACT RAW_REQ columns
        hdr = pd.read_csv(fp, nrows=0).columns.tolist()
        miss = [c for c in RAW_REQ if c not in hdr]
        if miss: raise RuntimeError(f"[MISS] {os.path.basename(fp)} missing: {', '.join(miss)}")

        df  = pd.read_csv(fp, usecols=RAW_REQ, low_memory=False)
        t   = pd.to_numeric(df["Offset"], errors="coerce").to_numpy()
        v   = pd.to_numeric(df["Vehicle Speed (SAE)"], errors="coerce").to_numpy()
        thr = pd.to_numeric(df["Throttle Position"], errors="coerce").to_numpy()
        ga  = pd.to_numeric(df["gear_actual"], errors="coerce").to_numpy()

        mask = np.isfinite(t) & np.isfinite(v) & np.isfinite(thr) & np.isfinite(ga) & (v >= args.speed_min) & (ga>=1) & (ga<=6)
        if not mask.any(): continue
        t   = t[mask]; v = v[mask]; thr = thr[mask]; ga = ga[mask].astype(int)

        # indices where gear changes
        dga = np.diff(ga)
        change_idx = np.where(dga != 0)[0] + 1
        if change_idx.size < 2: continue

        files_used += 1
        events_total += int(change_idx.size)

        # sign of consecutive changes (up vs down)
        sign = np.sign(np.diff(ga))
        nz   = np.where(np.diff(ga)!=0)[0]
        sign = sign[nz]

        for j in range(1, len(change_idx)):
            i0 = change_idx[j-1]; i1 = change_idx[j]
            s0 = sign[j-1]; s1 = sign[j]
            if s0 == 0 or s1 == 0 or s0 == s1: continue     # need up then down (or down then up)
            if abs(ga[i1] - ga[i0]) != 1: continue          # only adjacent gears
            dt = t[i1] - t[i0]
            if not (0 < dt <= args.max_gap_s): continue

            sl = slice(min(i0,i1), max(i0,i1)+1)
            tps_span = float(np.nanmax(thr[sl]) - np.nanmin(thr[sl]))
            if tps_span > args.tps_stability: continue

            cycles_total += 1
            g_hi = int(max(ga[i0], ga[i1]))                 # DOWN table row uses the higher gear
            row_idx = g_hi - 2                              # 2->1 maps to index 0
            thr_mid = float(np.nanmean(thr[sl]))
            col_idx = tps_bin(thr_mid)
            counts[row_idx, col_idx] += 1.0

    # convert counts → mph deltas (capped); leave zeros where no signal
    deltas = np.minimum(args.delta, 0.05 * counts)          # 0.05 mph per cycle up to --delta

    out_path = write_delta(args.out_dir, deltas)
    dbg = os.path.join(args.out_dir, "HUNT__DEBUG_SUMMARY.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(dbg, "w", encoding="utf-8") as f:
        nz = int(np.sum(deltas > 0)); total = deltas.size
        f.write("files_used,events_total,cycles_total,nonzero_cells,total_cells\n")
        f.write(f"{files_used},{events_total},{cycles_total},{nz},{total}\n")

    print(f"[OK] HUNT → {out_path} | nonzero_cells={int(np.sum(deltas>0))}")
if __name__ == "__main__":
    main()
