# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os, argparse, pandas as pd, numpy as np

TPS_AXIS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
HDR      = ["mph"] + [str(x) for x in TPS_AXIS] + ["%"]

REQ = ["speed_mph__canon","throttle_pct__canon","gear_actual__canon","brake__canon","time_s__canon"]

def die(msg): raise SystemExit(msg)
def require_columns(path):
    df = pd.read_csv(path, nrows=1000)
    miss = [c for c in REQ if c not in df.columns]
    if miss: die(f"[MISS] {path} missing: {', '.join(miss)}")

def tps_bin(v):
    # map TPS to the nearest defined TPS_AXIS bin index
    diffs = [abs(v - t) for t in TPS_AXIS]
    return int(np.argmin(diffs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-list", required=True)
    ap.add_argument("--out-dir", default=r".\newlogs\output\02_passes\STOPGO")
    ap.add_argument("--min-hits", type=int, default=6)       # << adjustable
    ap.add_argument("--delta-max", type=float, default=0.2)  # << adjustable
    ap.add_argument("--speed-max", type=float, default=12.0) # launch window
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = [p for p in open(args.clean_list,"r",encoding="utf-8").read().splitlines() if p.strip()]
    for p in files: require_columns(p)

    # low-TPS only bins we care about for stop/go
    low_bins = {6,12,19}

    # counts by (row name, TPS)
    rows = ["2 -> 1 Shift","3 -> 2 Shift","4 -> 3 Shift","5 -> 4 Shift","6 -> 5 Shift"]
    counts = { (r,t):0 for r in rows for t in TPS_AXIS }
    deltas = { (r,t):0.0 for r in rows for t in TPS_AXIS }

    # scan logs
    total_rows = 0
    for path in files:
        df = pd.read_csv(path, usecols=REQ)
        spd = pd.to_numeric(df["speed_mph__canon"], errors="coerce").values
        thr = pd.to_numeric(df["throttle_pct__canon"], errors="coerce").values
        brk = pd.to_numeric(df["brake__canon"], errors="coerce").fillna(0).values
        # gear_actual can carry strings like 'Park' -> coerce; strict/no-fallback: drop non-numeric
        gi  = pd.to_numeric(df["gear_actual__canon"], errors="coerce").values
        m   = np.isfinite(spd) & np.isfinite(thr) & np.isfinite(gi)
        spd, thr, gi, brk = spd[m], thr[m], gi[m], brk[m]

        # stop→go window: near stop, then light throttle within speed_max
        near_stop = spd < args.speed_max
        light_tps = thr <= 20.0
        idx = np.where(near_stop & light_tps)[0]
        if idx.size == 0: continue

        # count DOWN-shift related bins at low TPS during launches
        for i in idx:
            g = int(gi[i]) if np.isfinite(gi[i]) else -1
            if g in (2,3,4,5,6):
                row = f"{g} -> {g-1} Shift"
                tb  = TPS_AXIS[tps_bin(thr[i])]
                if tb in low_bins:
                    counts[(row,tb)] += 1

        total_rows += 1

    # convert counts into tiny positive DOWN bumps if coverage is met
    for r in rows:
        for tb in TPS_AXIS:
            c = counts[(r,tb)]
            if c >= args.min_hits and tb in low_bins:
                # proportional but capped; never negative
                bump = min(args.delta-max, 0.05 * c)  # 0.05 mph per qualifying hit
                deltas[(r,tb)] = round(bump, 1)

    # emit DELTA table (DOWN only)
    out_path = os.path.join(args.out_dir, "STOPGO__SHIFT_DOWN__DELTA.tsv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("mph\t" + "\t".join(map(str,TPS_AXIS)) + "\t%\n")
        for r in rows:
            row_vals = [f"{deltas[(r,tb)]:.1f}" if deltas[(r,tb)]>0 else "" for tb in TPS_AXIS]
            f.write("\t".join([r] + row_vals + [""]) + "\n")

    # quick console peek
    nonzero = sum(1 for v in deltas.values() if v>0)
    print(f"[OK] STOPGO → {out_path} | nonzero_cells={nonzero}")
    for (r,tb), c in counts.items():
        if c>0 and tb in low_bins:
            print(f"  [EVT] {r} @TPS{tb}: {c}")
if __name__ == "__main__":
    main()
