# -*- coding: utf-8 -*-
# shift_consistency_pass_weighted__RELAXED.py
#
# Purpose: produce nonzero CONSIST deltas by relaxing hard gates.
# Input:  FULL cleaned logs (with brake/time), Throttle17 UP/DOWN base tables
# Output: <out-prefix>__SHIFT_UP__DELTA.tsv and __SHIFT_DOWN__DELTA.tsv
#
# Usage (example):
#   python .\shift_consistency_pass_weighted__RELAXED.py ^
#     --logs-glob ".\newlogs\cleaned\__trans_focus__clean_FULL__*withbrake*.csv" ^
#     --out-prefix ".\newlogs\output\CONSIST_TUNE1\CONSIST" ^
#     --up ".\06_Logs\Trans_Review\SHIFT_TABLES__UP__Throttle17.tsv" ^
#     --down ".\06_Logs\Trans_Review\SHIFT_TABLES__DOWN__Throttle17.tsv" ^
#     --min-n 6 --std-max 4.0 --max-step 0.2 --range-mode loose ^
#     --debug-summary-out ".\newlogs\output\CONSIST_TUNE1\CONSIST__DEBUG_SUMMARY.csv"

import argparse, glob, os
import numpy as np
import pandas as pd

TPS_AXIS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
ROWS_UP  = ["1 -> 2 Shift","2 -> 3 Shift","3 -> 4 Shift","4 -> 5 Shift","5 -> 6 Shift"]
ROWS_DN  = ["2 -> 1 Shift","3 -> 2 Shift","4 -> 3 Shift","5 -> 4 Shift","6 -> 5 Shift"]
ALL_ROWS = set(ROWS_UP + ROWS_DN)

PAIR_SPEED_TIGHT = {
  "1 -> 2 Shift": (4,25),  "2 -> 3 Shift": (10,35), "3 -> 4 Shift": (25,55),
  "4 -> 5 Shift": (40,70), "5 -> 6 Shift": (55,90),
  "2 -> 1 Shift": (3,24),  "3 -> 2 Shift": (9,34),  "4 -> 3 Shift": (24,54),
  "5 -> 4 Shift": (39,69), "6 -> 5 Shift": (54,89),
}

def pair_range(pair, mode="loose"):
    lo, hi = PAIR_SPEED_TIGHT.get(pair, (0, 999))
    if mode == "tight":
        return lo, hi
    if mode == "loose":
        return max(0, lo-5), hi+5
    return (0, 9999)  # none

ALIASES = {
    "speed": [
        "speed_mph__canon","speed_mph","Vehicle Speed","Vehicle Speed (SAE)","VSS mph","vss_mph"
    ],
    "thr": [
        "throttle_pct","Throttle Position","Throttle Position (%)","Throttle (%)"
    ],
    "gear": [
        "gear_actual","Trans Current Gear","Transmission Current Gear","Trans Current Gear (SAE)",
        "Current Gear","Gear Current","Trans_Gear_Current","G_Cur","Gear_Actual","GEAR_ACTUAL"
    ],
}

def find_col(cols, wants):
    cs = {c.lower(): c for c in cols}
    for w in wants:
        lw = w.lower()
        if lw in cs: return cs[lw]
    # substring fallback
    for c in cols:
        lc = c.lower()
        for w in wants:
            if w.lower() in lc:
                return c
    return None

def snap_tps(x):
    if np.isnan(x): return None
    return min(TPS_AXIS, key=lambda v: abs(v - float(x)))

def tsv_to_map(path):
    df = pd.read_csv(path, sep="\t")
    # numeric columns for TPS
    for c in df.columns[1:-1]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    m = {}
    for _,row in df.iterrows():
        rname = str(row.iloc[0])
        m[rname] = np.array([float(row[str(t)]) if str(t) in df.columns else np.nan for t in TPS_AXIS])
    return m

def blank_delta(rows):
    header = ["mph"] + [str(t) for t in TPS_AXIS] + ["%"]
    data = []
    for r in rows:
        data.append([r] + [0.0]*len(TPS_AXIS) + [np.nan])
    return pd.DataFrame(data, columns=header)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-glob", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--up", required=True, help="Throttle17 UP table TSV")
    ap.add_argument("--down", required=True, help="Throttle17 DOWN table TSV")
    ap.add_argument("--min-n", type=int, default=6)
    ap.add_argument("--std-max", type=float, default=4.0)
    ap.add_argument("--max-step", type=float, default=0.2)
    ap.add_argument("--range-mode", choices=["tight","loose","none"], default="loose")
    ap.add_argument("--debug-summary-out", default="", help="Optional CSV of (pair,tps,n,mean,std)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.logs_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.logs_glob}")

    rows = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        speed = find_col(df.columns, ALIASES["speed"])
        thr   = find_col(df.columns, ALIASES["thr"])
        gear  = find_col(df.columns, ALIASES["gear"])
        if not (speed and thr and gear):
            print(f"[SKIP] {os.path.basename(f)} missing columns (speed/thr/gear)")
            continue

        s = pd.to_numeric(df[speed], errors="coerce")
        t = pd.to_numeric(df[thr], errors="coerce")
        g = pd.to_numeric(df[gear], errors="coerce").ffill().bfill()

        # find gear transitions
        g_prev = g.shift(1)
        trans = g != g_prev
        trans.iloc[0] = False

        idx = np.where(trans.values)[0]
        for i in idx:
            gp, gc = g_prev.iloc[i], g.iloc[i]
            if pd.isna(gp) or pd.isna(gc): continue
            gp = int(gp); gc = int(gc)
            if gp == gc: continue
            pair = f"{gp} -> {gc} Shift"
            if pair not in ALL_ROWS: continue

            spd = float(s.iloc[i]) if not pd.isna(s.iloc[i]) else None
            thrv= float(t.iloc[i]) if not pd.isna(t.iloc[i]) else None
            if spd is None or thrv is None: continue

            tb = snap_tps(thrv)
            if tb is None: continue
            rows.append((pair, tb, spd))

    if not rows:
        up_out   = f"{args.out_prefix}__SHIFT_UP__DELTA.tsv"
        down_out = f"{args.out_prefix}__SHIFT_DOWN__DELTA.tsv"
        blank_delta(ROWS_UP).to_csv(up_out, sep="\t", index=False)
        blank_delta(ROWS_DN).to_csv(down_out, sep="\t", index=False)
        print(f"[CONSIST] No transitions found; wrote zero-delta files:\n  {up_out}\n  {down_out}")
        return

    raw = pd.DataFrame(rows, columns=["pair","tps","mph"])
    raw = raw[raw["pair"].isin(ALL_ROWS)].copy()

    grp = raw.groupby(["pair","tps"]).agg(n=("mph","size"),
                                          mean=("mph","mean"),
                                          std=("mph","std")).reset_index()
    grp["std"] = grp["std"].fillna(99.0)

    def in_range(r):
        lo,hi = pair_range(r["pair"], args.range_mode)
        return (r["mean"] >= lo) and (r["mean"] <= hi)
    mask = grp.apply(in_range, axis=1)

    gated = grp[mask & (grp["n"] >= args.min_n) & (grp["std"] <= args.std_max)].copy()

    if args.debug_summary_out:
        os.makedirs(os.path.dirname(args.debug_summary_out), exist_ok=True)
        gated.to_csv(args.debug_summary_out, index=False)

    up_map   = tsv_to_map(args.up)
    down_map = tsv_to_map(args.down)
    du = blank_delta(ROWS_UP)
    dd = blank_delta(ROWS_DN)

    def weight(n, std):
        return max(0.0, min(1.0, (n/40.0))) * (1.0 / (1.0 + max(0.0, std)))

    def apply(pairset, table_map, df_delta):
        for _,r in gated.iterrows():
            pair = r["pair"]
            if pair not in pairset: continue
            tps  = int(r["tps"])
            base = table_map.get(pair, None)
            if base is None: continue
            j = TPS_AXIS.index(tps)
            delta_raw = float(r["mean"] - base[j])
            step = np.clip(delta_raw, -args.max_step, args.max_step) * weight(r["n"], r["std"])
            ridx = df_delta.index[df_delta["mph"] == pair][0]
            df_delta.at[ridx, str(tps)] = float(df_delta.at[ridx, str(tps)]) + float(step)

    apply(set(ROWS_UP), up_map, du)
    apply(set(ROWS_DN), down_map, dd)

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    up_out   = f"{args.out_prefix}__SHIFT_UP__DELTA.tsv"
    down_out = f"{args.out_prefix}__SHIFT_DOWN__DELTA.tsv"
    du.to_csv(up_out,   sep="\t", index=False)
    dd.to_csv(down_out, sep="\t", index=False)

    def nz(df):
        cols = df.columns[1:-1]
        vals = pd.to_numeric(df[cols].stack(), errors="coerce").fillna(0).values
        return int((np.abs(vals)>0).sum()), vals.size
    nzu, su = nz(du)
    nzd, sd = nz(dd)
    print(f"[CONSIST] wrote:\n  {up_out}\n  {down_out}\n  nonzero: {nzu}/{su} (UP), {nzd}/{sd} (DOWN)")
    if args.debug_summary_out:
        print(f"[CONSIST] debug summary: {args.debug_summary_out} (rows={len(gated)})")

if __name__ == "__main__":
    main()
