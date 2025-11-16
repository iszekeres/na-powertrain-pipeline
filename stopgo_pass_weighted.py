# stopgo_pass_weighted.py  (STRICT, alias-aware; no computed fallbacks)
import argparse, glob, os, sys
import numpy as np, pandas as pd
from tps_phases import is_cruise_or_mild

TPS_AXIS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
DOWN_ROWS = ["2 -> 1 Shift","3 -> 2 Shift"]

ALIAS = {
    "speed":   ["speed_mph","speed_mph__canon"],
    "pedal":   ["pedal_pct","Accelerator Pedal Position","Accelerator Pedal (%)"],
    # TPS is handled strictly from canonical throttle_pct__canon -> throttle_pct
    "brake":   ["brake"],  # CLEAN_FULL should provide this; we don't compute it here
}

def pick_col(df, keys, label):
    for k in keys:
        if k in df.columns:
            print(f"OK {label}: {k}")
            return k
    raise SystemExit(f"[STOPGO] missing required column(s) for {label}: {keys}")

def blank_down_table():
    cols = ["mph"] + [str(x) for x in TPS_AXIS] + ["%"]
    # start as object so we can mix row labels and numeric cells without dtype warnings
    df = pd.DataFrame(index=DOWN_ROWS, columns=cols, dtype=object)
    # numeric region as 0.0
    df.iloc[:, 1:-1] = 0.0
    # first column = row labels (e.g., "2 -> 1 Shift")
    df.iloc[:, 0] = df.index.astype(object)
    # last column = "%"
    df.iloc[:, -1] = "%"
    return df

def tps_bin_series(s):
    arr = np.asarray(TPS_AXIS, dtype=float)
    idx = np.abs(s.values.reshape(-1,1) - arr.reshape(1,-1)).argmin(axis=1)
    return pd.Series([str(TPS_AXIS[i]) for i in idx], index=s.index)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-speed", type=float, default=3.0)
    ap.add_argument("--max-speed", type=float, default=25.0)
    ap.add_argument("--pedal-max", type=float, default=18.0)
    ap.add_argument("--min-hits", type=int, default=6)
    ap.add_argument("--delta-mph", type=float, default=0.2)
    args = ap.parse_args()

    counts_21 = pd.Series(0, index=[str(x) for x in TPS_AXIS], dtype=int)
    counts_32 = pd.Series(0, index=[str(x) for x in TPS_AXIS], dtype=int)

    files = sorted(glob.glob(args.logs_glob))
    if not files:
        print(f"[STOPGO] no files for {args.logs_glob}", file=sys.stderr); sys.exit(2)

    any_hits = False
    for p in files:
        df = pd.read_csv(p, low_memory=False)
        print(f"[STOPGO] scanning {os.path.basename(p)}")
        speed = pick_col(df, ALIAS["speed"], "speed")
        pedal = pick_col(df, ALIAS["pedal"], "pedal")
        brake = pick_col(df, ALIAS["brake"], "brake")

        # Strict TPS canonicalization: require throttle_pct__canon
        if "throttle_pct__canon" not in df.columns:
            raise RuntimeError("STOPGO: required column 'throttle_pct__canon' missing in CLEAN_FULL")
        df["throttle_pct"] = pd.to_numeric(df["throttle_pct__canon"], errors="coerce").clip(0, 100)
        thr_col = "throttle_pct"

        q = df[
            (df[speed].astype(float).between(args.min_speed, args.max_speed)) &
            (df[brake].astype(int) == 1) &
            (df[pedal].astype(float) <= args.pedal_max)
        ]
        if q.empty:
            continue

        # TPS phase gating: only 8–25% TPS (cruise or mild accel)
        tps_series = q[thr_col]
        tps_mask = tps_series.apply(is_cruise_or_mild)
        q = q[tps_mask]
        if q.empty:
            continue

        any_hits = True
        bins = tps_bin_series(q[thr_col].astype(float))
        counts_21 = counts_21.add(bins.value_counts(), fill_value=0).astype(int)
        if (q[thr_col].astype(float) >= 12).any():
            counts_32 = counts_32.add(bins[q[thr_col].astype(float) >= 12].value_counts(), fill_value=0).astype(int)

    if not any_hits:
        print("[STOPGO] zero qualifying rows across logs; writing all-zero table.")
    out = blank_down_table()
    for col in counts_21.index:
        if counts_21[col] >= args.min_hits:
            out.loc["2 -> 1 Shift", col] = min(0.3, max(0.0, args.delta_mph))
    for col in counts_32.index:
        if counts_32[col] >= (args.min_hits + 4):
            out.loc["3 -> 2 Shift", col] = min(0.3, max(0.0, args.delta_mph))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    nz = int(np.count_nonzero(out.iloc[:,1:-1].to_numpy()))
    total = len(DOWN_ROWS)*len(TPS_AXIS)
    print(f"[STOPGO] wrote {args.out} | nonzero {nz}/{total}")
if __name__ == "__main__":
    main()


