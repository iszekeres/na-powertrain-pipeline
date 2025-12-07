#!/usr/bin/env python3
# corner_exit_pass_weighted__shim.py
# Build CORNER__SHIFT_DOWN__DELTA.tsv from CLEAN files using simple, explicit gates.
#
# Logic:
# - Identify "corner-exit" moments: speed in [--min-speed, --max-speed] mph AND d(throttle)/dt >= --thr-rate %/s.
# - Bin events by (gear_actual, TPS 17-pt bins). Require >= --min-hits per bin.
# - For qualifying bins, propose a positive delta (+--delta-mph) in the corresponding DOWN row:
#     gear 2 -> "2 -> 1 Shift"
#     gear 3 -> "3 -> 2 Shift"
#     gear 4 -> "4 -> 3 Shift"
#     gear 5 -> "5 -> 4 Shift"
#     gear 6 -> "6 -> 5 Shift"
# - Output format: "mph … %" header with the 17 TPS bins.
#
# Usage:
#   python corner_exit_pass_weighted__shim.py \
#       --logs-glob ".\06_Logs\Trans_Review\__trans_focus__clean__*withtime*.csv" \
#       --out "C:\tuning\logs\06_Logs\Trans_Review\CORNER__SHIFT_DOWN__DELTA.tsv" \
#       --min-speed 8 --max-speed 30 --thr-rate 18 --min-hits 80 --delta-mph 0.3
#
import os, glob, argparse, numpy as np, pandas as pd
from tps_phases import is_corner_exit_band

TPS_AXIS17 = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
DOWN_ROWS = ["2 -> 1 Shift","3 -> 2 Shift","4 -> 3 Shift","5 -> 4 Shift","6 -> 5 Shift"]
HEADER = ["mph"] + [str(x) for x in TPS_AXIS17] + ["%"]

def ensure_corner_canon(df, src_name):
    """
    Ensure CORNER has gear_actual, speed_mph, throttle_pct columns.
    If the raw names are missing or all-NaN, alias them from canonical fields.
    """
    alias_map = {
        "gear_actual": [
            "gear_actual__canon",
            "Gear Actual",
            "Trans Current Gear",
        ],
        "speed_mph": [
            "speed_mph__canon",
            "Vehicle Speed (SAE)",
            "Speed MPH",
        ],
        # TPS: require canonical throttle_pct__canon (no fallback to raw TPS)
        "throttle_pct": [
            "throttle_pct__canon",
        ],
    }

    for target, candidates in alias_map.items():
        need_alias = (target not in df.columns) or df[target].isna().all()
        if need_alias:
            for cand in candidates:
                if cand in df.columns and not df[cand].isna().all():
                    df[target] = df[cand]
                    break

    key_cols = ["gear_actual","speed_mph","throttle_pct"]
    missing = [c for c in key_cols if c not in df.columns or df[c].isna().all()]
    if missing:
        raise RuntimeError(f"{src_name}: missing required column(s) for CORNER after alias: {missing}")

    df = df.dropna(subset=key_cols, how="any").copy()
    return df

def tps_bin_value(x):
    # Return the axis value for the bin (not the index)
    if np.isnan(x): return TPS_AXIS17[0]
    idx = np.searchsorted(TPS_AXIS17, x, side='right') - 1
    return TPS_AXIS17[max(0, min(idx, len(TPS_AXIS17)-1))]

def main():
    ap = argparse.ArgumentParser(description="Corner-exit shim (generates SHIFT_DOWN deltas).")
    ap.add_argument("--logs-glob", required=True, help="Glob of CLEAN files (need time_s, throttle_pct, speed_mph, gear_actual)")
    ap.add_argument("--out", required=True, help="Output TSV path (CORNER__SHIFT_DOWN__DELTA.tsv)")
    ap.add_argument("--min-speed", type=float, default=8.0)
    ap.add_argument("--max-speed", type=float, default=30.0)
    ap.add_argument("--thr-rate", type=float, default=18.0, help="min d(throttle)/dt in %%/s")
    ap.add_argument("--min-hits", type=int, default=80, help="min events per (gear,TPSbin) to emit a delta")
    ap.add_argument("--delta-mph", type=float, default=0.3, help="positive mph added to DOWN targets")
    args = ap.parse_args()

    files = sorted(glob.glob(args.logs_glob))
    if not files:
        raise SystemExit("No files matched: " + args.logs_glob)

    # Initialize counts
    counts = {(row, tps): 0 for row in DOWN_ROWS for tps in TPS_AXIS17}

    for fp in files:
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception:
            df = pd.read_csv(fp, low_memory=False, engine="python", on_bad_lines="skip")

        try:
            df = ensure_corner_canon(df, os.path.basename(fp))
        except RuntimeError as exc:
            print(f"[WARN] {exc}; skipping")
            continue

        t   = pd.to_numeric(df["time_s"], errors="coerce")
        thr = pd.to_numeric(df["throttle_pct"], errors="coerce").clip(0,100)
        v   = pd.to_numeric(df["speed_mph"], errors="coerce")
        g   = pd.to_numeric(df["gear_actual"], errors="coerce").round().astype("Int64")

        dt = t.diff().replace(0, np.nan)
        dthr = (thr.diff() / dt).replace([np.inf,-np.inf], np.nan)

        window = v.between(args.min_speed, args.max_speed) & g.between(2,6)
        hit = window & (dthr >= args.thr_rate)

        # TPS phase gating: require TPS in corner-exit band (>= ~12%)
        tps_ok = thr.apply(is_corner_exit_band)
        hit = hit & tps_ok

        # Aggregate
        sel = pd.DataFrame({
            "gear": g.where(hit),
            "tpsbin": thr.where(hit).apply(tps_bin_value)
        }).dropna()
        # Map gear -> row label
        sel["row"] = sel["gear"].map({2:"2 -> 1 Shift",3:"3 -> 2 Shift",4:"4 -> 3 Shift",5:"5 -> 4 Shift",6:"6 -> 5 Shift"})
        grp = sel.groupby(["row","tpsbin"]).size()
        for (row, tpsv), n in grp.items():
            counts[(row, int(tpsv))] += int(n)

    # Build delta matrix
    data = []
    for row in DOWN_ROWS:
        vals = []
        for tpsv in TPS_AXIS17:
            n = counts[(row, tpsv)]
            vals.append(args.delta_mph if n >= args.min_hits else 0.0)
        data.append([row] + vals + [""])

    out_df = pd.DataFrame(data, columns=HEADER)
    # Ensure 0.1 mph resolution formatting
    for c in out_df.columns[1:-1]:
        out_df[c] = out_df[c].map(lambda x: f"{x:.1f}" if isinstance(x, (int,float)) else x)
    out_df.to_csv(args.out, sep="\t", index=False)
    print("[CORNER_SHIM] wrote", args.out)

if __name__ == "__main__":
    main()
