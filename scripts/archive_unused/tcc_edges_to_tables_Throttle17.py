import argparse, pandas as pd, numpy as np, os, sys

TPS_AXIS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
TPS_STR  = [str(x) for x in TPS_AXIS]

def pick(df, candidates, what):
    for c in candidates:
        if c in df.columns: return c
    raise SystemExit(f"[ERROR] Need a {what} column; tried: {candidates}")

def tps_bin_series(vals):
    a = np.asarray(TPS_AXIS, dtype=float)
    v = pd.to_numeric(vals, errors="coerce").fillna(0).to_numpy().reshape(-1,1)
    idx = np.abs(v - a.reshape(1,-1)).argmin(axis=1)
    return pd.Series([TPS_STR[i] for i in idx], index=vals.index)

def build_table(edges_df, stat, min_n, row_map):
    speed_col = pick(edges_df,
                     ["speed_mph","mph","Vehicle Speed","Vehicle Speed (mph)","VSS mph"],
                     "speed (mph)")
    thr_col   = pick(edges_df,
                     ["throttle_pct","Throttle Position (%)","Accelerator Pedal Position"],
                     "throttle/pedal")
    gear_col  = pick(edges_df, ["gear","gear_actual"], "gear")

    if "tps_bin" not in edges_df.columns:
        edges_df["tps_bin"] = tps_bin_series(edges_df[thr_col])

    # aggregate mph by gear × TPS bin
    agg = edges_df.groupby([gear_col,"tps_bin"])[speed_col].agg(stat).to_frame("mph")
    counts = edges_df.groupby([gear_col,"tps_bin"]).size().to_frame("n")

    # assemble output grid
    cols = ["mph"] + TPS_STR + ["%"]
    out  = pd.DataFrame(columns=cols)

    for g, row_name in row_map.items():
        row = {c: "" for c in cols}
        row["mph"] = row_name
        row["%"]   = "%"
        for t in TPS_STR:
            n_here = counts["n"].get((g,t), 0)
            if n_here >= min_n:
                val = float(agg["mph"].get((g,t), np.nan))
                if np.isfinite(val):
                    row[t] = round(val, 1)
        out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)

    # header row first, as in HPT paste format
    hdr = pd.DataFrame([dict(zip(cols, ["mph"] + TPS_STR + ["%"]))])
    out = pd.concat([hdr, out], ignore_index=True)
    return out[cols]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply",   required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-n",   type=int, default=3)
    ap.add_argument("--stat",    choices=["median","mean"], default="median")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dfA = pd.read_csv(args.apply,   sep="\t")
    dfR = pd.read_csv(args.release, sep="\t")

    apply_rows   = {3:"3rd Apply", 4:"4th Apply", 5:"5th Apply", 6:"6th Apply"}
    release_rows = {3:"3rd Release",4:"4th Release",5:"5th Release",6:"6th Release"}

    A = build_table(dfA, args.stat, args.min_n, apply_rows)
    R = build_table(dfR, args.stat, args.min_n, release_rows)

    pA = os.path.join(args.out_dir, "TCC_APPLY__Throttle17.tsv")
    pR = os.path.join(args.out_dir, "TCC_RELEASE__Throttle17.tsv")
    A.to_csv(pA, sep="\t", index=False)
    R.to_csv(pR, sep="\t", index=False)

    # quick coverage print
    def nonzero(df):
        num = pd.to_numeric(df.iloc[1:,1:-1].stack(), errors="coerce")
        return int(np.count_nonzero(np.isfinite(num))), int(num.size)
    nzA, totA = nonzero(A); nzR, totR = nonzero(R)
    print(f"[TCC_EDGES→TABLES] wrote {pA} ({nzA}/{totA} cells) and {pR} ({nzR}/{totR} cells).")

if __name__ == "__main__":
    main()
