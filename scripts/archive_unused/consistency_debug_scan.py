import argparse, glob, os
import pandas as pd
import numpy as np

TPS_AXIS = np.array([0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100], dtype=float)

def pick(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def nearest_tps_bin(vals):
    vals = np.clip(vals.astype(float), 0, 100)
    idx = np.abs(vals[:,None] - TPS_AXIS[None,:]).argmin(axis=1)
    return TPS_AXIS[idx].astype(int).astype(str)

def main():
    ap = argparse.ArgumentParser(description="Shift consistency debug scan")
    ap.add_argument("--glob", required=True, help="Glob for full cleaned CSVs")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--min-speed", type=float, default=2.0, help="Minimum mph")
    args = ap.parse_args()

    files = glob.glob(args.glob)
    if not files:
        print("No matching files for", args.glob)
        return

    frames = []
    for f in files:
        df = pd.read_csv(f)

        speed = pick(df, ["speed_mph__canon","speed_mph","vss_mph","Vehicle Speed"])
        tps   = pick(df, ["throttle_pct","Throttle Position"])
        gear  = pick(df, ["gear_actual"])
        if not (speed and tps and gear):
            print("[SKIP] missing columns in", os.path.basename(f))
            continue

        g  = pd.to_numeric(df[gear],  errors="coerce").to_numpy()
        v  = pd.to_numeric(df[speed], errors="coerce").to_numpy()
        tp = pd.to_numeric(df[tps],   errors="coerce").to_numpy()

        valid = np.isfinite(g)
        prev_valid = np.roll(valid, 1); prev_valid[0] = False
        change = valid & prev_valid & (g != np.roll(g, 1))
        change[0] = False
        idx = np.flatnonzero(change)
        if idx.size == 0:
            continue

        frm = g[idx-1].astype(int)
        to  = g[idx].astype(int)
        mph = v[idx]
        thr = tp[idx]

        keep = np.isfinite(frm) & np.isfinite(to) & np.isfinite(mph) & (mph >= args.min_speed)
        if not np.any(keep):
            continue

        pair = (frm[keep].astype(str) + " -> " + to[keep].astype(str) + " Shift")
        tbin = nearest_tps_bin(thr[keep])
        frames.append(pd.DataFrame({"pair": pair, "tps": tbin, "mph": mph[keep]}))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if not frames:
        pd.DataFrame(columns=["pair","tps","mph"]).to_csv(args.out, index=False)
        print("[OK] wrote empty summary to", args.out)
        return

    ev = pd.concat(frames, ignore_index=True)
    summary = (ev.groupby(["pair","tps"])["mph"]
                 .agg(n="count", mean="mean", std="std")
                 .reset_index()
                 .sort_values(["pair","tps"]))
    summary.to_csv(args.out, index=False)
    print("[OK] wrote", args.out, "rows:", len(summary))

if __name__ == "__main__":
    main()
