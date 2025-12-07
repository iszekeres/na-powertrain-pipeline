# tcc_table_builder_Throttle17.py  (streaming + progress, robust to bad lines)
import argparse, glob, os
import numpy as np, pandas as pd

TPS_AXIS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
NEEDED   = {"speed_mph","throttle_pct","gear_actual","tcc_locked_built"}

def axis_header():
    return ["mph"] + [str(x) for x in TPS_AXIS] + ["%"]

def ordinal(g):
    return ["1st","2nd","3rd","4th","5th","6th"][g-1] if 1<=g<=6 else f"{g}th"

def bin_for_tps(t):
    if pd.isna(t): return None
    if t >= 100: return 100
    for i in range(len(TPS_AXIS)-1):
        a,b = TPS_AXIS[i], TPS_AXIS[i+1]
        if t >= a and t < b: return a
    return TPS_AXIS[0]

def edges_from_lock(lock):
    d = lock.diff().fillna(0)
    ups = d[d>0].index.values   # 0->1 Apply
    dns = d[d<0].index.values   # 1->0 Release
    return ups, dns

def edges_from_df(df):
    mph  = pd.to_numeric(df["speed_mph"], errors="coerce")
    tps  = pd.to_numeric(df["throttle_pct"], errors="coerce")
    gear = pd.to_numeric(df["gear_actual"], errors="coerce").round().clip(1,6).astype("Int64")
    lock = pd.to_numeric(df["tcc_locked_built"], errors="coerce").fillna(0).astype(int).clip(0,1)
    ups, dns = edges_from_lock(lock)
    rows = []
    for idx in ups:
        if idx < len(mph) and idx < len(tps) and idx < len(gear) and pd.notna(gear.iat[idx]):
            rows.append(("APPLY", int(gear.iat[idx]), float(tps.iat[idx]), float(mph.iat[idx])))
    for idx in dns:
        if idx < len(mph) and idx < len(tps) and idx < len(gear) and pd.notna(gear.iat[idx]):
            rows.append(("RELEASE", int(gear.iat[idx]), float(tps.iat[idx]), float(mph.iat[idx])))
    return pd.DataFrame(rows, columns=["type","gear","tps","mph"])

def collect_edges_streaming(path, chunksize=200000):
    """Python-engine, on_bad_lines=skip, streaming to avoid parser stalls"""
    rows = []
    prev_lock = None
    try:
        it = pd.read_csv(
            path,
            usecols=lambda c: c in NEEDED,
            encoding="utf-8-sig",
            engine="python",
            on_bad_lines="skip",
            chunksize=chunksize
        )
    except Exception as e:
        print(f"[WARN] Streaming open failed for {os.path.basename(path)}: {e}")
        return pd.DataFrame(columns=["type","gear","tps","mph"])

    total = 0
    for chunk in it:
        total += len(chunk)
        mph  = pd.to_numeric(chunk.get("speed_mph"), errors="coerce")
        tps  = pd.to_numeric(chunk.get("throttle_pct"), errors="coerce")
        gear = pd.to_numeric(chunk.get("gear_actual"), errors="coerce").round().clip(1,6).astype("Int64")
        lock = pd.to_numeric(chunk.get("tcc_locked_built"), errors="coerce").fillna(0).astype(int).clip(0,1)

        # boundary-aware diff
        d = lock.diff()
        if prev_lock is not None and len(lock):
            d.iloc[0] = lock.iloc[0] - prev_lock
        else:
            d.iloc[0] = 0 if len(lock) else 0
        prev_lock = lock.iloc[-1] if len(lock) else prev_lock

        ups = d[d>0].index.values
        dns = d[d<0].index.values
        for idx in ups:
            if idx < len(mph) and idx < len(tps) and idx < len(gear) and pd.notna(gear.iat[idx]):
                rows.append(("APPLY", int(gear.iat[idx]), float(tps.iat[idx]), float(mph.iat[idx])))
        for idx in dns:
            if idx < len(mph) and idx < len(tps) and idx < len(gear) and pd.notna(gear.iat[idx]):
                rows.append(("RELEASE", int(gear.iat[idx]), float(tps.iat[idx]), float(mph.iat[idx])))

        if total % (chunksize*5) == 0:
            print(f"    .. {os.path.basename(path)} parsed rows: {total}")

    return pd.DataFrame(rows, columns=["type","gear","tps","mph"])

def collect_edges_from_clean(path):
    size_mb = (os.path.getsize(path) / (1024*1024.0)) if os.path.exists(path) else 0.0
    print(f"[SCAN] {os.path.basename(path)}  ({size_mb:.1f} MB)")

    # Fast path: C-engine with usecols
    try:
        df = pd.read_csv(path, usecols=lambda c: c in NEEDED, encoding="utf-8-sig", low_memory=False)
        if not NEEDED.issubset(df.columns):
            missing = [c for c in NEEDED if c not in df.columns]
            print(f"[WARN] {os.path.basename(path)} missing columns: {', '.join(missing)}")
            return pd.DataFrame(columns=["type","gear","tps","mph"])
        return edges_from_df(df)
    except Exception as e:
        print(f"[INFO] Falling back to python-engine streaming for {os.path.basename(path)}: {e}")
        return collect_edges_streaming(path)

def build_table(edges_df, labeler):
    stats = {}
    for _,r in edges_df.iterrows():
        key = labeler(int(r["gear"]))
        tb = bin_for_tps(r["tps"])
        if tb is None: continue
        mph = r["mph"]
        if pd.isna(mph) or mph < 0 or mph > 140: continue
        stats.setdefault((key, float(tb)), []).append(float(mph))
    med = {k: float(pd.Series(v).median()) for k,v in stats.items() if len(v)}
    rows = sorted(list({labeler(int(g)) for g in edges_df["gear"].dropna().unique()}),
                  key=lambda s: int(s.split()[0][0]) if s[0].isdigit() else 9)
    tbl = [axis_header()]
    for row_name in rows:
        line = [row_name]
        for x in TPS_AXIS:
            v = med.get((row_name, float(x)), np.nan)
            line.append("" if np.isnan(v) else f"{v:.1f}")
        line.append("%")
        tbl.append(line)
    return tbl, rows

def save_tsv(path, tbl):
    with open(path, "w", encoding="utf-8") as f:
        for row in tbl:
            f.write("\t".join(map(str,row)) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", required=True, help="Folder with __trans_focus__clean__*.csv")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.clean_dir, "__trans_focus__clean__*.csv")))
    if not paths:
        print("[WARN] No clean files found."); return

    all_edges = []
    n = len(paths)
    for i, p in enumerate(paths, 1):
        print(f"({i}/{n}) Processing {os.path.basename(p)} ...")
        ed = collect_edges_from_clean(p)
        if not ed.empty: all_edges.append(ed)

    if not all_edges:
        print("[WARN] No lock/unlock edges found."); return
    E = pd.concat(all_edges, ignore_index=True)

    A = E[E["type"]=="APPLY"]
    R = E[E["type"]=="RELEASE"]

    A_tbl, a_rows = build_table(A, lambda g: f"{ordinal(g)} Apply")
    R_tbl, r_rows = build_table(R, lambda g: f"{ordinal(g)} Release")

    os.makedirs(args.out_dir, exist_ok=True)
    apath = os.path.join(args.out_dir, "TCC_APPLY__Throttle17.tsv")
    rpath = os.path.join(args.out_dir, "TCC_RELEASE__Throttle17.tsv")
    save_tsv(apath, A_tbl)
    save_tsv(rpath, R_tbl)

    with open(os.path.join(args.out_dir,"TCC_TABLES__summary.txt"),"w",encoding="utf-8") as f:
        f.write("TPS axis: " + ", ".join(map(str,TPS_AXIS)) + "\n")
        f.write("Apply rows: " + ", ".join(a_rows) + "\n")
        f.write("Release rows: " + ", ".join(r_rows) + "\n")

    print(f"[OK] Wrote\n  {apath}\n  {rpath}")

if __name__ == "__main__":
    main()
