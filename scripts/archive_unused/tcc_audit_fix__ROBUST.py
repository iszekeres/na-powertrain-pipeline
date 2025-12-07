import os,glob,pandas as pd, numpy as np, argparse
TPS = [str(x) for x in [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]]
def load_tsv(path):
    try:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1]==1: df = pd.read_csv(path, sep=",")
        return df
    except Exception:
        return pd.read_csv(path, sep=",")
def norm(df):
    cols = df.columns.tolist()
    if cols[0].strip().lower()!="mph":
        # assume first row is header line
        df = pd.read_csv(pd.compat.StringIO(df.to_csv(index=False, header=False)), header=None, sep="\t")
    # Ensure canonical header
    want = ["mph"]+TPS
    if df.shape[1]==len(want)+1 and str(df.columns[-1]).strip()=="%":
        df.columns = want+["%"]
    elif df.shape[1]==len(want):
        df.columns = want
    return df
def snap(v):
    try:
        if pd.isna(v): return v
        fv = float(v)
        if fv in (317.0,318.0): return fv
        return round(fv,1)
    except: return v
def enforce_gap(apply_df, release_df):
    for r in range(apply_df.shape[0]):
        row_name = str(apply_df.iloc[r,0]).lower()
        if "apply" in row_name:
            rel_row = release_df.iloc[r]
            ap_row  = apply_df.iloc[r]
            for c in apply_df.columns[1:]:
                try:
                    ap = float(ap_row[c]); rl = float(rel_row[c])
                    if ap in (317.0,318.0) or rl in (317.0,318.0): continue
                    if rl < ap + 1.1: release_df.at[r,c] = ap + 1.1
                except: pass
    # 1dp rounding
    for df in (apply_df, release_df):
        for c in df.columns[1:]:
            df[c] = df[c].map(snap)
    return apply_df, release_df

ap_in = r"newlogs/output/01_tables/tcc/TCC_APPLY__Throttle17.tsv"
rl_in = r"newlogs/output/01_tables/tcc/TCC_RELEASE__Throttle17.tsv"
ap = norm(load_tsv(ap_in))
rl = norm(load_tsv(rl_in))
ap, rl = enforce_gap(ap, rl)
outdir = r"newlogs/output/01_tables/NEUTRAL_CANDIDATE"
os.makedirs(outdir, exist_ok=True)
ap.to_csv(os.path.join(outdir,"TCC_APPLY__Throttle17.tsv"), sep="\t", index=False)
rl.to_csv(os.path.join(outdir,"TCC_RELEASE__Throttle17.tsv"), sep="\t", index=False)
print("[TCC_ROBUST] wrote to NEUTRAL_CANDIDATE")
