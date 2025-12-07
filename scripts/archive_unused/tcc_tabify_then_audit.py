# -*- coding: utf-8 -*-
import os, sys, pandas as pd, numpy as np, argparse

TPS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]

def read_any(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        hdr = f.readline()
    sep = "\t" if ("\t" in hdr) else ","
    return pd.read_csv(path, sep=sep), sep

def canon(df):
    has_pct = (str(df.columns[-1]).strip() == "%")
    cols = ["mph"] + list(map(str, TPS)) + (["%"] if has_pct else [])
    if df.shape[1] == 1:
        raise SystemExit("tcc_audit: input not parsed into columns; wrong delimiter?")
    if len(cols) == df.shape[1]:
        df.columns = cols
        return df
    have = df.shape[1]
    take = min(have-1, len(TPS))
    core = ["mph"] + list(df.columns[1:1+take])
    if has_pct and (have >= (2+take)): core += [df.columns[-1]]
    df = df[core]
    df.columns = ["mph"] + list(map(str, TPS[:take])) + (["%"] if has_pct and len(core)==(2+take) else [])
    # pad missing TPS cols with NaN
    for c in map(str, TPS):
        if c not in df.columns and c != "%":
            df[c] = np.nan
    # reorder to mph + full TPS (+ % if present)
    out_cols = ["mph"] + list(map(str, TPS)) + (["%"] if has_pct else [])
    df = df[out_cols]
    return df

def snap(x):
    if pd.isna(x): return x
    try: xv=float(x)
    except: return x
    if abs(xv-317.0)<0.05 or abs(xv-318.0)<0.05: return int(round(xv))
    return round(xv,1)

def enforce_gap(ap, re_):
    num = list(map(str, TPS))
    for g in ("3rd","4th","5th","6th"):
        ra = ap["mph"].astype(str).str.contains(g) & ap["mph"].astype(str).str.contains("Apply")
        rr = re_["mph"].astype(str).str.contains(g) & re_["mph"].astype(str).str.contains("Release")
        if not ra.any() or not rr.any(): continue
        ia = int(np.where(ra)[0][0]); ir = int(np.where(rr)[0][0])
        for c in num:
            va = ap.at[ia,c]; vr = re_.at[ir,c]
            if pd.isna(va) or pd.isna(vr): continue
            if (va in (317,318)) or (vr in (317,318)): continue
            try:
                if float(vr) < float(va) + 1.1:
                    re_.at[ir,c] = round(float(va)+1.1, 1)
            except: pass
    return ap, re_

def main(tcc_dir, out_dir):
    ap_in = os.path.join(tcc_dir, "TCC_APPLY__Throttle17.tsv")
    re_in = os.path.join(tcc_dir, "TCC_RELEASE__Throttle17.tsv")
    if not (os.path.isfile(ap_in) and os.path.isfile(re_in)):
        raise SystemExit("tcc_audit: missing TCC pair in "+tcc_dir)

    ap, sa = read_any(ap_in); re, sr = read_any(re_in)
    if (sa != "\t"): ap.to_csv(ap_in, sep="\t", index=False)
    if (sr != "\t"): re.to_csv(re_in, sep="\t", index=False)

    ap = canon(ap); re = canon(re)
    num = list(map(str, TPS))
    ap[num] = ap[num].applymap(snap)
    re[num] = re[num].applymap(snap)
    ap, re = enforce_gap(ap, re)

    os.makedirs(out_dir, exist_ok=True)
    ap_out = os.path.join(out_dir, "TCC_APPLY__Throttle17.tsv")
    re_out = os.path.join(out_dir, "TCC_RELEASE__Throttle17.tsv")
    ap.to_csv(ap_out, sep="\t", index=False)
    re.to_csv(re_out, sep="\t", index=False)
    print("[TCC-AUDIT] wrote", ap_out, "and", re_out)

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--tcc-dir", required=True)
    a.add_argument("--out-dir", required=True)
    ns = a.parse_args()
    main(ns.tcc_dir, ns.out_dir)
