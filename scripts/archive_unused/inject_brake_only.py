#!/usr/bin/env python3
# inject_brake_only.py — add 'brake' from brake pressure (kPa) or boolean brake switch if missing.
import argparse, sys
from pathlib import Path
import pandas as pd
SOURCES = ["Brake Pressure (kPa)","Brake Pressure","Brake Pressure (SAE)","brake","Brake","Brake Switch"]
ap=argparse.ArgumentParser(); ap.add_argument("--cleaned-dir",required=True); a=ap.parse_args()
base=Path(a.cleaned_dir)
for p in sorted(base.glob("*.csv")):
    df=pd.read_csv(p, low_memory=False)
    if "brake" in df.columns:
        print(f"[OK] {p.name}: brake present"); continue
    src=None
    for s in SOURCES:
        if s in df.columns: src=s; break
    if not src:
        print(f"[WARN] {p.name}: no brake source found"); continue
    if "Pressure" in src:
        vals=pd.to_numeric(df[src], errors="coerce")
        df["brake"]=(vals>=15).astype(int)
    else:
        v=df[src].astype(str).str.strip().str.lower()
        df["brake"]=v.isin(["1","true","on","yes"]).astype(int)
    df.to_csv(p, index=False); print(f"[FIX] {p.name}: added brake from '{src}'")
