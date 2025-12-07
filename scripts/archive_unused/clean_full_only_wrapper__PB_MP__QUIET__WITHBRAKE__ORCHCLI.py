#!/usr/bin/env python3
import argparse, sys, subprocess
from pathlib import Path
import pandas as pd, numpy as np
PREF_SOURCES=[("Brake Pressure","kpa"),("Brake Pressure (kPa)","kpa"),("Brake_Pressure","kpa"),("brake_pressure","kpa"),
("Brake_Pressure_kPa","kpa"),("Brake_Pressure_kpa","kpa"),("brake","binary"),("Brake","binary"),
("Brake Switch","binary"),("Brake_Switch","binary"),("Brake_On","binary"),("BrakeSwitch","binary")]
CORE_REQ=["speed_mph","throttle_pct","gear_actual","time_s"]
def fail(m): print(f"[ERROR] {m}", file=sys.stderr); sys.exit(2)
def ok(m): print(f"[OK] {m}")
def find_col(df,name):
  lower={c.lower():c for c in df.columns}
  cands=[name,name.lower(),name.replace(" ","_").lower(),name.replace(" ","").lower()]
  for k in cands:
    if k in lower: return lower[k]
  tgt=name.lower().replace(" ","")
  for k,v in lower.items():
    if tgt in k.replace("_",""): return v
  return None
def inject_brake(cleaned_dir, threshold_kpa):
  for p in sorted(Path(cleaned_dir).glob("*.csv")):
    df=pd.read_csv(p, low_memory=False)
    miss=[c for c in CORE_REQ if c not in df.columns]
    if miss: fail(f"{p.name}: missing core columns {miss}")
    if "brake" in df.columns: ok(f"{p.name}: brake present (no change)"); continue
    src_col=src_mode=None; tried=[]
    for cand,mode in PREF_SOURCES:
      tried.append(cand); col=find_col(df,cand)
      if col is not None: src_col,src_mode=col,mode; break
    if not src_col: fail(f"{p.name}: no brake source found. Looked for any of {tried}")
    if src_mode=="kpa":
      vals=pd.to_numeric(df[src_col], errors="coerce")
      brake=(vals>=threshold_kpa).astype("Int64").fillna(0).astype(int)
    else:
      vals=df[src_col].astype(str).str.strip().str.lower()
      brake=vals.isin(["1","true","on","yes"]).astype(int)
    df["brake"]=brake; df.to_csv(p, index=False); ok(f"{p.name}: added brake from '{src_col}' [{src_mode}] ≥ {threshold_kpa} kPa => 1")
ap=argparse.ArgumentParser()
ap.add_argument("--raw-glob",required=True); ap.add_argument("--cleaner",required=True)
ap.add_argument("--staging",required=True); ap.add_argument("--cleaned-dir",required=True)
ap.add_argument("--out-root",required=True); ap.add_argument("--workers",required=True)
ap.add_argument("--threshold-kpa",type=float,default=15.0)
ap.add_argument("--underlying",default=r".\clean_full_only_wrapper__PB_MP__QUIET.py")
args,unknown=ap.parse_known_args()
under=Path(args.underlying)
if not under.exists(): fail(f"Underlying wrapper not found: {under}")
cmd=[sys.executable,str(under),"--raw-glob",args.raw_glob,"--cleaner",args.cleaner,"--staging",args.staging,"--cleaned-dir",args.cleaned_dir,"--out-root",args.out_root,"--workers",str(args.workers)]+unknown
print("[RUN] "+" ".join(cmd)); rc=subprocess.run(cmd).returncode
if rc!=0: fail(f"Underlying wrapper returned nonzero exit {rc}")
inject_brake(args.cleaned_dir, args.threshold_kpa); ok("Brake injection complete. CLEAN_FULL is ready.")
