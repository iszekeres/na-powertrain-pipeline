#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
REQ=["speed_mph","throttle_pct","gear_actual","time_s","brake","tcc_locked_built","ect_c","tft_c"]
def fail(m): print(f"[ERROR] {m}", file=sys.stderr); sys.exit(2)
def ok(m): print(f"[OK] {m}")
def f_to_c(f): return (f-32.0)*(5.0/9.0)
def load_edi(path):
  df=pd.read_csv(path, sep="\t")
  need=["gear","mph","base_tps","sigma","scale","bias_idle","slip_on","slip_k"]
  miss=[c for c in need if c not in df.columns]
  if miss: fail(f"{path} missing columns {miss}")
  return df
def interp_prof(prof,g,mph):
  p=prof[prof["gear"]==g].sort_values("mph")
  if p.empty: return (np.nan,)*4
  xs=p["mph"].to_numpy()
  base=float(np.interp(mph,xs,p["base_tps"].to_numpy()))
  sig=float(np.interp(mph,xs,p["sigma"].to_numpy()))
  scale=float(np.interp(mph,xs,p["scale"].to_numpy()))
  bias=float(np.interp(mph,xs,p["bias_idle"].to_numpy()))
  return base,sig,scale,bias
def compute_slip(df):
  if "tcc_slip_fused" in df.columns: return pd.to_numeric(df["tcc_slip_fused"], errors="coerce").to_numpy()
  if "engine_rpm" in df.columns and "trans_input_rpm" in df.columns:
    er=pd.to_numeric(df["engine_rpm"], errors="coerce").to_numpy()
    ir=pd.to_numeric(df["trans_input_rpm"], errors="coerce").to_numpy()
    return er-ir
  fail("Slip source missing")
ap=argparse.ArgumentParser()
ap.add_argument("--clean",required=True); ap.add_argument("--edi-file",required=True)
ap.add_argument("--out-dir",required=True); ap.add_argument("--mode",choices=["excess","launch"],required=True)
ap.add_argument("--gate-edi",type=float,default=1.2); ap.add_argument("--gate-dedi",type=float,default=0.8)
ap.add_argument("--launch-baseline",type=str,default=None); ap.add_argument("--launch-accel-min",type=float,default=0.05)
ap.add_argument("--warm",type=float,default=100.0)
a=ap.parse_args(); out=Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
edi=pd.read_csv(a.edi_file, sep="\t"); warm_c=f_to_c(a.warm)
paths=sorted(Path(a.clean).glob("*.csv"))
if not paths: fail(f"No CSVs in {a.clean}")
for p in paths:
  df=pd.read_csv(p, low_memory=False)
  miss=[c for c in REQ if c not in df.columns]
  if miss: fail(f"{p.name} missing {miss}")
  for c in ["speed_mph","throttle_pct","time_s","gear_actual","brake","tcc_locked_built","ect_c","tft_c"]:
    df[c]=pd.to_numeric(df[c], errors="coerce")
  df=df[(df["ect_c"]>=warm_c)&(df["tft_c"]>=warm_c)].copy().sort_values("time_s")
  dv=df["speed_mph"].diff().to_numpy(); dt=df["time_s"].diff().to_numpy()
  with np.errstate(divide='ignore', invalid='ignore'): dvdt=np.where(dt>0, dv/dt, 0.0)
  df["dvdt"]=dvdt
  # Simple EDI proxy: base TPS profile not re-evaluated here (we rely on edi_builder's outputs)
  df.to_csv(Path(a.out_dir)/p.name, index=False); ok(f"Wrote {(Path(a.out_dir)/p.name)}")
