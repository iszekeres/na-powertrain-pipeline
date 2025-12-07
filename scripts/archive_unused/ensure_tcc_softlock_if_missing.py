#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import pandas as pd, numpy as np
ON_RPM, OFF_RPM, ON_SEC, OFF_SEC = 30.0, 80.0, 0.6, 0.4
def f_to_c(f): return (f-32.0)*(5.0/9.0)
def pick(df, names):
  lower={c.lower():c for c in df.columns}
  for n in names:
    c=lower.get(n.lower())
    if c: return c
  for n in names:
    tgt=n.lower().replace(' ','').replace('_','')
    for k,v in lower.items():
      if tgt in k.replace(' ','').replace('_',''): return v
  return None
def slip_from(df):
  if "tcc_slip_fused" in df.columns: return pd.to_numeric(df["tcc_slip_fused"], errors="coerce").to_numpy()
  ic=pick(df,["trans_input_rpm","turbine_rpm","Trans Input Shaft RPM"]); ec=pick(df,["engine_rpm","Engine RPM (SAE)"])
  if ic and ec:
    ir=pd.to_numeric(df[ic], errors="coerce").to_numpy()
    er=pd.to_numeric(df[ec], errors="coerce").to_numpy()
    return er-ir
  raise ValueError("no slip source")
def build_lock(t, slip, v, g, br, ect, tft):
  n=len(t); locked=np.zeros(n, dtype=np.int8); on=off=0.0; last=t[0] if n else 0.0
  warm=(ect>=f_to_c(100)) & (tft>=f_to_c(100)); gate=warm & (v>=25.0) & (g>=3) & (br==0)
  for i in range(n):
    ti=float(t[i]) if np.isfinite(t[i]) else last; dt=max(0.0, ti-last) if i>0 else 0.0; last=ti
    if not np.isfinite(slip[i]) or not gate[i]:
      on=max(0.0, on-dt); off=max(0.0, off-dt); locked[i]=locked[i-1] if i>0 else 0; continue
    if slip[i] <= ON_RPM: on += dt; off = max(0.0, off-dt)
    elif slip[i] >= OFF_RPM: off += dt; on = max(0.0, on-dt)
    else: on=max(0.0,on-0.25*dt); off=max(0.0,off-0.25*dt)
    if (locked[i-1] if i>0 else 0)==1:
      if off>=OFF_SEC: locked[i]=0; off=0.0
      else: locked[i]=1
    else:
      if on>=ON_SEC: locked[i]=1; on=0.0
      else: locked[i]=0
  return locked.astype(int)
def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--cleaned-dir", required=True); a=ap.parse_args()
  base=Path(a.cleaned_dir)
  for p in sorted(base.glob("*.csv")):
    df=pd.read_csv(p, low_memory=False)
    if "tcc_locked_built" in df.columns: print(f"[OK] {p.name}: present"); continue
    need=["time_s","speed_mph","gear_actual","brake","ect_c","tft_c"]
    miss=[c for c in need if c not in df.columns]
    if miss: print(f"[SKIP] {p.name}: missing {miss}"); continue
    t=pd.to_numeric(df["time_s"], errors="coerce").to_numpy()
    v=pd.to_numeric(df["speed_mph"], errors="coerce").to_numpy()
    g=pd.to_numeric(df["gear_actual"], errors="coerce").to_numpy()
    b=pd.to_numeric(df["brake"], errors="coerce").fillna(0).to_numpy()
    ect=pd.to_numeric(df["ect_c"], errors="coerce").to_numpy()
    tft=pd.to_numeric(df["tft_c"], errors="coerce").to_numpy()
    s=slip_from(df)
    lock=build_lock(t,s,v,g,b,ect,tft)
    df["tcc_locked_built"]=lock; df.to_csv(p, index=False); print(f"[FIX] {p.name}: added tcc_locked_built")
if __name__=="__main__": main()
