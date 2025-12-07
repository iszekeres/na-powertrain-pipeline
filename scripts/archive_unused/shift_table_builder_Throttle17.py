# shift_table_builder_Throttle17.py — STRICT RAW (CLEAN_FULL)
# RAW headers only: Offset, Vehicle Speed (SAE), Throttle Position, gear_actual
# De-dups in favor of __withbrake; ±3% TPS band; neutral (no bias); 0.1 mph.

import os, sys, argparse
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

# TPS sampling window center bias (percent)
TPS_BIAS = 1.0

RAW_REQ   = ["Offset","Vehicle Speed (SAE)","Throttle Position","gear_actual"]
TPS_AXIS  = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]

def r1(x): 
    if pd.isna(x): return x
    return float(Decimal(str(x)).quantize(Decimal("0.0"), rounding=ROUND_HALF_UP))

def find_shift_events(df):
    g   = pd.to_numeric(df["gear_actual"], errors="coerce")
    mph = pd.to_numeric(df["Vehicle Speed (SAE)"], errors="coerce")
    tps = pd.to_numeric(df["Throttle Position"],   errors="coerce")
    d = g.diff(); idx = d.index
    rows = []
    for i in idx[(d== 1)]:
        if i==0 or pd.isna(g.iat[i-1]) or pd.isna(g.iat[i]): continue
        rows.append((int(g.iat[i-1]), int(g.iat[i]), tps.iat[i], mph.iat[i]))
    for i in idx[(d==-1)]:
        if i==0 or pd.isna(g.iat[i-1]) or pd.isna(g.iat[i]): continue
        rows.append((int(g.iat[i-1]), int(g.iat[i]), tps.iat[i], mph.iat[i]))
    if not rows: 
        return pd.DataFrame(columns=["from","to","tps","mph"])
    ev = pd.DataFrame(rows, columns=["from","to","tps","mph"]).dropna()
    ev["tps"]=pd.to_numeric(ev["tps"],errors="coerce"); ev["mph"]=pd.to_numeric(ev["mph"],errors="coerce")
    return ev.dropna().reset_index(drop=True)

def build_tables(events, band=5.0):
    def row_for(f,t):
        vals=[]
        pair=(events["from"]==f)&(events["to"]==t)
        for tp in TPS_AXIS:
            sel = pair & events["tps"].between(tp-band, tp+band)
            mphs= events.loc[sel,"mph"]
            vals.append(r1(mphs.median()) if not mphs.empty else np.nan)
        return (f"{f} -> {t} Shift", vals)
    up   = [row_for(g,g+1) for g in (1,2,3,4,5)]
    down = [row_for(g,g-1) for g in (6,5,4,3,2)]
    return up, down

def write_table(rows, out_path):
    hdr="mph\t"+"\t".join(map(str,TPS_AXIS))+"\t%"
    with open(out_path,"w",newline="") as f:
        f.write(hdr+"\n")
        for label,vals in rows:
            f.write(label+"\t"+"\t".join("" if pd.isna(v) else str(r1(v)) for v in vals)+"\t\n")

def main(clean_dir,out_dir):
    os.makedirs(out_dir,exist_ok=True)
    csvs=[os.path.join(clean_dir,f) for f in os.listdir(clean_dir) if f.lower().endswith(".csv")]
    if not csvs: raise SystemExit("No CSV files found")
    cf=[c for c in csvs if "clean_full" in os.path.basename(c).lower()]
    for k in (set(csvs)-set(cf)): sys.stderr.write("[shift_builder] skipping non-CLEAN_FULL: "+os.path.basename(k)+"\n")
    if not cf: raise SystemExit("No CLEAN_FULL CSVs found")
    # de-dup: prefer __withbrake
    bykey={}
    for c in cf:
        key=os.path.basename(c).lower().replace("__withbrake","")
        bykey.setdefault(key,[]).append(c)
    chosen=[]; skipped=[]
    for key,files in bykey.items():
        if len(files)==1: chosen.append(files[0])
        else:
            wb=[f for f in files if "__withbrake" in os.path.basename(f).lower()]
            pick=wb[0] if wb else files[0]
            chosen.append(pick); skipped += [f for f in files if f!=pick]
    for k in skipped: sys.stderr.write("[shift_builder] de-dup: skipping "+os.path.basename(k)+" (duplicate)\n")

    events=[]
    for p in chosen:
        hdr=pd.read_csv(p,nrows=0).columns.tolist()
        miss=[h for h in RAW_REQ if h not in hdr]
        if miss: raise SystemExit("Required RAW headers missing in "+os.path.basename(p)+": "+", ".join(miss))
        sys.stderr.write("[shift_builder] processing "+os.path.basename(p)+"\n")
        df=pd.read_csv(p,usecols=RAW_REQ,low_memory=False).sort_values("Offset",kind="mergesort").reset_index(drop=True)
        ev=find_shift_events(df)
        if not ev.empty:
            cts=ev.groupby(["from","to"]).size()
            pairs=", ".join([f"{a}->{b}:{int(n)}" for (a,b),n in sorted(cts.items())])
        else:
            pairs="none"
        sys.stderr.write("[shift_builder]   rows="+format(len(df),",")+"  edges: "+pairs+"\n")
        events.append(ev)
    if not any([not e.empty for e in events]): raise SystemExit("No shift edges found across inputs")
    E=pd.concat(events,ignore_index=True)
    up,down=build_tables(E,band=5.0)
    up_path=os.path.join(out_dir,"SHIFT_TABLES__UP__Throttle17.tsv")
    dn_path=os.path.join(out_dir,"SHIFT_TABLES__DOWN__Throttle17.tsv")
    write_table(up,up_path); write_table(down,dn_path)
    print("Wrote:",up_path); print("Wrote:",dn_path); sys.stderr.write("[shift_builder] done. outputs written\n")

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--clean-dir",required=True); a.add_argument("--out-dir",required=True)
    args=a.parse_args(); main(args.clean_dir,args.out_dir)

