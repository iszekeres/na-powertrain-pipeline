#!/usr/bin/env python3
import argparse, glob, os, math, csv
import pandas as pd
import numpy as np
from weight_utils import combined_weight
TPS=[0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
def empty_shift(kind):
    labs=(["1 -> 2 Shift","2 -> 3 Shift","3 -> 4 Shift","4 -> 5 Shift","5 -> 6 Shift"] if kind=="up" else
          ["2 -> 1 Shift","3 -> 2 Shift","4 -> 3 Shift","5 -> 4 Shift","6 -> 5 Shift"])
    return {lab:[np.nan]*17 for lab in labs}
def empty_tcc(kind):
    labs=(["1st Apply","2nd Apply","3rd Apply","4th Apply","5th Apply","6th Apply"] if kind=="apply" else
          ["1st Release","2nd Release","3rd Release","4th Release","5th Release","6th Release"])
    return {lab:[np.nan]*17 for lab in labs}
def tps_idx(val):
    return min(range(17), key=lambda i: abs(TPS[i]-float(val)))
def write_delta(path, body):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.writer(f, delimiter="\t")
        w.writerow(["mph"]+[str(x) for x in TPS]+["%"])
        for lab in body:
            row=[lab]
            for v in body[lab]:
                if v is None or (isinstance(v,float) and (math.isnan(v))): row.append("")
                else: row.append(f"{float(v):.1f}")
            row.append("")
            w.writerow(row)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--logs-glob',default=r'.\06_Logs\Trans_Review\__trans_focus__clean__*.csv')
    ap.add_argument('--out-prefix',default=r'.\CORNER')
    ap.add_argument('--half-life-days',type=float,default=30.0)
    ap.add_argument('--route-bias',default='neighborhood=1.5,inbound=1.2,outbound=1.2,highway=1.1')
    args=ap.parse_args(); route_map=dict(kv.split('=') for kv in args.route_bias.split(',') if '=' in kv)
    need=['speed_mph','throttle_pct','gear_actual','Lateral Acceleration','Yaw Rate','__file']; frames=[]
    for p in sorted(glob.glob(args.logs_glob)):
        try: df=pd.read_csv(p,low_memory=False)
        except: continue
        for c in need:
            if c not in df.columns: df[c]=pd.NA
        df['__file']=df['__file'].fillna(os.path.basename(p)); frames.append(df[need].copy())
    if not frames: print('[CORNER_W] No data'); return
    d=pd.concat(frames,ignore_index=True)
    for c in ['speed_mph','throttle_pct','gear_actual','Lateral Acceleration','Yaw Rate']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['speed_mph','throttle_pct','gear_actual'])
    corner=(d['Lateral Acceleration'].abs()>0.12)&(d['speed_mph'].between(12,45))
    d['w']=[combined_weight(fn,spd,args.half_life_days,route_map) for fn,spd in zip(d['__file'],d['speed_mph'])]
    dn=empty_shift('down')
    for g,label in [(3,'3 -> 2 Shift'),(4,'4 -> 3 Shift')]:
        seg=d[corner&(d['gear_actual']==g)]; score=seg['w'].sum()
        if score>180:
            for t in [31,37,44]:
                i=tps_idx(t); dn[label][i]=(+0.6) if math.isnan(dn[label][i]) else (dn[label][i]+0.6)
    out=args.out_prefix.rstrip('\\/'); write_delta(f'{out}__SHIFT_DOWN__DELTA.tsv',dn); print('[CORNER_W] WROTE',f'{out}__SHIFT_DOWN__DELTA.tsv')
if __name__=='__main__': main()
