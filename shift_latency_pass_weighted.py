#!/usr/bin/env python3
import argparse, glob, os, math, csv
import pandas as pd
import numpy as np
from weight_utils import combined_weight
from tps_phases import is_cruise_or_mild
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
    ap.add_argument('--out-prefix',default=r'.\LAT')
    ap.add_argument('--half-life-days',type=float,default=30.0)
    ap.add_argument('--route-bias',default='neighborhood=1.5,inbound=1.2,outbound=1.2,highway=1.1')
    args=ap.parse_args(); route_map=dict(kv.split('=') for kv in args.route_bias.split(',') if '=' in kv)
    need_core=['gear_actual__canon','gear_cmd__canon','throttle_pct__canon','speed_mph__canon']
    frames=[]
    for p in sorted(glob.glob(args.logs_glob)):
        try:
            df=pd.read_csv(p,low_memory=False)
        except Exception:
            continue
        missing=[c for c in need_core if c not in df.columns]
        if missing:
            raise RuntimeError(f"LAT: required columns {missing} missing in CLEAN_FULL: {os.path.basename(p)}")
        df_sub=pd.DataFrame({
            'gear_actual': pd.to_numeric(df['gear_actual__canon'],errors='coerce'),
            'gear_cmd':    pd.to_numeric(df['gear_cmd__canon'],errors='coerce'),
            'throttle_pct': pd.to_numeric(df['throttle_pct__canon'],errors='coerce'),
            'speed_mph':   pd.to_numeric(df['speed_mph__canon'],errors='coerce'),
            '__file':      os.path.basename(p),
        })
        frames.append(df_sub)
    if not frames:
        print('[LAT_W] No data'); return
    d=pd.concat(frames,ignore_index=True)
    d=d.dropna(subset=['gear_actual','gear_cmd','throttle_pct'])
    # TPS phase gating: focus on cruise+mild (8–25%)
    d = d[d['throttle_pct'].apply(is_cruise_or_mild)].copy()
    if d.empty:
        print('[LAT_W] No data in cruise/mild TPS band'); return
    d['mismatch']=(d['gear_cmd']!=d['gear_actual']).astype(float)
    d['w']=[combined_weight(fn,spd,args.half_life_days,route_map) for fn,spd in zip(d['__file'],d['speed_mph'].fillna(0))]
    score=(d['mismatch']*d['w']).sum()
    up=empty_shift('up')
    if score>300:
        for g in [2,3,4,5]:
            for t in [31,37,44,50,56]:
                i=tps_idx(t); up[f'{g} -> {g+1} Shift'][i]=(-0.3) if math.isnan(up[f'{g} -> {g+1} Shift'][i]) else (up[f'{g} -> {g+1} Shift'][i]-0.3)
    out=args.out_prefix.rstrip('\\/'); write_delta(f'{out}__SHIFT_UP__DELTA.tsv',up); print('[LAT_W] WROTE',f'{out}__SHIFT_UP__DELTA.tsv')
if __name__=='__main__': main()
