#!/usr/bin/env python3
import argparse, glob, os, math, csv
import pandas as pd
import numpy as np
from weight_utils import combined_weight

# === TAHOE INTENT GATING SPEC ===
# 1. SPEED WINDOW (MUCH WIDER)
INTENT_SPEED_MIN = 18.0
INTENT_SPEED_MAX = 60.0

# 2. TPS WINDOW (MORE REALISTIC)
INTENT_TPS_MIN = 10.0
INTENT_TPS_MAX = 50.0

# 3. TPS RATE REQUIREMENTS (DRASTICALLY LOWER)
INTENT_DTPS_MIN = 1.5
INTENT_DTPS_MAX = 40.0

# 4. PEDAL RATE REQUIREMENTS (SMOOTH INTENT)
INTENT_DPEDAL_MIN = 0.8
INTENT_DPEDAL_MAX = 35.0

# 5. BRAKE FILTER (RELAXED)
INTENT_BRAKE_MAX = 25.0

# 6. GEAR REQUIREMENTS (TAHOE)
# Use snapped gear_int and accept any stable segment;
# no extra adjacency or oncoming-clutch requirements.
INTENT_REQUIRE_ADJACENT_SHIFTS = False
INTENT_REQUIRE_ONCOMING_CLUTCH = False

# 7. TIME WINDOW FOR MASK (MUCH WIDER)
INTENT_MIN_DURATION = 0.08
INTENT_MAX_DURATION = 2.0

# 9. TCC RELEASE MIN SPEED (for heavy Tahoe)
INTENT_TCC_RELEASE_MINSPEED = 22.0
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
AXIS_17=[0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
SHIFT_LABELS={i:f"{i} -> {i+1} Shift" for i in range(1,6)}
TCC_RELEASE_LABELS={3:"3rd Release",4:"4th Release",5:"5th Release",6:"6th Release"}

def _bucket_tps_to_17pt_axis(tps):
    if pd.isna(tps):
        return 0
    v=float(tps)
    v=max(0.0,min(100.0,v))
    return min(AXIS_17,key=lambda a: abs(a-v))

def gather_intent_events(df, mask):
    hits=df.loc[mask]
    records=[]
    for _,row in hits.iterrows():
        # Prefer stabilized integer gear if present
        gear = row.get("gear_int", row.get("gear_actual", np.nan))
        if not pd.notna(gear):
            continue
        try:
            gear_id=int(float(gear))
        except Exception:
            continue
        if gear_id<1 or gear_id>6:
            continue
        mph=row["speed_mph"]
        tps=row["throttle_pct"]
        if not (pd.notna(mph) and pd.notna(tps)):
            continue
        # Additional guard for TCC release events: respect a slightly higher min speed
        if float(mph) < INTENT_TCC_RELEASE_MINSPEED:
            # Still allow UP events at lower speeds; only restrict TCC rows
            label_release=TCC_RELEASE_LABELS.get(gear_id)
            if label_release:
                continue
        label_row=SHIFT_LABELS.get(gear_id)
        if label_row and 1<=gear_id<=5:
            records.append({"table":"UP","row":label_row,"mph":float(mph),"tps":float(tps)})
        label_release=TCC_RELEASE_LABELS.get(gear_id)
        if label_release and 3<=gear_id<=6:
            records.append({"table":"TCC","row":label_release,"mph":float(mph),"tps":float(tps)})
    if not records:
        return pd.DataFrame(columns=["table","row","mph","tps"])
    return pd.DataFrame.from_records(records,columns=["table","row","mph","tps"])

def write_intent_summary(events_df,out_prefix):
    summary_path=f"{out_prefix}DEBUG_SUMMARY.csv"
    cols=["table","row","tps_bin","count","median_mph","std_mph"]
    if events_df is None or events_df.empty:
        pd.DataFrame(columns=cols).to_csv(summary_path,index=False)
        return
    df=events_df.copy()
    required={"table","row","mph","tps"}
    missing=required.difference(df.columns)
    if missing:
        pd.DataFrame(columns=cols).to_csv(summary_path,index=False)
        return
    df["tps_bin"]=df["tps"].map(_bucket_tps_to_17pt_axis)
    rows=[]
    for (table,row,tps_bin),group in df.groupby(["table","row","tps_bin"]):
        count=int(len(group))
        if count==0:
            continue
        median_mph=float(np.median(group["mph"]))
        std_mph=float(np.nan_to_num(np.std(group["mph"], ddof=0)))
        rows.append({"table":table,"row":row,"tps_bin":int(tps_bin),"count":count,"median_mph":median_mph,"std_mph":std_mph})
    summary=pd.DataFrame(rows,columns=cols)
    summary.to_csv(summary_path,index=False)

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
    ap.add_argument('--out-prefix',default=r'.\INTENT')
    ap.add_argument('--half-life-days',type=float,default=30.0)
    ap.add_argument('--route-bias',default='neighborhood=1.5,inbound=1.2,outbound=1.2,highway=1.1')
    args=ap.parse_args(); route_map=dict(kv.split('=') for kv in args.route_bias.split(',') if '=' in kv)
    need=['time_s','speed_mph','throttle_pct','pedal_pct','brake','gear_actual','__file']; frames=[]
    for p in sorted(glob.glob(args.logs_glob)):
        try: df=pd.read_csv(p,low_memory=False)
        except: continue
        for c in need:
            if c not in df.columns: df[c]=pd.NA
        df['__file']=df['__file'].fillna(os.path.basename(p)); frames.append(df[need].copy())
    if not frames: print('[INTENT_W] No data'); return
    d=pd.concat(frames,ignore_index=True)
    for c in ['speed_mph','throttle_pct','gear_actual','time_s','pedal_pct','brake']:
        if c in d.columns:
            d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['speed_mph','throttle_pct','gear_actual'])
    # stabilized integer gear for event labeling
    d['gear_int'] = np.round(d['gear_actual']).astype('Int64')
    # time delta and rates
    d['dt']=d['time_s'].diff().fillna(0.01).clip(lower=1e-3)
    d['dTPS']=(d['throttle_pct'].diff()/d['dt']).abs().clip(0,200)
    if 'pedal_pct' in d.columns:
        d['dPED']=(d['pedal_pct'].diff()/d['dt']).abs().clip(0,200)
    else:
        d['dPED']=np.nan
    d['w']=[combined_weight(fn,spd,args.half_life_days,route_map) for fn,spd in zip(d['__file'],d['speed_mph'])]
    # relaxed Tahoe intent mask
    spd_ok = d['speed_mph'].between(INTENT_SPEED_MIN, INTENT_SPEED_MAX)
    tps_ok = d['throttle_pct'].between(INTENT_TPS_MIN, INTENT_TPS_MAX)
    dtps_ok= d['dTPS'].between(INTENT_DTPS_MIN, INTENT_DTPS_MAX)
    dped_ok= d['dPED'].between(INTENT_DPEDAL_MIN, INTENT_DPEDAL_MAX) | d['dPED'].isna()
    brk_ok = (~d['brake'].notna()) | (d['brake']<=INTENT_BRAKE_MAX)
    base_sel = spd_ok & tps_ok & dtps_ok & dped_ok & brk_ok
    # duration gate across contiguous True segments
    grp = (base_sel != base_sel.shift()).cumsum()
    seg_dur = d['dt'].groupby(grp).transform('sum')
    mask = base_sel & (seg_dur >= INTENT_MIN_DURATION) & (seg_dur <= INTENT_MAX_DURATION)
    up=empty_shift('up'); rl=empty_tcc('release')
    # Use stabilized integer gear for Tahoe intent heuristics
    for g,label in [(4,'4 -> 5 Shift'),(5,'5 -> 6 Shift')]:
        seg=d[mask & (d['gear_int'] == g)]; score=seg['w'].sum()
        if score>250:
            for t in [37,44,50,56]:
                i=tps_idx(t); up[label][i]=(-0.4) if math.isnan(up[label][i]) else (up[label][i]-0.4)
                rl[f'{g}th Release'][i]=(+0.2) if math.isnan(rl[f'{g}th Release'][i]) else (rl[f'{g}th Release'][i]+0.2)
    out=args.out_prefix.rstrip('\\/')
    up_path=f'{out}__SHIFT_UP__DELTA.tsv'
    rl_path=f'{out}__TCC_RELEASE__DELTA.tsv'
    write_delta(up_path,up)
    write_delta(rl_path,rl)
    events_df=gather_intent_events(d,mask)
    write_intent_summary(events_df, out)
    summary_path=f'{out}DEBUG_SUMMARY.csv'
    print('[INTENT_W] WROTE', up_path, rl_path, summary_path)
if __name__=='__main__': main()
