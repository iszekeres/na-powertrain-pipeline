import os, glob, pandas as pd, numpy as np
ROOT=r'.\\newlogs\\output'; CLEAN=r'.\\newlogs\\cleaned'
SHIFT=os.path.join(ROOT,'01_tables','shift'); OUT=os.path.join(ROOT,'VALIDATION')
os.makedirs(OUT, exist_ok=True)
up=pd.read_csv(os.path.join(SHIFT,'SHIFT_TABLES__UP__Throttle17.tsv'),sep='\t',dtype=str)
dn=pd.read_csv(os.path.join(SHIFT,'SHIFT_TABLES__DOWN__Throttle17.tsv'),sep='\t',dtype=str)

def rd(rel):
    path=os.path.join(ROOT,'02_passes',*rel.split('/'))
    return pd.read_csv(path,sep='\t',dtype=str) if os.path.exists(path) else None

def add(base,delta):
    out=base.copy()
    for c in out.columns[1:-1]:
        a=pd.to_numeric(out[c],errors='coerce'); b=pd.to_numeric(delta[c],errors='coerce').fillna(0)
        out[c]=(a+b).astype(float)
    return out

partsU=[rd('CONSIST/CONSIST__SHIFT_UP__DELTA.tsv'), rd('LAT/LAT__SHIFT_UP__DELTA.tsv'), rd('CRUISE_TIPIN/INTENT__SHIFT_UP__DELTA.tsv')]
partsD=[rd('CONSIST/CONSIST__SHIFT_DOWN__DELTA.tsv'), rd('CORNER/CORNER__SHIFT_DOWN__DELTA__COMBINED.tsv'), rd('STOPGO/STOPGO__SHIFT_DOWN__DELTA.tsv'), rd('KICKDOWN/KICKDOWN__SHIFT_DOWN__DELTA.tsv')]
U=up.copy()
for d in partsU:
    if d is not None: U=add(U,d)
D=dn.copy()
for d in partsD:
    if d is not None: D=add(D,d)

TPS=[0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
near=lambda x: min(TPS,key=lambda v:abs(v-float(x))) if pd.notna(x) else None
rows=[]
for p in sorted(glob.glob(os.path.join(CLEAN,'__trans_focus__clean_FULL__*.csv'))):
    try:
        df=pd.read_csv(p,usecols=['gear_actual','speed_mph','throttle_pct'])
    except Exception:
        continue
    g=pd.to_numeric(df['gear_actual'],errors='coerce'); sp=pd.to_numeric(df['speed_mph'],errors='coerce'); tp=pd.to_numeric(df['throttle_pct'],errors='coerce')
    prev=g.shift(1); ch=(g!=prev)&g.notna()&prev.notna()
    idx=np.where(ch.values)[0]
    for i in idx:
        a=int(prev.iloc[i]); b=int(g.iloc[i]); T=near(tp.iloc[i])
        if T is None or pd.isna(sp.iloc[i]): continue
        rows.append((f'{a} -> {b} Shift',T,float(sp.iloc[i])))
if not rows:
    print('[WARP] no events found');
else:
    ev=pd.DataFrame(rows,columns=['row','tps','mph'])
    med=ev.groupby(['row','tps'])['mph'].median().reset_index().rename(columns={'mph':'emp'})
    def melt(df,label):
        m=df.melt(id_vars=['mph','%'],var_name='tps',value_name='pred'); m['row']=m['mph']; m['tps']=pd.to_numeric(m['tps'],errors='coerce'); m['kind']=label
        m['pred']=pd.to_numeric(m['pred'],errors='coerce'); return m[['row','tps','pred','kind']]
    comp=pd.concat([melt(U,'UP'),melt(D,'DOWN')]).merge(med,on=['row','tps'],how='inner')
    comp['err']=comp['pred']-comp['emp']
    def rms(s): return float(np.sqrt(np.mean(s**2))) if len(s) else float('nan')
    def p95(s): return float(np.percentile(np.abs(s),95)) if len(s) else float('nan')
    summ=comp.groupby(['row','kind']).agg(n=('err','count'),rms=('err',rms),p95=('err',p95)).reset_index()
    summ.to_csv(os.path.join(OUT,'SHIFT_WARP__SUMMARY.csv'),index=False)
    print('[WARP] rows:', len(summ))
    print(summ.sort_values('rms').head(10).to_string(index=False))
