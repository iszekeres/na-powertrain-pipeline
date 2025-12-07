import os, glob, pandas as pd
CLEAN=r'.\\newlogs\\cleaned'

def counts(th):
    cnt={}
    for p in sorted(glob.glob(os.path.join(CLEAN,'__trans_focus__clean_FULL__*.csv'))):
        try:
            df=pd.read_csv(p,usecols=['gear_actual','ectF__canon','tftF__canon'])
        except Exception:
            continue
        ect=pd.to_numeric(df['ectF__canon'],errors='coerce')
        tft=pd.to_numeric(df['tftF__canon'],errors='coerce')
        warm=(ect>=th)&(tft>=th)
        g=pd.to_numeric(df['gear_actual'],errors='coerce')
        prev=g.shift(1)
        ch=(g!=prev)&g.notna()&prev.notna()&warm
        for idx in ch.index[ch]:
            a=int(prev.loc[idx]); b=int(g.loc[idx]); cnt[(a,b)]=cnt.get((a,b),0)+1
    return cnt

a=counts(100); b=counts(95)
keys=sorted(set(a)|set(b))
print('[SENS] warm>=100 vs >=95 -- event counts (100,95,delta):')
for k in keys:
    A=a.get(k,0); B=b.get(k,0)
    print(f'  {k[0]} -> {k[1]} : {A},{B},{B-A}')
