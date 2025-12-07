import os, pandas as pd, numpy as np

ROOT = r'.\\newlogs\\output'
BASE_SHIFT = os.path.join(ROOT,'01_tables','shift')
BASE_TCC   = os.path.join(ROOT,'01_tables','tcc')
BLENDED    = os.path.join(ROOT,'01_tables','BLENDED_AUDITED')
OUTD       = os.path.join(ROOT,'VALIDATION_FALLBACK'); os.makedirs(OUTD, exist_ok=True)

def load(p): return pd.read_csv(p, sep='\t', dtype=str)
def nd(df):
    X=df.copy()
    for c in X.columns[1:-1]: X[c]=pd.to_numeric(X[c], errors='coerce')
    return X

def diff_one(name, base, blnd):
    b=nd(load(base)); d=nd(load(blnd))
    cols=[c for c in b.columns if c not in ('mph','%')]
    D=[]
    for r in range(len(b)):
        for c in cols:
            v0=b.iloc[r][c]; v1=d.iloc[r][c]
            if pd.isna(v0) and pd.isna(v1): continue
            if v0==v1: continue
            D.append((b.iloc[r]['mph'], c, v1 - v0))
    return pd.DataFrame(D, columns=['row','tps','delta'])

shift_up  = diff_one('SHIFT_UP',
    os.path.join(BASE_SHIFT,'SHIFT_TABLES__UP__Throttle17.tsv'),
    os.path.join(BLENDED,'shift','SHIFT_TABLES__UP__Throttle17.tsv'))
shift_dn  = diff_one('SHIFT_DOWN',
    os.path.join(BASE_SHIFT,'SHIFT_TABLES__DOWN__Throttle17.tsv'),
    os.path.join(BLENDED,'shift','SHIFT_TABLES__DOWN__Throttle17.tsv'))
tcc_app   = diff_one('TCC_APPLY',
    os.path.join(BASE_TCC,'TCC_APPLY__Throttle17.tsv'),
    os.path.join(BLENDED,'tcc','TCC_APPLY__Throttle17.tsv'))
tcc_rel   = diff_one('TCC_RELEASE',
    os.path.join(BASE_TCC,'TCC_RELEASE__Throttle17.tsv'),
    os.path.join(BLENDED,'tcc','TCC_RELEASE__Throttle17.tsv'))

# Top changes
for name, df in (('SHIFT_UP',shift_up),('SHIFT_DOWN',shift_dn),('TCC_APPLY',tcc_app),('TCC_RELEASE',tcc_rel)):
    df.assign(absd=df['delta'].abs()).sort_values('absd',ascending=False).head(20) \
      .to_csv(os.path.join(OUTD, f'DIFF__{name}__TOP20.csv'), index=False)

# Per-row RMS (UP/DOWN only)
def rms_by_row(df, label):
    if df.empty: return
    s=df.groupby('row')['delta'].apply(lambda x: float(np.sqrt(np.mean(x**2)))).reset_index().rename(columns={'delta':'rms'})
    s.to_csv(os.path.join(OUTD, f'DIFF__{label}__RMS_BY_ROW.csv'), index=False)

rms_by_row(shift_up,'SHIFT_UP')
rms_by_row(shift_dn,'SHIFT_DOWN')
print('[WRITE] DIFF summaries ->', OUTD)
