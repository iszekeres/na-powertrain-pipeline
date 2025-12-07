import os, glob, numpy as np, pandas as pd, sys

ROOT  = r'.\newlogs\output'
CLEAN = r'.\newlogs\cleaned'
SHIFT = os.path.join(ROOT,'01_tables','shift')
OUT   = os.path.join(ROOT,'VALIDATION_FALLBACK'); os.makedirs(OUT, exist_ok=True)
TPS   = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]

def pick(df, prefer=(), contains=()):
    cols=list(df.columns); low={c.lower():c for c in cols}
    for k in prefer:
        k=k.lower()
        if k in low: return low[k]
    for c in cols:
        lc=c.lower()
        if all(tok in lc for tok in contains): return c
    return None

def nearest_tps(x):
    try:
        x=float(x); return min(TPS, key=lambda v: abs(v-x))
    except: return None

def add(A,B):
    if B is None: return A
    O=A.copy()
    for c in O.columns[1:-1]:
        a=pd.to_numeric(O[c], errors='coerce')
        b=pd.to_numeric(B[c], errors='coerce').fillna(0)
        O[c]=(a+b).astype(float)
    return O

# load seeds
up   = pd.read_csv(os.path.join(SHIFT,'SHIFT_TABLES__UP__Throttle17.tsv'),   sep='\t', dtype=str)
down = pd.read_csv(os.path.join(SHIFT,'SHIFT_TABLES__DOWN__Throttle17.tsv'), sep='\t', dtype=str)

# load deltas if present
def rd(rel):
    p=os.path.join(ROOT,'02_passes',*rel.split('/'))
    return pd.read_csv(p, sep='\t', dtype=str) if os.path.exists(p) else None

UP  = up.copy()
for rel in ['CONSIST/CONSIST__SHIFT_UP__DELTA.tsv','LAT/LAT__SHIFT_UP__DELTA.tsv','CRUISE_TIPIN/INTENT__SHIFT_UP__DELTA.tsv']:
    d=rd(rel)
    if d is not None: UP=add(UP,d)

DOWN = down.copy()
for rel in ['CONSIST/CONSIST__SHIFT_DOWN__DELTA.tsv','CORNER/CORNER__SHIFT_DOWN__DELTA__COMBINED.tsv','STOPGO/STOPGO__SHIFT_DOWN__DELTA.tsv','KICKDOWN/KICKDOWN__SHIFT_DOWN__DELTA.tsv']:
    d=rd(rel)
    if d is not None: DOWN=add(DOWN,d)

# collect empirical events (alias-aware)
rows=[]
files=sorted(glob.glob(os.path.join(CLEAN,'__trans_focus__clean_FULL__*.csv')))
for p in files:
    try:
        df=pd.read_csv(p, low_memory=False)
    except Exception:
        continue
    gcol = pick(df, ('gear_actual__canon','gear_actual'), ('gear','actual'))
    scol = pick(df, ('speed_mph__canon','speed_mph','vss_mph'), ('speed','mph'))
    tcol = pick(df, ('throttle_pct__canon','throttle_pct'), ('throttle','%'))
    if gcol is None or scol is None or tcol is None:
        continue
    g = pd.to_numeric(df[gcol], errors='coerce')
    sp = pd.to_numeric(df[scol], errors='coerce')
    tp = pd.to_numeric(df[tcol], errors='coerce')
    prev = g.shift(1); ch=(g!=prev)&g.notna()&prev.notna()
    idx=np.where(ch.values)[0]
    for i in idx:
        a=int(prev.iat[i]); b=int(g.iat[i]);
        t=tp.iat[i]; v=sp.iat[i]
        if pd.isna(t) or pd.isna(v): continue
        bin_tps=nearest_tps(t);
        if bin_tps is None: continue
        rows.append((f"{a} -> {b} Shift", int(bin_tps), float(v)))

if not rows:
    print('[WARP] No events')
else:
    ev=pd.DataFrame(rows, columns=['row','tps','mph'])
    med=ev.groupby(['row','tps'])['mph'].median().reset_index().rename(columns={'mph':'emp'})
    def melt(df,label):
        m=df.melt(id_vars=['mph','%'], var_name='tps', value_name='pred')
        m['row']=m['mph']; m['tps']=pd.to_numeric(m['tps'], errors='coerce'); m['pred']=pd.to_numeric(m['pred'], errors='coerce'); m['kind']=label
        return m[['row','tps','pred','kind']]
    comp=pd.concat([melt(UP,'UP'), melt(DOWN,'DOWN')]).merge(med, on=['row','tps'], how='inner')
    comp['err']=comp['pred']-comp['emp']
    def rms(vals):
        s=np.array(list(vals),dtype=float)
        return float(np.sqrt(np.mean(s**2))) if len(s) else float('nan')
    def p95(vals):
        s=np.array(list(vals),dtype=float)
        return float(np.percentile(np.abs(s),95)) if len(s) else float('nan')
    summary=comp.groupby(['row','kind']).agg(n=('err','count'),rms=('err',rms),p95=('err',p95)).reset_index()
    worst=comp.assign(abs_err=np.abs(comp['err'])).sort_values('abs_err',ascending=False).head(20)
    summary.to_csv(os.path.join(OUT,'SHIFT_WARP__SUMMARY__FALLBACK.csv'),index=False)
    worst.to_csv(os.path.join(OUT,'SHIFT_WARP__WORST20__FALLBACK.csv'),index=False)
    print('[WARP] wrote:', os.path.join(OUT,'SHIFT_WARP__SUMMARY__FALLBACK.csv'))
