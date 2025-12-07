import os, pandas as pd
PASS=r'.\\newlogs\\output\\02_passes'

def load(rel):
    path=os.path.join(PASS,*rel.split('/'))
    return pd.read_csv(path,sep='\t') if os.path.exists(path) else None

def nonzero(df):
    if df is None: return 0
    cols=[c for c in df.columns if c not in ('mph','%')]
    return sum((pd.to_numeric(df[c],errors='coerce').fillna(0)!=0).sum() for c in cols)

def range_check(name,df,lo,hi,sign=None):
    if df is None:
        print(f'[MISS] {name}'); return
    bad=[]; N=nonzero(df)
    cols=[c for c in df.columns if c not in ('mph','%')]
    for col in cols:
        val=pd.to_numeric(df[col],errors='coerce')
        vv=val[(val.notna())&(val!=0)]
        if sign=='<=0':
            bad.extend([(df.loc[i,'mph'],col,float(x)) for i,x in vv[vv>0].items()])
        else:
            bad.extend([(df.loc[i,'mph'],col,float(x)) for i,x in vv[(vv<lo)|(vv>hi)].items()])
    status='OK' if not bad else f'{len(bad)} out of {int(N)} out-of-range'
    print(f'[RANGE] {name}: {status}')
    for row,col,x in bad[:8]:
        print(f'  row={row} tps={col}% val={x}')

def main():
    stopgo=load('STOPGO/STOPGO__SHIFT_DOWN__DELTA.tsv')
    kick=load('KICKDOWN/KICKDOWN__SHIFT_DOWN__DELTA.tsv')
    intent_u=load('CRUISE_TIPIN/INTENT__SHIFT_UP__DELTA.tsv')
    intent_r=load('CRUISE_TIPIN/INTENT__TCC_RELEASE__DELTA.tsv')
    range_check('STOPGO_DOWN',stopgo,0.0,0.3)
    range_check('KICKDOWN_DOWN',kick,0.0,0.3)
    range_check('INTENT_UP',intent_u,0.0,0.2)
    range_check('INTENT_TCC',intent_r,0.0,0.0,sign='<=0')
    print('[NOTE] Any delta rows with <6 shift events should be treated cautiously during blend.')

if __name__=='__main__': main()
