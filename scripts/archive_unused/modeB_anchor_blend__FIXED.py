import sys, pathlib as p, pandas as pd, numpy as np, re
TPSH=["mph","0","6","12","19","25","31","37","44","50","56","62","69","75","81","87","94","100","%"]
TPS=TPSH[1:-1]

def read_tsv(tsv):
    df=pd.read_csv(tsv,sep='\t',header=None,engine='python')
    if str(df.iloc[0,0]).strip().lower()=="mph": df.columns=TPSH; df=df.iloc[1:].reset_index(drop=True)
    assert df.shape[1]==19
    df.columns=TPSH; df=df.drop(columns=['%']).set_index('mph')
    for c in TPS: df[c]=pd.to_numeric(df[c],errors='coerce')
    return df

def write_tsv(df,outp):
    o=df.copy(); o['%']=''; o=o.reset_index(); o.iloc[:,1:-1]=np.round(o.iloc[:,1:-1].astype(float),1)
    with open(outp,'w',newline='') as f: f.write('\t'.join(TPSH)+'\n'); o.to_csv(f,sep='\t',header=False,index=False)

def pav(y):
    blocks=[[i] for i in range(len(y))]
    vals=list(y.astype(float))
    i=0
    while i<len(blocks)-1:
        if vals[i] <= vals[i+1] + 1e-9:
            i+=1; continue
        new_block = blocks[i] + blocks[i+1]
        new_val = (vals[i] + vals[i+1]) / 2.0
        blocks[i]=new_block; vals[i]=new_val
        del blocks[i+1]; del vals[i+1]
        if i>0: i-=1
    z=np.zeros_like(y)
    idx=0
    for b,v in zip(blocks, vals):
        for _ in b:
            z[idx]=v; idx+=1
    return z

def band(col):
    t=int(col)
    if t<=31: return 3.5
    if t<=56: return 2.5
    if t<=75: return 1.8
    return 1.2

def fit_row(rowname, sug_row, neu_row):
    y = sug_row.values.astype(float)
    z = pav(y)
    for j,c in enumerate(TPS):
        lo, hi = neu_row[c]-band(c), neu_row[c]+band(c)
        z[j] = np.clip(z[j], lo, hi)
    for j in range(1,len(z)):
        if z[j] < z[j-1]: z[j] = z[j-1]
        if j>=2 and abs(z[j]-z[j-1])<1e-9 and abs(z[j-1]-z[j-2])<1e-9:
            z[j] = z[j-1] + 0.1
    return pd.Series(z, index=TPS)

final_dir, tf_dir, suggest_tsv, out_dir = map(p.Path, sys.argv[1:5]); out_dir.mkdir(parents=True, exist_ok=True)
up_log = read_tsv(final_dir/'SHIFT_TABLES__UP__Throttle17.tsv')
dn_log = read_tsv(final_dir/'SHIFT_TABLES__DOWN__Throttle17.tsv')
up_neu = read_tsv(tf_dir/'SHIFT_TABLES__UP__Throttle17.tsv')
dn_neu = read_tsv(tf_dir/'SHIFT_TABLES__DOWN__Throttle17.tsv')
sug    = read_tsv(suggest_tsv)
up_s = sug.loc[[r for r in sug.index if re.match(r"^\s*(\d)\s*->\s*(\d)",r) and int(re.match(r"^\s*(\d)",r).group(1)) < int(re.search(r"->\s*(\d)",r).group(1))]].reindex(up_neu.index).fillna(up_log)
dn_s = sug.loc[[r for r in sug.index if re.match(r"^\s*(\d)\s*->\s*(\d)",r) and int(re.match(r"^\s*(\d)",r).group(1)) > int(re.search(r"->\s*(\d)",r).group(1))]].reindex(dn_neu.index).fillna(dn_log)
up_new = up_neu.copy(); dn_new = dn_neu.copy()
for r in up_new.index: up_new.loc[r] = fit_row(r, up_s.loc[r], up_neu.loc[r])
for r in dn_new.index: dn_new.loc[r] = fit_row(r, dn_s.loc[r], dn_neu.loc[r])
write_tsv(up_new, out_dir/'SHIFT_TABLES__UP__Throttle17.tsv')
write_tsv(dn_new, out_dir/'SHIFT_TABLES__DOWN__Throttle17.tsv')
print('[MODEB_FIXED] wrote anchored SHIFT tables.')
