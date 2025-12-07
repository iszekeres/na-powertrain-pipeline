import os, glob, numpy as np, pandas as pd, sys
clean_dir = sys.argv[1]
out_path  = sys.argv[2]
def pick_col(df, prefer_exact=(), prefer_contains=()):
    cols = list(df.columns)
    low  = {c.lower(): c for c in cols}
    for key in prefer_exact:
        k = key.lower()
        if k in low: return low[k]
    for c in cols:
        lc = c.lower()
        ok = True
        for token in prefer_contains:
            if token not in lc:
                ok = False; break
        if ok: return c
    return None

def sample_window(s, i, w=3):
    if s is None or i<0 or i>=len(s): return np.nan
    if pd.notna(s.iat[i]): return s.iat[i]
    for d in range(1, w+1):
        for j in (i-d, i+d):
            if 0<=j<len(s) and pd.notna(s.iat[j]): return s.iat[j]
    return np.nan

health = dict(total=0, tps_exact=0, tps_window=0, pedal_used=0, speed_window_used=0, dropped=0)
files = sorted(glob.glob(os.path.join(clean_dir, '__trans_focus__clean_FULL__*.csv')))
for p in files:
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception:
        continue
    gear = pick_col(df, ('gear_actual__canon','gear_actual'), ('gear','actual'))
    spd  = pick_col(df, ('speed_mph__canon','speed_mph','vss_mph'), ('speed','mph'))
    thr  = pick_col(df, ('throttle_pct__canon','throttle_pct'), ('throttle','%'))
    ped  = pick_col(df, ('pedal_pct__canon','pedal_pct'), ('pedal','%'))
    if gear is None or spd is None: continue
    g   = pd.to_numeric(df[gear], errors='coerce')
    sp  = pd.to_numeric(df[spd],  errors='coerce')
    tp  = pd.to_numeric(df[thr],  errors='coerce') if thr else None
    pdal= pd.to_numeric(df[ped],  errors='coerce') if ped else None
    prev = g.shift(1); ch = (g!=prev)&g.notna()&prev.notna()
    idxs = np.where(ch.values)[0]
    for i in idxs:
        health['total'] += 1
        t = sample_window(tp, i, 3) if tp is not None else np.nan
        used_pedal = False
        if pd.isna(t) and pdal is not None:
            t = sample_window(pdal, i, 3); used_pedal = pd.notna(t)
        if pd.isna(t):
            health['dropped'] += 1; continue
        v = sample_window(sp, i, 2)
        if pd.isna(v):
            health['dropped'] += 1; continue
        if tp is not None and pd.notna(tp.iat[i]): health['tps_exact'] += 1
        elif not used_pedal: health['tps_window'] += 1
        else: health['pedal_used'] += 1
        if pd.isna(sp.iat[i]): health['speed_window_used'] += 1
with open(out_path, 'w', encoding='utf-8') as f:
    for k in ('total','tps_exact','tps_window','pedal_used','speed_window_used','dropped'):
        f.write(f"{k}: {health[k]}\n")
print('[WRITE]', out_path)
