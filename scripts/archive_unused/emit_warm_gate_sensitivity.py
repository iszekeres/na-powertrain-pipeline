import os, glob, pandas as pd, sys
clean_dir = sys.argv[1]
cache_dir = os.path.join(clean_dir, '_NG_cache')
out_path  = sys.argv[2]
def pair_counts(files):
    counts={}
    for p in files:
        try:
            g = pd.read_csv(p, usecols=['gear_actual'])
        except Exception:
            continue
        s = pd.to_numeric(g['gear_actual'], errors='coerce').dropna().astype(int)
        prev = s.shift(1); ch = s[s!=prev]
        for idx, val in ch.items():
            if pd.isna(prev.loc[idx]): continue
            a=int(prev.loc[idx]); b=int(val)
            counts[(a,b)] = counts.get((a,b),0) + 1
    return counts
clean_files = sorted(glob.glob(os.path.join(clean_dir, '__trans_focus__clean_FULL__*.csv')))
cache_files = sorted(glob.glob(os.path.join(cache_dir, '*withbrake__NG*.csv')))
C = pair_counts(clean_files)
W = pair_counts(cache_files)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('pair,clean_count,warmed_cache_count,delta\n')
    keys = sorted(set(C) | set(W))
    for k in keys:
        A = C.get(k,0); B = W.get(k,0)
        f.write(f'{k[0]}->{k[1]},{A},{B},{B-A}\n')
print('[WRITE]', out_path)
