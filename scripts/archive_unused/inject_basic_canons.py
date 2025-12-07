import sys, os, glob, pandas as pd, numpy as np

"""
usage: python inject_basic_canons.py <clean_dir>
Adds time_s__canon, ectF__canon, tftF__canon if missing (alias-aware).
"""

def pick_col(cols, *cands, contains=()):
    low = {c.lower(): c for c in cols}
    for k in cands:
        if k and k.lower() in low:
            return low[k.lower()]
    for c in cols:
        lc = c.lower()
        if all(tok in lc for tok in contains):
            return c
    return None

def as_float(series):
    return pd.to_numeric(series, errors='coerce')

def convert_to_f(series):
    data = as_float(series)
    med = data.dropna().median()
    if pd.isna(med):
        return data
    if 20 <= med <= 120:
        return data * 9/5 + 32
    return data

def ensure_time(df):
    if 'time_s__canon' in df.columns:
        return False
    col = pick_col(df.columns, 'time_s', 'offset', contains=('time',))
    if col is None:
        return False
    df['time_s__canon'] = as_float(df[col])
    return True

def ensure_temp(df, out_col, prefer_exact, contains_alias):
    if out_col in df.columns:
        return False
    col = pick_col(df.columns, *prefer_exact, contains=contains_alias)
    if col is None:
        return False
    df[out_col] = convert_to_f(df[col])
    return True

def process(path):
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print('[SKIP]', os.path.basename(path), exc)
        return False
    changed = False
    if ensure_time(df):
        changed = True
    if ensure_temp(df, 'ectF__canon', ('ectF__canon','ectF','ect (f)','engine coolant temp'), ('coolant','temp')):
        changed = True
    if ensure_temp(df, 'tftF__canon', ('tftF__canon','tftF','trans temp'), ('trans','temp')):
        changed = True
    if changed:
        df.to_csv(path, index=False)
        print('[WRITE]', os.path.basename(path))
    else:
        print('[OK   ]', os.path.basename(path))
    return changed

def main(root):
    files = sorted(glob.glob(os.path.join(root, '__trans_focus__clean_FULL__*.csv')))
    if not files:
        print('[ERR] No CLEAN_FULL files in', root)
        return 1
    count = 0
    missing = []
    for path in files:
        before = set(pd.read_csv(path, nrows=0).columns)
        changed = process(path)
        after = set(pd.read_csv(path, nrows=0).columns)
        for col in ('time_s__canon','ectF__canon','tftF__canon'):
            if col not in after:
                missing.append((os.path.basename(path), col))
        if changed:
            count += 1
    if missing:
        print('[WARN] Some files still missing canon columns:')
        for fname, col in missing:
            print('   ', fname, '->', col)
    print(f'[DONE] changed {count}/{len(files)} files')
    return 0

if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else r'.\\newlogs\\cleaned'
    sys.exit(main(root))
