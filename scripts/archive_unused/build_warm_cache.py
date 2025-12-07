import os, glob, pandas as pd

CLEAN = r'.\\newlogs\\cleaned'
OUT   = os.path.join(CLEAN, '_NG_cache')
os.makedirs(OUT, exist_ok=True)

def derive_brake(df):
    if 'brake' in df.columns:
        return df
    aliases = [
        'brake pressure','brake_pressure','brake pressure (kpa)',
        'brake_press_kpa','brake_kpa','brakepressure'
    ]
    hits = [c for c in df.columns if c.strip().lower() in aliases]
    if hits:
        bp = pd.to_numeric(df[hits[0]], errors='coerce')
        df['brake'] = (bp >= 15).astype(int)
    else:
        df['brake'] = 0
    return df

def warm_gate(df):
    ect = df.get('ectF__canon')
    tft = df.get('tftF__canon')
    mask = pd.Series(True, index=df.index)
    if ect is not None:
        mask &= pd.to_numeric(ect, errors='coerce') >= 100
    if tft is not None:
        mask &= pd.to_numeric(tft, errors='coerce') >= 100
    return df[mask].copy()

files = sorted(glob.glob(os.path.join(CLEAN, '__trans_focus__clean_FULL__*.csv')))
if not files:
    raise SystemExit(f'No CLEAN_FULL files in {CLEAN}')

for src in files:
    df = pd.read_csv(src, low_memory=False)
    df = warm_gate(derive_brake(df))
    base = os.path.basename(src).replace('.csv','')
    out = os.path.join(OUT, f'{base}__withbrake__NG.csv')
    df.to_csv(out, index=False)
    print('[WRITE]', out, len(df))
print('[DONE] wrote', len(files), 'cache file(s) to', OUT)
