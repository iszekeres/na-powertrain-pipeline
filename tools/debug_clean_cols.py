import os, glob
import pandas as pd

root_dir = 'newlogs'
cleaned_dir = os.path.join(root_dir, 'cleaned')
clean_glob = os.path.join(cleaned_dir, '__trans_focus__clean_FULL__cold-hot1__*.csv')
files = sorted(glob.glob(clean_glob))
print('CLEAN files', [os.path.basename(f) for f in files])
if not files:
    raise SystemExit('no CLEAN_FULL')
clean_file = files[-1]
print('using', clean_file)
clean = pd.read_csv(clean_file)
for col in ['speed_mph__canon','throttle_pct__canon']:
    s = pd.to_numeric(clean[col], errors='coerce')
    print(col, 'non-null', s.notna().sum(), 'total', len(s), 'min', s.min(), 'max', s.max())
print('time_s non-null', pd.to_numeric(clean['time_s'], errors='coerce').notna().sum())
