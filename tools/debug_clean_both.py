import os, glob
import pandas as pd

root_dir = 'newlogs'
cleaned_dir = os.path.join(root_dir, 'cleaned')
clean_glob = os.path.join(cleaned_dir, '__trans_focus__clean_FULL__cold-hot1__*.csv')
files = sorted(glob.glob(clean_glob))
clean_file = files[-1]
clean = pd.read_csv(clean_file)

s = pd.to_numeric(clean['speed_mph__canon'], errors='coerce')
t = pd.to_numeric(clean['throttle_pct__canon'], errors='coerce')
both = s.notna() & t.notna()
print('both non-null count', both.sum())
if both.any():
    idx = both[both].index[0]
    print('example idx/time, speed, tps:', idx, clean.loc[idx,'time_s'], s.loc[idx], t.loc[idx])
