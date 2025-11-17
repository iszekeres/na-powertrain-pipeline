import os, glob
import pandas as pd

root_dir = 'newlogs'
cleaned_dir = os.path.join(root_dir, 'cleaned')
clean_glob = os.path.join(cleaned_dir, '__trans_focus__clean_FULL__cold-hot1__*.csv')
files = sorted(glob.glob(clean_glob))
clean_file = files[-1]
clean = pd.read_csv(clean_file)

for col in ['speed_mph__canon','throttle_pct__canon']:
    s = pd.to_numeric(clean[col], errors='coerce')
    non_null = s.dropna()
    print(col, 'non-null count', non_null.shape[0])
    if not non_null.empty:
        idx_first = non_null.index[0]
        idx_last = non_null.index[-1]
        print('  first idx/time, val:', idx_first, clean.loc[idx_first,'time_s'], non_null.iloc[0])
        print('  last idx/time, val:', idx_last, clean.loc[idx_last,'time_s'], non_null.iloc[-1])
