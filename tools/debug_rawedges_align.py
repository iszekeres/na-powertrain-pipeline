import glob, os
import pandas as pd

TPS_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]
MIN_SPEED_MPH = 3.0


def ensure_canonical_columns(df: pd.DataFrame):
    alias_map = {
        "from_gear": ["from_gear", "from"],
        "to_gear": ["to_gear", "to"],
    }
    for canon, candidates in alias_map.items():
        if canon in df.columns:
            continue
        for cand in candidates:
            if cand in df.columns:
                df[canon] = df[cand]
                break
    required = ["time_s", "from_gear", "to_gear"]
    missing = [c for c in required if c not in df.columns]
    return df, missing

root_dir = "newlogs"

events_glob = os.path.join(root_dir, "output", "00_cleaner", "__trans_focus__shift_events__*.csv")
files = sorted(glob.glob(events_glob))
print('GLOB', events_glob)
print('FILES', [os.path.basename(f) for f in files])
if not files:
    raise SystemExit('no shift_events')

events_file = max(files, key=os.path.getmtime)
base = os.path.basename(events_file)
print('USING events:', events_file)

parts = base.split("__")
print('parts', parts)
idx = parts.index('shift_events')
log_id = parts[idx+1]
print('log_id', log_id)

se = pd.read_csv(events_file)
print('se rows', len(se))
se, missing = ensure_canonical_columns(se)
print('missing after alias', missing)

cleaned_dir = os.path.join(root_dir, 'cleaned')
clean_glob = os.path.join(cleaned_dir, f"__trans_focus__clean_FULL__{log_id}__*.csv")
clean_files = glob.glob(clean_glob)
print('CLEAN glob', clean_glob)
print('CLEAN files', [os.path.basename(f) for f in clean_files])
if not clean_files:
    raise SystemExit('no CLEAN_FULL')

clean_file = max(clean_files, key=os.path.getmtime)
print('USING clean:', clean_file)

clean = pd.read_csv(clean_file)
print('clean rows', len(clean))

clean_slim = clean[['time_s','speed_mph__canon','throttle_pct__canon']].copy()
clean_slim = clean_slim.rename(columns={'speed_mph__canon':'speed_mph','throttle_pct__canon':'throttle_pct'})

se['time_s'] = pd.to_numeric(se['time_s'], errors='coerce')
clean_slim['time_s'] = pd.to_numeric(clean_slim['time_s'], errors='coerce')

se = se.dropna(subset=['time_s']).copy()
clean_slim = clean_slim.dropna(subset=['time_s']).copy()

se = se.sort_values('time_s')
clean_slim = clean_slim.sort_values('time_s')

merged = pd.merge_asof(se, clean_slim, on='time_s', direction='nearest', suffixes=('', '_full'))
print('merged rows', len(merged))
print('merged cols', list(merged.columns))
print('merged head:', merged[['time_s','from_gear','to_gear','speed_mph','throttle_pct']].head(20))

for col in ['speed_mph','throttle_pct','from_gear','to_gear']:
    merged[col] = pd.to_numeric(merged[col], errors='coerce')

non_null = merged.dropna(subset=['speed_mph','throttle_pct','from_gear','to_gear'])
print('non_null rows', len(non_null))
print('non_null head:', non_null[['time_s','from_gear','to_gear','speed_mph','throttle_pct']].head(20))
