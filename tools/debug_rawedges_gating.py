import glob, os
import pandas as pd

TPS_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]
MIN_SPEED_MPH = 3.0

def ensure_canonical_columns(df: pd.DataFrame):
    alias_map = {
        "from_gear": ["from_gear", "from"],
        "to_gear": ["to_gear", "to"],
        "speed_mph": ["speed_mph", "mph"],
        "throttle_pct": ["throttle_pct", "tps"],
    }
    for canon, candidates in alias_map.items():
        if canon in df.columns:
            continue
        for cand in candidates:
            if cand in df.columns:
                df[canon] = df[cand]
                break
    required = ["from_gear", "to_gear", "speed_mph", "throttle_pct"]
    missing = [c for c in required if c not in df.columns]
    return df, missing

root = r"C:\tuning\na-trans-data\newlogs"
events_glob = os.path.join(root, "output", "00_cleaner", "__trans_focus__shift_events__*.csv")
files = sorted(glob.glob(events_glob))
print('GLOB', events_glob)
print('FILES', [os.path.basename(f) for f in files])
if not files:
    raise SystemExit('no files')

events_file = max(files, key=os.path.getmtime)
print('USING', events_file)

df = pd.read_csv(events_file)
print('raw rows', len(df))
print('cols', list(df.columns))

df, missing = ensure_canonical_columns(df)
print('missing after alias:', missing)

for col in ["speed_mph","throttle_pct","from_gear","to_gear"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print('rows after to_numeric dropna subset:', len(df.dropna(subset=["speed_mph","throttle_pct","from_gear","to_gear"])))

filt = df.dropna(subset=["speed_mph","throttle_pct","from_gear","to_gear"]).copy()
filt = filt[filt["speed_mph"] >= MIN_SPEED_MPH]
filt = filt[(filt["throttle_pct"] >= 0.0) & (filt["throttle_pct"] <= 100.0)]

filt["from_gear"] = filt["from_gear"].astype(int)
filt["to_gear"] = filt["to_gear"].astype(int)
filt = filt[filt["from_gear"].between(1,6) & filt["to_gear"].between(1,6)]

print('rows after full gating:', len(filt))
print('unique pairs:', sorted(set(zip(filt["from_gear"],filt["to_gear"]))))
