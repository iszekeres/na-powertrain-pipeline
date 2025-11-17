import pandas as pd, glob
files = glob.glob(r'newlogs\\cleaned\\__trans_focus__clean_FULL__cold-hot1__*.csv')
assert files, "No CLEAN_FULL file found for cold-hot1"
files.sort()
print("USING", files[-1])
df = pd.read_csv(files[-1])
cols = [
    "time_s",
    "speed_mph",
    "throttle_pct",
    "gear_actual",
    "gear_cmd",
    "brake",
    "tcc_slip_fused",
    "tcc_locked_built",
]
print("[NON-NULL COUNTS]")
for c in cols:
    if c not in df.columns:
        print(f"{c:20s} MISSING")
    else:
        print(f"{c:20s} {df[c].notna().sum()} / {len(df)}")
