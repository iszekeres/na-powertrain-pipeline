import pandas as pd

p = r".\newlogs\output\02_passes\CONSIST\CONSIST__SHIFT_DOWN__DELTA.tsv"
df = pd.read_csv(p, sep="\t")

# Grab numeric columns (skip the left row-label, and trailing '%' if present)
cols = df.columns[1:-1] if df.columns[-1] == "%" else df.columns[1:]
num  = df[cols].apply(pd.to_numeric, errors="coerce")

# Clamp any positive values to 0 to preserve hysteresis
df[cols] = num.where(num <= 0, 0)

df.to_csv(p, sep="\t", index=False)
print("Clamped positive DOWN deltas to 0.")
