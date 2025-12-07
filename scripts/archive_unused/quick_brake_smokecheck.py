#!/usr/bin/env python3
import sys, glob, pandas as pd
files=sorted(glob.glob(sys.argv[1] if len(sys.argv)>1 else r".\newlogs\cleaned\*.csv"))
if not files: print("[ERROR] No CLEAN files found"); raise SystemExit(2)
df=pd.read_csv(files[0], nrows=200000, low_memory=False)
if "brake" not in df.columns: print("[ERROR] 'brake' column missing in sample file:", files[0]); raise SystemExit(2)
print("[OK]", files[0]); print(df["brake"].value_counts(dropna=False).to_string())
