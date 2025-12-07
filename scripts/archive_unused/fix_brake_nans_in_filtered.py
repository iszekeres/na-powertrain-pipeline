import os,glob,pandas as pd
for d in (r"newlogs/output/00_prefilter/edi_filtered__EXCESS", r"newlogs/output/00_prefilter/edi_filtered__LAUNCH"):
    if not os.path.isdir(d): 
        continue
    for f in glob.glob(os.path.join(d,"*.csv")):
        df=pd.read_csv(f,low_memory=False)
        if "brake" in df.columns:
            df["brake"]=pd.to_numeric(df["brake"],errors="coerce").fillna(0).astype(int)
            df.to_csv(f,index=False)
print("[OK] brake NaNs fixed")
