import pandas as pd
p=r"newlogs/output/02_passes/TPS_PROFILE/TPS_PROFILE__pergear_cruise.tsv"
df=pd.read_csv(p,sep="\t")
if "tps_base" not in df.columns:
    if "tps_med" in df.columns:   df["tps_base"]=df["tps_med"]
    elif "tps_mean" in df.columns:df["tps_base"]=df["tps_mean"]
    else: raise SystemExit("TPS profile missing a usable base column")
df.to_csv(p,sep="\t",index=False); print("[TPS] tps_base ready")
