import pandas as pd, sys
p = r"newlogs/output/02_passes/TPS_PROFILE/TPS_PROFILE__pergear_cruise.tsv"
df = pd.read_csv(p, sep="\t")
if "tps_base" not in df.columns and "tps_med" in df.columns:
    df["tps_base"] = df["tps_med"]
    df.to_csv(p, sep="\t", index=False)
    print("[TPS→EDI] added tps_base")
else:
    print("[TPS→EDI] no change needed")
