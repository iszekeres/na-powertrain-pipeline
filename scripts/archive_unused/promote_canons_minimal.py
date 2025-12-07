import glob, os, pandas as pd, numpy as np
def pick(df,*names):
    for n in names:
        if n and n in df.columns: return n
    return None

changed_any=False
for p in sorted(glob.glob(r"newlogs/cleaned/*.csv")):
    df = pd.read_csv(p, low_memory=False)
    changed=False

    if "speed_mph" not in df.columns:
        c = pick(df,"speed_mph__canon","Vehicle Speed (SAE)","vss_mph")
        if c is not None:
            df["speed_mph"] = pd.to_numeric(df[c], errors="coerce"); changed=True

    if "throttle_pct" not in df.columns:
        c = pick(df,"throttle_pct__canon","Throttle Position","Throttle Position (SAE)","Throttle Position (%)")
        if c is not None:
            df["throttle_pct"] = pd.to_numeric(df[c], errors="coerce"); changed=True

    if "pedal_pct" not in df.columns:
        c = pick(df,"pedal_pct__canon","Accelerator Pedal Position","Accelerator Pedal Position (SAE)")
        if c is not None:
            df["pedal_pct"] = pd.to_numeric(df[c], errors="coerce"); changed=True

    if "gear_actual" not in df.columns:
        c = pick(df,"gear_actual__canon","gear_actual")
        if c is not None:
            df["gear_actual"] = pd.to_numeric(df[c], errors="coerce"); changed=True

    if "time_s" not in df.columns:
        c = pick(df,"time_s","offset__canon","Offset")
        if c is not None:
            df["time_s"] = pd.to_numeric(df[c], errors="coerce"); changed=True

    if "brake" not in df.columns:
        c = pick(df,"brake__canon","Brake Pressure","Brake Pedal")
        if c is not None:
            s = pd.to_numeric(df[c], errors="coerce")
            if "Pressure" in c: br = (s>=15).astype("Int64")  # kPa threshold ~15
            else:               br = (s>0.5).astype("Int64")
            df["brake"] = br.fillna(0).astype(int); changed=True

    if changed:
        df.to_csv(p, index=False)
        print(f"[PROMOTE] fixed -> {os.path.basename(p)}")
        changed_any=True
    else:
        print(f"[PROMOTE] ok -> {os.path.basename(p)}")

print("[PROMOTE] done")
