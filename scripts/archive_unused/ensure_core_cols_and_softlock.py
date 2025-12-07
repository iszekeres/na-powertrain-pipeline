import glob, os, pandas as pd, numpy as np

def pick(df, *cands):
    for c in cands:
        if c in df.columns: return c
    return None

def ensure_one(df, out_name, *cands, numeric=False):
    if out_name in df.columns: return False
    src = pick(df, *cands)
    if src is None: return False
    df[out_name] = pd.to_numeric(df[src], errors="coerce") if numeric else df[src]
    return True

def build_tcc_locked(df):
    # need slip + time if we can
    slip_col = pick(df,
        "tcc_slip_fused","tcc_slip_fused__canon","tcc_slip__canon",
        "TCC Slip","TCC Slip RPM","Trans Slip RPM")
    if slip_col is None:
        return False
    slip = pd.to_numeric(df[slip_col], errors="coerce")
    # estimate sampling rate from time_s if present
    if "time_s" in df.columns:
        ts = pd.to_numeric(df["time_s"], errors="coerce")
        dt = ts.diff().median()
        rate = 10.0 if not (pd.notna(dt) and dt>0) else (1.0/float(dt))
    else:
        rate = 10.0
    on_n  = max(1, int(round(0.6*rate)))   # about 0.6 s window
    off_n = max(1, int(round(0.4*rate)))   # about 0.4 s window
    low  = slip.le(30).rolling(on_n, min_periods=1).mean().ge(0.8)
    high = slip.ge(80).rolling(off_n,min_periods=1).mean().ge(0.8)
    lock = pd.Series(False, index=df.index)
    L=False
    for i in range(len(df)):
        if L:
            if high.iat[i]: L=False
        else:
            if low.iat[i]: L=True
        lock.iat[i]=L
    df["tcc_locked_built"] = lock.astype(int)
    return True

changed_any=False
for path in sorted(glob.glob(r"newlogs/cleaned/*.csv")):
    df = pd.read_csv(path, low_memory=False)
    changed=False

    # Promote base columns from canon/raw
    changed |= ensure_one(df,"speed_mph",
        "speed_mph","speed_mph__canon","Vehicle Speed (SAE)","Vehicle Speed","VSS","VSS (mph)",
        numeric=True)
    changed |= ensure_one(df,"throttle_pct",
        "throttle_pct","throttle_pct__canon","Throttle Position","Throttle Position (SAE)","Throttle Position (%)",
        numeric=True)
    changed |= ensure_one(df,"pedal_pct",
        "pedal_pct","pedal_pct__canon","Accelerator Pedal Position","Accelerator Pedal Position (SAE)",
        numeric=True)
    changed |= ensure_one(df,"gear_actual",
        "gear_actual","gear_actual__canon","Trans Gear","Gear","Transmission Gear",
        numeric=True)
    if "time_s" not in df.columns:
        src = pick(df, "time_s","offset__canon","Offset","offset")
        if src is not None:
            df["time_s"] = pd.to_numeric(df[src], errors="coerce"); changed=True

    # tcc_slip_fused if missing
    if "tcc_slip_fused" not in df.columns:
        c = pick(df,"tcc_slip_fused__canon","tcc_slip__canon","TCC Slip","TCC Slip RPM","Trans Slip RPM")
        if c is not None:
            df["tcc_slip_fused"] = pd.to_numeric(df[c], errors="coerce"); changed=True

    # brake binary if missing (15 kPa threshold on pressure; else >0)
    if "brake" not in df.columns:
        bsrc = pick(df,"brake__canon","Brake Pressure","Brake Pressure (kPa)","Brake Pedal")
        if bsrc is not None:
            s = pd.to_numeric(df[bsrc], errors="coerce")
            if "Pressure" in bsrc:
                df["brake"] = (s.fillna(0)>=15).astype(int)
            else:
                df["brake"] = (s.fillna(0)>0.5).astype(int)
        else:
            df["brake"] = 0
        changed=True

    # tcc_locked_built if missing
    if "tcc_locked_built" not in df.columns:
        if build_tcc_locked(df): changed=True

    if changed:
        df.to_csv(path, index=False)
        print("[COREFIX] fixed ->", os.path.basename(path))
        changed_any=True
    else:
        print("[COREFIX] ok    ->", os.path.basename(path))

if not changed_any:
    print("[COREFIX] nothing to change")
