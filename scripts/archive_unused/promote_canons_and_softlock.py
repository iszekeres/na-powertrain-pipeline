import glob, os, pandas as pd, numpy as np
clean_dir = r"newlogs/cleaned"
files = sorted(glob.glob(os.path.join(clean_dir, "*.csv")))
if not files: print("[PROMOTE] no cleaned files found"); raise SystemExit(0)

def pick(df, names):
    for n in names:
        if n in df.columns: return n
    return None

for f in files:
    df = pd.read_csv(f, low_memory=False)
    changed = False

    # expanded candidates
    mapcand = {
        "speed_mph": [
            "speed_mph","speed_mph__canon","Vehicle Speed (SAE)","Vehicle Speed",
            "VSS","VSS (mph)"
        ],
        "throttle_pct": [
            "throttle_pct","throttle_pct__canon","Throttle Position","Throttle Position (SAE)",
            "Throttle Position (%)","Throttle Blade Position","Throttle Desired Position"
        ],
        "pedal_pct": [
            "pedal_pct","pedal_pct__canon","Accelerator Pedal Position","Driver Pedal Axle Torque Req"
        ],
        "gear_actual": [
            "gear_actual","gear_actual__canon","Trans Gear","Gear","Transmission Gear"
        ],
        "time_s": [
            "time_s","offset__canon","Offset","offset"
        ],
    }
    for out, cands in mapcand.items():
        if out not in df.columns:
            src = pick(df, cands)
            if src:
                s = pd.to_numeric(df[src], errors="coerce") if out in ("speed_mph","pedal_pct","throttle_pct","time_s") else df[src]
                df[out] = s
                changed = True

    # brake from pressure if missing
    if "brake" not in df.columns:
        bpname = pick(df, ["brake","Brake Pressure","Brake Pressure (kPa)"])
        if bpname:
            bp = pd.to_numeric(df[bpname], errors="coerce")
            df["brake"] = (bp.fillna(0) >= 15).astype(int)
        else:
            df["brake"] = 0
        changed = True

    # fused slip + soft-lock
    if "tcc_slip_fused" not in df.columns:
        slip = pick(df, ["tcc_slip_fused__canon","tcc_slip_fused","TCC Slip","TCC Slip RPM","Trans Slip RPM"])
        if slip:
            df["tcc_slip_fused"] = pd.to_numeric(df[slip], errors="coerce"); changed = True
    if "tcc_locked_built" not in df.columns:
        sl = pd.to_numeric(df.get("tcc_slip_fused", np.nan), errors="coerce")
        on  = sl.le(30).rolling(6, min_periods=1).mean().le(30)
        off = sl.ge(80).rolling(4, min_periods=1).mean().ge(80)
        lock = pd.Series(False, index=df.index)
        locked = False
        for i in range(len(df)):
            locked = (not locked and on.iat[i]) or (locked and not off.iat[i])
            lock.iat[i] = locked
        df["tcc_locked_built"] = lock.astype(int); changed = True

    if changed: df.to_csv(f, index=False); print("[PROMOTE] fixed ->", os.path.basename(f))
    else:       print("[PROMOTE] ok    ->", os.path.basename(f))
