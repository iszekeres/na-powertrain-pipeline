# -*- coding: utf-8 -*-
import sys, pandas as pd

def pick(cols, *names):
    low = {c.lower(): c for c in cols}
    for n in names:
        c = low.get(n.lower())
        if c: return c
    return None

def try_speed(df, cols):
    # mph first
    mph = pick(cols,
        "speed_mph","speed_mph__canon",
        "vss_mph","vss_mph__canon",
        "Vehicle Speed (mph)","Vehicle Speed (mph)__canon",
        "vehicle_speed_mph","vehicle_speed_mph__canon",
        "gps_mph","gps_mph__canon"
    )
    if mph: return df[mph]
    # kph fallback
    kph = pick(cols,
        "vehicle_speed","vehicle_speed__canon",
        "vehicle_speed_kph","vehicle_speed_kph__canon",
        "vss_kph","vss_kph__canon",
        "gps_kph","gps_kph__canon",
        "Vehicle Speed (kph)","Vehicle Speed (kph)__canon"
    )
    if kph: return df[kph].astype("float64") * 0.621371
    raise SystemExit("corefix: missing vehicle speed column (mph/kph)")

def main(p):
    df = pd.read_csv(p, low_memory=False)
    cols = list(df.columns)

    # speed_mph
    if "speed_mph" not in df.columns:
        df["speed_mph"] = try_speed(df, cols)

    # time_s
    t = pick(cols, "time_s","time_s__canon","offset","Offset (s)","Time (s)","Elapsed Time (s)","elapsed_s","elapsed_time_s")
    if t is None: raise SystemExit("corefix: missing time/offset column")
    if "time_s" not in df.columns: df["time_s"] = df[t]

    # throttle_pct (prefer blade; fall back to pedal)
    thr = pick(cols,
        "throttle_pct","throttle_pct__canon",
        "Throttle Position (%)","Throttle (%)","TPS","TPS (%)",
        "Accelerator Pedal Position (%)","pedal_pct","pedal_pct__canon"
    )
    if thr is None: raise SystemExit("corefix: missing throttle column")
    if "throttle_pct" not in df.columns: df["throttle_pct"]=df[thr]

    # gear_actual (optional)
    gear = pick(cols,"gear_actual","gear_actual__canon","Trans Current Gear","Current Gear","Gear","Gear__canon")
    if (gear is not None) and ("gear_actual" not in df.columns): df["gear_actual"]=df[gear]

    # tcc_locked_built (soft-lock from slip if present)
    if "tcc_locked_built" not in df.columns:
        slip = pick(cols,"tcc_slip_fused","tcc_slip_fused__canon","TCC Slip (rpm)","TCC Slip","Converter Slip (RPM)")
        if slip:
            import numpy as np
            s = df[slip].astype("float64")
            g = df["gear_actual"] if "gear_actual" in df.columns else 0
            sp= df["speed_mph"].astype("float64")
            lock = (s.abs()<=30) & (sp>=25) & ((g if hasattr(g,"__array__") else 0) >= 3)
            df["tcc_locked_built"] = lock.astype(int)

    df.to_csv(p, index=False)
    print("[COREFIX] OK ->", p)

if __name__=="__main__":
    if len(sys.argv)<2: raise SystemExit("usage: corefix_strict.py <clean_full.csv>")
    main(sys.argv[1])
