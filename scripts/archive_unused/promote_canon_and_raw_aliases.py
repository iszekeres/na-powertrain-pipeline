#!/usr/bin/env python3
# promote_canon_and_raw_aliases.py
# Promotes __canon fields and, if missing, maps from common RAW aliases to unsuffixed canonical names.
# Ensures: time_s, speed_mph, throttle_pct, gear_actual, engine_rpm, trans_input_rpm, trans_output_rpm, pedal_pct, tft_c, ect_c, brake

import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np

ALI = {
  "time_s": ["offset__canon","Offset","offset"],
  "speed_mph": ["speed_mph__canon","Vehicle Speed (SAE)","Vehicle Speed","vss_mph","VSS"],
  "throttle_pct": ["throttle_pct__canon","Throttle Position","Throttle Position (SAE)","Throttle Position (%)","TPS (%)","TPS"],
  "pedal_pct": ["pedal_pct__canon","Accelerator Pedal Position","Accel Pedal Position","APP (%)","Accel Pedal (%)"],
  "gear_actual": ["gear_actual__canon","gear_actual"],
  "gear_cmd": ["gear_cmd__canon","gear_cmd"],
  "engine_rpm": ["engine_rpm__canon","Engine RPM (SAE)","RPM","Engine RPM"],
  "trans_input_rpm": ["turbine_rpm__canon","Trans Input Shaft RPM","Transmission Input Shaft RPM","Turbine RPM","ISS","Trans Input Shaft Speed"],
  "trans_output_rpm": ["output_rpm__canon","Trans Output Shaft RPM","OSS","Output Shaft RPM"],
  "tcc_slip_fused": ["tcc_slip_fused__canon","tcc_slip__canon","TCC Slip"],
  "tcc_desired": ["tcc_desired__canon","TCC Desired Slip","TCC Desired"],
  "tft_c": ["trans_temp_c__canon","Trans Fluid Temp","Transmission Fluid Temperature","Trans Temp C","Trans Temp (°C)","Trans Temp"],
  "tft_f": ["trans_temp_f__canon","Trans Temp F","Trans Temp (°F)"],
  "ect_c": ["Engine Coolant Temp (SAE)","ECT (°C)","Coolant Temp C"],
  "brake": ["brake__canon","Brake"],
  "brake_pressure_kpa": ["Brake Pressure (kPa)","Brake Pressure","Brake Pressure (SAE)"],
}

def pick(df, names):
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        c = cols.get(n.lower())
        if c: return c
    # substring fallback
    for n in names:
        tgt = n.lower().replace(" ","").replace("_","")
        for k,v in cols.items():
            if tgt in k.replace(" ","").replace("_",""):
                return v
    return None

def ensure_time(df):
    if "time_s" in df.columns: return
    c = pick(df, ALI["time_s"])
    if c: df["time_s"] = pd.to_numeric(df[c], errors="coerce")

def ensure_core(df):
    # speed_mph
    if "speed_mph" not in df.columns:
        c = pick(df, ALI["speed_mph"])
        if c: df["speed_mph"] = pd.to_numeric(df[c], errors="coerce")
    # throttle_pct
    if "throttle_pct" not in df.columns:
        c = pick(df, ALI["throttle_pct"])
        if c: df["throttle_pct"] = pd.to_numeric(df[c], errors="coerce")
    # gear_actual
    if "gear_actual" not in df.columns:
        c = pick(df, ALI["gear_actual"])
        if c: df["gear_actual"] = pd.to_numeric(df[c], errors="coerce")
    # engine_rpm
    if "engine_rpm" not in df.columns:
        c = pick(df, ALI["engine_rpm"])
        if c: df["engine_rpm"] = pd.to_numeric(df[c], errors="coerce")
    # turbine/input
    if "trans_input_rpm" not in df.columns:
        c = pick(df, ALI["trans_input_rpm"])
        if c: df["trans_input_rpm"] = pd.to_numeric(df[c], errors="coerce")
    if "turbine_rpm" not in df.columns and "trans_input_rpm" in df.columns:
        df["turbine_rpm"] = pd.to_numeric(df["trans_input_rpm"], errors="coerce")
    # output
    if "trans_output_rpm" not in df.columns:
        c = pick(df, ALI["trans_output_rpm"])
        if c: df["trans_output_rpm"] = pd.to_numeric(df[c], errors="coerce")
    # pedal
    if "pedal_pct" not in df.columns:
        c = pick(df, ALI["pedal_pct"])
        if c: df["pedal_pct"] = pd.to_numeric(df[c], errors="coerce")

def ensure_temps_brake(df):
    # ECT
    if "ect_c" not in df.columns:
        c = pick(df, ALI["ect_c"])
        if c: df["ect_c"] = pd.to_numeric(df[c], errors="coerce")
    # TFT C or convert from F
    if "tft_c" not in df.columns:
        cC = pick(df, ALI["tft_c"])
        cF = pick(df, ALI["tft_f"])
        if cC:
            df["tft_c"] = pd.to_numeric(df[cC], errors="coerce")
        elif cF:
            f = pd.to_numeric(df[cF], errors="coerce")
            df["tft_c"] = (f - 32.0) * (5.0/9.0)
    # brake
    if "brake" not in df.columns:
        c = pick(df, ALI["brake"])
        if c:
            v = df[c].astype(str).str.strip().str.lower()
            df["brake"] = v.isin(["1","true","on","yes"]).astype(int)
        else:
            bp = pick(df, ALI["brake_pressure_kpa"])
            if bp:
                vals = pd.to_numeric(df[bp], errors="coerce")
                df["brake"] = (vals >= 15).astype(int)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaned-dir", required=True)
    a = ap.parse_args()
    from pathlib import Path
    base = Path(a.cleaned_dir)
    paths = sorted(base.glob("*.csv"))
    if not paths:
        print(f"[ERROR] No CSVs in {base}", file=sys.stderr); sys.exit(2)
    for p in paths:
        try:
            df = pd.read_csv(p, low_memory=False)
            ensure_time(df)
            ensure_core(df)
            ensure_temps_brake(df)
            df.to_csv(p, index=False)
            print(f"[OK] Promoted canon/raw aliases in {p.name}")
        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")
            sys.exit(2)
if __name__ == "__main__":
    main()
