#!/usr/bin/env python3
# promote_canon_columns.py
# Promote CLEAN_FULL __canon fields to unsuffixed canonical names expected by downstream tools.
# Also derive time_s from offset/offset__canon if missing; create ect_c/tft_c if only canon/raw fields exist.
import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np

MAPS = [
    ("speed_mph__canon","speed_mph"),
    ("throttle_pct__canon","throttle_pct"),
    ("pedal_pct__canon","pedal_pct"),
    ("gear_actual__canon","gear_actual"),
    ("gear_cmd__canon","gear_cmd"),
    ("engine_rpm__canon","engine_rpm"),
    ("turbine_rpm__canon","trans_input_rpm"),   # preferred alias
    ("output_rpm__canon","trans_output_rpm"),
    ("tcc_slip_fused__canon","tcc_slip_fused"),
    ("tcc_slip__canon","tcc_slip_fused"),       # fallback if fused missing
    ("tcc_desired__canon","tcc_desired"),
    ("trans_temp_c__canon","tft_c"),
    ("trans_temp_f__canon","tft_f"),
    ("brake__canon","brake"),
    ("tcc_locked_built__canon","tcc_locked_built"),
]

def col(df, name):
    return name if name in df.columns else None

def ensure_time_s(df):
    if "time_s" in df.columns: return
    if "offset__canon" in df.columns:
        df["time_s"] = pd.to_numeric(df["offset__canon"], errors="coerce")
        return
    # raw "Offset" often present
    if "Offset" in df.columns:
        df["time_s"] = pd.to_numeric(df["Offset"], errors="coerce")
        return
    # last resort: try lowercase offset
    if "offset" in df.columns:
        df["time_s"] = pd.to_numeric(df["offset"], errors="coerce")
        return

def ensure_ect_tft(df):
    if "ect_c" not in df.columns:
        # SAE ECT is usually °C
        if "Engine Coolant Temp (SAE)" in df.columns:
            df["ect_c"] = pd.to_numeric(df["Engine Coolant Temp (SAE)"], errors="coerce")
    # TFT
    if "tft_c" not in df.columns:
        if "trans_temp_c__canon" in df.columns:
            df["tft_c"] = pd.to_numeric(df["trans_temp_c__canon"], errors="coerce")
        elif "Trans Fluid Temp" in df.columns:
            # unknown units; do not convert blindly; pass through
            df["tft_c"] = pd.to_numeric(df["Trans Fluid Temp"], errors="coerce")
        elif "trans_temp_f__canon" in df.columns:
            f = pd.to_numeric(df["trans_temp_f__canon"], errors="coerce")
            df["tft_c"] = (f - 32.0) * (5.0/9.0)

def ensure_rpm_aliases(df):
    # Some tools expect both turbine and trans_input_rpm aliases
    if "turbine_rpm" not in df.columns and "trans_input_rpm" in df.columns:
        df["turbine_rpm"] = pd.to_numeric(df["trans_input_rpm"], errors="coerce")
    if "trans_input_rpm" not in df.columns and "turbine_rpm" in df.columns:
        df["trans_input_rpm"] = pd.to_numeric(df["turbine_rpm"], errors="coerce")

def promote(df):
    # Straight copies if destination missing or all-NaN
    for src, dst in MAPS:
        if src in df.columns:
            if dst not in df.columns or df[dst].isna().all():
                df[dst] = df[src]
    ensure_time_s(df)
    ensure_ect_tft(df)
    ensure_rpm_aliases(df)
    # Derive 'brake' if still missing from Brake Pressure (kPa)
    if "brake" not in df.columns:
        for cand in ["Brake Pressure","Brake Pressure (SAE)","Brake Pressure (kPa)"]:
            if cand in df.columns:
                vals = pd.to_numeric(df[cand], errors="coerce")
                df["brake"] = (vals >= 15).astype(int)
                break

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaned-dir", required=True)
    args = ap.parse_args()
    base = Path(args.cleaned_dir)
    paths = sorted(base.glob("*.csv"))
    if not paths:
        print(f"[ERROR] No CSVs in {base}", file=sys.stderr); sys.exit(2)
    for p in paths:
        try:
            df = pd.read_csv(p, low_memory=False)
            before_cols = set(df.columns)
            promote(df)
            df.to_csv(p, index=False)
            print(f"[OK] Promoted canon aliases in {p.name}")
        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")
            sys.exit(2)
if __name__ == "__main__":
    main()
