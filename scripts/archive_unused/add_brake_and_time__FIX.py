#!/usr/bin/env python3
# add_brake_and_time.py (FIXED HELP STRINGS, QUIET DOCSTRING)
# Adds:
#   - time_s (from raw 'offset' if time_s missing)
#   - brake (boolean) derived from any brake source found, preferring pressure
#
# Inputs: one or more CSVs (typically __trans_focus__clean_FULL__*.csv)
# Outputs: new CSVs with suffixes:
#   - __withtime (only if time_s was added)
#   - __withbrake (only if brake was added)
#
# Examples:
#   python .\add_brake_and_time.py --in-glob ".\newlogs\cleaned\__trans_focus__clean_FULL__*.csv" --out-dir ".\newlogs\cleaned"
#   python .\add_brake_and_time.py --in-glob ".\newlogs\cleaned\__trans_focus__clean_FULL__*outbound*.csv" --brake-thresh-kpa 15

import argparse, glob, sys, os
import pandas as pd
import numpy as np
from pathlib import Path

def pick_brake_source(df):
    cols = list(df.columns)
    # pressure first
    press_cols = [c for c in cols if ('brake' in c.lower()) and ('press' in c.lower())]
    # alternates
    switch_cols = [c for c in cols if ('brake' in c.lower()) and (('switch' in c.lower()) or ('applied' in c.lower()) or ('on' in c.lower())) and ('press' not in c.lower())]
    pedal_cols  = [c for c in cols if ('brake' in c.lower()) and (('pedal' in c.lower()) or ('position' in c.lower())) and ('press' not in c.lower())]

    def best(col_list):
        if not col_list: return None
        nn = [(c, int(pd.Series(df[c]).notna().sum())) for c in col_list]
        nn.sort(key=lambda t: t[1], reverse=True)
        return nn[0][0]

    return best(press_cols), best(switch_cols), best(pedal_cols)

def derive_brake_series(df, press_col, switch_col, pedal_col, thr_kpa, thr_psi, thr_pedal_pct):
    # pressure logic
    if press_col is not None:
        s = pd.to_numeric(df[press_col], errors='coerce')
        name = press_col.lower()
        # unit hint from name
        if 'kpa' in name:
            thr = thr_kpa
        elif 'psi' in name:
            thr = thr_psi
        else:
            m = float(np.nanmax(s.values)) if s.size else 0.0
            thr = thr_kpa if m > 200 else thr_psi
        return (s >= thr).astype('int32'), f"pressure:{press_col} (thr={thr})"

    # switch logic
    if switch_col is not None:
        s = df[switch_col].astype(str).str.strip().str.lower()
        truthy = {'1','true','on','pressed','applied','yes'}
        return s.isin(truthy).astype('int32'), f"switch:{switch_col}"

    # pedal percent logic
    if pedal_col is not None:
        s = pd.to_numeric(df[pedal_col], errors='coerce')
        return (s >= thr_pedal_pct).astype('int32'), f"pedal:{pedal_col} (thr={thr_pedal_pct}pct)"

    return None, None

def main():
    ap = argparse.ArgumentParser(description="Add time_s and brake to CLEAN_FULL CSVs")
    ap.add_argument("--in-glob", required=True, help=r"Glob of input CSVs (e.g., .\newlogs\cleaned\__trans_focus__clean_FULL__*.csv)")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: same folder as input file)")
    ap.add_argument("--brake-thresh-kpa", type=float, default=15.0, help="Threshold for kPa pressure >= => brake=1 (default 15.0)".replace("%","%%"))
    ap.add_argument("--brake-thresh-psi", type=float, default=2.2,  help="Threshold for psi pressure >= => brake=1 (default 2.2)".replace("%","%%"))
    ap.add_argument("--brake-pedal-thresh-pct", type=float, default=5.0, help="Threshold for brake pedal percent >= => brake=1 (default 5.0)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.in_glob))
    if not files:
        print(f"[add_brake_and_time] No files matched {args.in_glob}")
        sys.exit(0)

    for inp in files:
        try:
            df = pd.read_csv(inp)
        except Exception as e:
            print(f"[add_brake_and_time] READ FAIL {inp}: {e}")
            continue

        added_time = False
        added_brake = False

        # time_s
        if 'time_s' not in df.columns:
            if 'offset' in df.columns:
                ts = pd.to_numeric(df['offset'], errors='coerce')
                df['time_s'] = ts
                added_time = True
            else:
                print(f"[add_brake_and_time] {Path(inp).name}: no 'time_s' and no 'offset' to derive from")

        # brake
        if 'brake' not in df.columns:
            press_col, switch_col, pedal_col = pick_brake_source(df)
            br, meta = derive_brake_series(df, press_col, switch_col, pedal_col, args.brake_thresh_kpa, args.brake_thresh_psi, args.brake_pedal_thresh_pct)
            if br is not None:
                df['brake'] = br
                added_brake = True
                print(f"[add_brake_and_time] {Path(inp).name}: brake from {meta} | nonzeros={int(br.sum())}/{len(br)}")
            else:
                print(f"[add_brake_and_time] {Path(inp).name}: could not derive brake (no pressure/switch/pedal source found)")
        else:
            print(f"[add_brake_and_time] {Path(inp).name}: brake already present")

        # Decide output name
        out_dir = Path(args.out_dir) if args.out_dir else Path(inp).parent
        base = Path(inp).name
        suffixes = []
        if added_time: suffixes.append("withtime")
        if added_brake: suffixes.append("withbrake")
        if suffixes:
            stem, ext = os.path.splitext(base)
            out = out_dir / f"{stem}__{'__'.join(suffixes)}{ext}"
        else:
            stem, ext = os.path.splitext(base)
            out = out_dir / f"{stem}__passthrough{ext}"

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(f"[add_brake_and_time] WROTE {out}")
        except Exception as e:
            print(f"[add_brake_and_time] WRITE FAIL {out}: {e}")

if __name__ == "__main__":
    main()
