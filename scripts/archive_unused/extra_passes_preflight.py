#!/usr/bin/env python3
# extra_passes_preflight.py
# Inspect CLEAN files and report, per "weighted pass," whether required columns exist
# and whether there's enough qualifying data to produce non-empty outputs.
#
# Usage:
#   python extra_passes_preflight.py --review-dir ".\06_Logs\Trans_Review" --glob "__trans_focus__clean__*headerfix*.csv"

import os, glob, argparse
import pandas as pd
import numpy as np

PASSES = {
    "driver_intent": {
        "needs": ["speed_mph","throttle_pct","pedal_pct","gear_actual","time_s"],
        "notes": "Needs real throttle/pedal movement; looks for intent surges. Check dTPS/dt and mid TPS bins."
    },
    "corner_exit": {
        "needs": ["speed_mph","throttle_pct","gear_actual"],
        "notes": "Looks for throttle pickup after low-speed/high-gear corners."
    },
    "occupancy": {
        "needs": ["speed_mph","throttle_pct","gear_actual"],
        "notes": "Counts bin occupancy; empty if coverage is thin in many TPS/speed bins."
    },
    "rpm_floor_guard": {
        "needs": ["speed_mph","throttle_pct","gear_actual","engine_rpm","turbine_rpm"],
        "notes": "Protects against lugging; needs RPMs."
    },
    "shift_consistency": {
        "needs": ["speed_mph","throttle_pct","gear_actual"],
        "notes": "Assesses scatter; needs many repeats per pair/TPS bin."
    },
    "shift_latency": {
        "needs": ["speed_mph","throttle_pct","gear_actual","gear_cmd","time_s"],
        "notes": "Measures commanded->actual latency; requires both gear_cmd and gear_actual with time."
    },
    "stopngo": {
        "needs": ["speed_mph","throttle_pct","gear_actual","brake","time_s"],
        "notes": "Focuses on low-speed launch transitions and brake release; check speed<=12 mph samples."
    },
    "engine_brake_downhill": {
        "needs": ["speed_mph","throttle_pct","gear_actual","brake","time_s"],
        "notes": "Looks for foot-off downhill decel; brake often 0; throttle near 0."
    },
    "traction_softener": {
        "needs": ["speed_mph","throttle_pct","gear_actual","time_s"],
        "notes": "Looks for wheelspin hints; needs accel/jerk proxies (time continuity)."
    },
    "dfco_helper": {
        "needs": ["speed_mph","throttle_pct","gear_actual"],
        "notes": "Zero/near-zero fuel throttle-off decel; often requires gear>=3."
    },
}

def dcount(x): return int(pd.notna(x).sum())

def preflight_file(fp):
    try:
        df = pd.read_csv(fp, low_memory=False)
    except Exception:
        df = pd.read_csv(fp, low_memory=False, engine="python", on_bad_lines="skip")
    cols = set(df.columns)
    stats = {"rows": len(df)}
    # Cheap derivatives
    if "time_s" in cols:
        t = pd.to_numeric(df["time_s"], errors="coerce")
        dt = t.diff().clip(lower=1e-6)
        if "throttle_pct" in cols:
            thr = pd.to_numeric(df["throttle_pct"], errors="coerce").clip(0,100)
            dthr = thr.diff().abs() / dt
            stats["thr_moves_per_s>20"] = int((dthr > 20).sum())
        # Low-speed samples
        if "speed_mph" in cols:
            v = pd.to_numeric(df["speed_mph"], errors="coerce")
            stats["low_speed_rows<=12mph"] = int((v <= 12).sum())
    # Non-null per common columns
    for c in ["speed_mph","throttle_pct","pedal_pct","gear_actual","gear_cmd","engine_rpm","turbine_rpm","brake","time_s","tcc_locked_built","trans_fluid_temp_c"]:
        if c in cols:
            stats[f"nn__{c}"] = dcount(df[c])
    return stats, df.columns.tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-dir", default=r".\06_Logs\Trans_Review")
    ap.add_argument("--glob", default="__trans_focus__clean__*.csv")
    args = ap.parse_args()

    pattern = os.path.join(args.review_dir, args.glob)
    files = sorted(glob.glob(pattern))
    if not files:
        print("No CLEAN files found for pattern:", pattern)
        return

    print("Review dir:", args.review_dir)
    print("Files:")
    for f in files:
        print(" -", os.path.basename(f))

    # Check per-file stats
    aggregate = {k:0 for k in ["rows","thr_moves_per_s>20","low_speed_rows<=12mph"]}
    file_stats = {}
    for f in files:
        s, cols = preflight_file(f)
        file_stats[f] = s
        for k in aggregate:
            aggregate[k] += s.get(k, 0)

    print("\n=== Per-file coverage ===")
    for f,s in file_stats.items():
        print("\n", os.path.basename(f))
        for k,v in s.items():
            print(f"  {k:>22s}: {v}")

    # Pass-by-pass readiness
    print("\n=== Pass readiness checks ===")
    all_cols = set()
    for s in file_stats.values():
        pass
    # Union columns across files
    col_union = set()
    for f in files:
        try:
            df = pd.read_csv(f, nrows=0)
            col_union.update(df.columns)
        except Exception:
            pass

    for name, spec in PASSES.items():
        missing = [c for c in spec["needs"] if c not in col_union]
        print(f"\n{name:24s} needs ->", ", ".join(spec["needs"]))
        if missing:
            print("  MISSING headers:", ", ".join(missing))
        else:
            print("  Headers OK")
        # Quick gating hints
        if name=="driver_intent":
            print(f"  throttle surges (>20%/s) total: {aggregate.get('thr_moves_per_s>20',0)}")
        if name=="stopngo":
            print(f"  low-speed rows (<=12 mph) total: {aggregate.get('low_speed_rows<=12mph',0)}")
        print("  note:", spec["notes"])

if __name__ == "__main__":
    main()
