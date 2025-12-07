import argparse
import os
import sys
import pandas as pd
import numpy as np

def norm_band(b):
    if pd.isna(b):
        return None
    s = str(b)
    if "-" in s and "[" not in s and "," not in s:
        return s
    s = s.strip()
    s = s.lstrip("[").rstrip(")")
    if "," in s:
        lo, hi = s.split(",", 1)
        return f"{lo.strip()}-{hi.strip()}"
    return s

def low_edge(bin_str):
    try:
        return float(str(bin_str).split("-")[0])
    except Exception:
        return np.nan

def build_overlay(super_dir, out_path, min_speed, max_speed):
    occ_path = os.path.join(super_dir, "highway_speed_pedal_occupancy.csv")
    td_path  = os.path.join(super_dir, "highway_torque_deficit_integral__by_bin.csv")
    vs_path  = os.path.join(super_dir, "virtual_schedule__bin_summary.csv")
    ifs_path = os.path.join(super_dir, "highway_intent_frustration__summary.csv")

    for p in [occ_path, td_path, vs_path, ifs_path]:
        if not os.path.exists(p):
            print(f"[ERROR] Missing file: {p}")
            sys.exit(1)

    occ = pd.read_csv(occ_path)
    td  = pd.read_csv(td_path)
    vs  = pd.read_csv(vs_path)
    ifs = pd.read_csv(ifs_path)

    occ = occ[occ["mode"] == "pattern a"].copy()
    td  = td[td["mode"]  == "pattern a"].copy()
    vs  = vs[vs["mode"]  == "pattern a"].copy()
    ifs = ifs[ifs["mode"] == "pattern a"].copy()

    occ = occ.dropna(subset=["speed_bin_mph", "pedal_bin_pct"]).copy()
    occ["speed_low"] = occ["speed_bin_mph"].apply(low_edge)
    occ = occ[(occ["speed_low"] >= min_speed) & (occ["speed_low"] < max_speed)].copy()

    key_cols = ["speed_bin_mph", "pedal_bin_pct"]

    td_small = td[key_cols + ["torque_deficit_integral", "mean_delta_T"]].copy()
    vs_small = vs[key_cols + [
        "mean_T_axle_actual",
        "mean_T_axle_virtual",
        "mean_delta_T_axle",
        "p95_delta_T_axle",
        "dominant_gear_virtual",
    ]].copy()

    ifs = ifs.copy()
    ifs["speed_bin_mph"] = ifs["speed_band"].apply(norm_band)
    ifs["pedal_bin_pct"] = ifs["pedal_band"].apply(norm_band)
    ifs_small = ifs[[
        "speed_bin_mph",
        "pedal_bin_pct",
        "intent_strength",
        "n_episodes",
        "n_frustrated",
        "frac_frustrated",
        "median_torque_gap_pct",
        "median_dv5_mph",
    ]].copy()

    merged = occ.merge(td_small, on=key_cols, how="left", suffixes=("", "_td"))
    merged = merged.merge(vs_small, on=key_cols, how="left", suffixes=("", "_vs"))
    merged = merged.merge(ifs_small, on=key_cols, how="left")

    def classify(row):
        occ_pct = row.get("time_pct", 0.0)
        gap     = row.get("mean_delta_T", np.nan)
        frac    = row.get("frac_frustrated", np.nan)
        occ_pct = 0.0 if pd.isna(occ_pct) else float(occ_pct)
        gap     = 0.0 if pd.isna(gap)     else float(gap)
        frac    = 0.0 if pd.isna(frac)    else float(frac)
        if occ_pct < 1e-4:
            return "unused"
        score = (gap / 50.0) + frac * 2.0
        if score < 0.8:
            return "ok"
        elif score < 1.5:
            return "warn"
        else:
            return "hot"

    merged["severity"] = merged.apply(classify, axis=1)
    merged = merged.sort_values(["speed_low", "pedal_bin_pct"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)

    print(f"[OK] Wrote overlay map to {out_path}")
    print(merged[[
        "speed_bin_mph",
        "pedal_bin_pct",
        "time_pct",
        "mean_delta_T",
        "median_torque_gap_pct",
        "frac_frustrated",
        "dominant_gear_virtual",
        "severity",
    ]].head(20).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Build NOCRUISE highway overlay map (occupancy + torque deficit + intent).")
    parser.add_argument("--super-dir", default=r"newlogs\highway_MAX_extracted\highway_super_analysis__NOCRUISE")
    parser.add_argument("--out", default=r"newlogs\highway_MAX_extracted\highway_super_analysis__NOCRUISE\highway_overlay_map__NOCRUISE.csv")
    parser.add_argument("--min-speed", type=float, default=60.0)
    parser.add_argument("--max-speed", type=float, default=90.0)
    args = parser.parse_args()

    build_overlay(args.super_dir, args.out, args.min_speed, args.max_speed)

if __name__ == "__main__":
    main()
