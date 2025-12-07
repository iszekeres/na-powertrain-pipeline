import argparse
import os
import sys
import pandas as pd
import numpy as np


def build_lat_lite(harsh_dir, out_path, min_speed, max_speed):
    events_path = os.path.join(harsh_dir, "shift_harshness_events.csv")
    if not os.path.exists(events_path):
        print(f"[ERROR] Missing file: {events_path}")
        sys.exit(1)

    df = pd.read_csv(events_path)

    required = ["from_gear", "to_gear", "speed_mph_pre"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("[ERROR] shift_harshness_events.csv is missing required columns:")
        for c in missing:
            print(f"  - {c}")
        print("Please adjust column names in this script to match your file.")
        sys.exit(1)

    df = df.copy()
    df["speed_mid_mph"] = pd.to_numeric(df["speed_mph_pre"], errors="coerce")
    df = df[(df["speed_mid_mph"] >= min_speed) & (df["speed_mid_mph"] < max_speed)].copy()

    df_up = df[df["to_gear"] > df["from_gear"]].copy()
    mask = df_up["from_gear"].isin([4, 5]) & df_up["to_gear"].isin([5, 6])
    df_up = df_up[mask].copy()

    if df_up.empty:
        print("[WARN] No 4->5 or 5->6 upshift events found in the selected speed range.")
        return

    jerk_cols = [c for c in df_up.columns if "jerk" in c.lower()]

    def q90(x):
        return float(np.nanpercentile(x, 90)) if len(x) else np.nan

    group_cols = ["from_gear", "to_gear"]
    agg = df_up.groupby(group_cols)["speed_mid_mph"].agg(
        count="count",
        mean_speed_mph="mean",
        p90_speed_mph=q90,
    ).reset_index()

    for jc in jerk_cols:
        g = df_up.groupby(group_cols)[jc]
        agg[f"{jc}_median"] = g.median().values
        agg[f"{jc}_p90"] = g.apply(q90).values

    if "max_abs_jerk" in df_up.columns:
        g2 = df_up.groupby(group_cols)["max_abs_jerk"].agg(median_shift_intensity="median", p90_shift_intensity=q90).reset_index()
        agg = agg.merge(g2, on=group_cols, how="left")
    if "pedal_pct_ref" in df_up.columns:
        g3 = df_up.groupby(group_cols)["pedal_pct_ref"].mean().reset_index().rename(columns={"pedal_pct_ref": "mean_pedal_pct"})
        agg = agg.merge(g3, on=group_cols, how="left")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    agg.to_csv(out_path, index=False)

    print(f"[OK] Wrote LAT-lite summary to {out_path}")
    print(agg.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Highway LAT-lite / harshness summary for 4-5-6 (NOCRUISE).")
    parser.add_argument("--harsh-dir", default=r"highway_super_analysis__HARSHNESS__NOCRUISE")
    parser.add_argument("--out", default=r"highway_super_analysis__HARSHNESS__NOCRUISE\highway_LAT_lite__NOCRUISE.csv")
    parser.add_argument("--min-speed", type=float, default=60.0)
    parser.add_argument("--max-speed", type=float, default=90.0)
    args = parser.parse_args()

    build_lat_lite(args.harsh_dir, args.out, args.min_speed, args.max_speed)


if __name__ == "__main__":
    main()
