import argparse
import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np


REQUIRED_COLS = [
    "speed_mph",
    "pedal_pct",
    "throttle_pct",
    "gear_actual",
    "brake",
    "time_s",
]


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_mapping(df, path):
    mapping = {}
    # speed
    speed_col = pick_col(df, ["speed_mph", "Vehicle Speed (SAE)", "Vehicle Speed"])
    if speed_col:
        mapping[speed_col] = "speed_mph"
    # pedal
    pedal_col = pick_col(df, ["pedal_pct", "Accelerator Pedal Position"])
    if pedal_col:
        mapping[pedal_col] = "pedal_pct"
    # throttle
    throttle_col = pick_col(df, ["throttle_pct", "Throttle Position", "Throttle Position (SAE) %"])
    if throttle_col:
        mapping[throttle_col] = "throttle_pct"
    # gear
    gear_col = pick_col(df, ["gear_actual", "gear_actual__canon", "Trans Current Gear"])
    if gear_col:
        mapping[gear_col] = "gear_actual"
    # brake (binary or pressure)
    brake_col = pick_col(df, ["brake", "brake_on", "Brake", "Brake Pressure", "brake_kpa"])
    if brake_col:
        mapping[brake_col] = "brake"
    # time
    time_col = pick_col(df, ["time_s", "Time", "time"])
    if time_col:
        mapping[time_col] = "time_s"

    present = set(mapping.values())
    present.update([c for c in REQUIRED_COLS if c in df.columns])
    missing = [c for c in REQUIRED_COLS if c not in present]
    if missing:
        print(f"[ERROR] {os.path.basename(path)} is missing required columns:")
        for c in missing:
            print(f"   - {c}")
        return None
    return mapping


def build_nocruise(df):
    # Make sure types are sane
    for col in ["speed_mph", "pedal_pct", "throttle_pct", "gear_actual", "brake", "time_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute speed derivative (mph per second)
    dt = df["time_s"].diff()
    ds = df["speed_mph"].diff()
    ds_dt = ds / dt
    ds_dt.replace([np.inf, -np.inf], np.nan, inplace=True)
    ds_dt = ds_dt.fillna(0.0)

    # Cruise detection thresholds (can tweak later)
    pedal_thresh = 1.0      # % pedal
    tps_min_for_cruise = 6  # % throttle blade
    speed_slope_max = 0.1   # mph/s, "basically constant speed"

    cruise_mask = (
        (df["pedal_pct"] < pedal_thresh) &
        (df["throttle_pct"] >= tps_min_for_cruise) &
        (df["gear_actual"] >= 4) &
        (df["brake"] == 0) &
        (ds_dt.abs() <= speed_slope_max)
    )

    # For now: we simply drop cruise rows
    kept = df.loc[~cruise_mask].copy()

    return kept, cruise_mask.sum(), len(df)


def main():
    ap = argparse.ArgumentParser(
        description="Build __NOCRUISE versions of prepped highway logs."
    )
    ap.add_argument(
        "--prepped-dir",
        required=True,
        help="Directory containing *__prepped.csv files (e.g. newlogs\\highway_MAX_analysis\\prepped)",
    )
    ap.add_argument(
        "--out-suffix",
        default="__NOCRUISE",
        help="Suffix to append before .csv (default: __NOCRUISE)",
    )
    args = ap.parse_args()

    prepped_dir = args.prepped_dir

    if not os.path.isdir(prepped_dir):
        print(f"[ERROR] Prepped dir not found: {prepped_dir}")
        sys.exit(1)

    any_error = False

    for name in sorted(os.listdir(prepped_dir)):
        if not name.lower().endswith(".csv"):
            continue
        in_path = os.path.join(prepped_dir, name)
        print(f"\n[INFO] Processing {in_path} ...")

        try:
            header_df = pd.read_csv(in_path, nrows=0)
            mapping = build_mapping(header_df, in_path)
            if mapping is None:
                any_error = True
                continue
            # Determine columns to load: original names used in mapping plus any already-standard required cols
            usecols = set(mapping.keys())
            usecols.update([c for c in REQUIRED_COLS if c in header_df.columns])

            chunks = pd.read_csv(in_path, usecols=list(usecols), low_memory=False, chunksize=200000)
            kept_parts = []
            prev_tail = None
            n_cruise_total = 0
            n_total_rows = 0
            for chunk_idx, chunk in enumerate(chunks):
                df_chunk = chunk.rename(columns=mapping)
                if prev_tail is not None:
                    df_chunk = pd.concat([prev_tail, df_chunk], ignore_index=True)
                kept, n_cruise, n_total = build_nocruise(df_chunk)
                n_cruise_total += n_cruise
                n_total_rows += n_total
                if len(kept) == 0:
                    prev_tail = None
                    continue
                # hold last row for continuity with next chunk
                prev_tail = kept.iloc[[-1]].copy()
                kept_parts.append(kept.iloc[:-1])
            # append final tail if present
            if prev_tail is not None:
                kept_parts.append(prev_tail)
            if any_error:
                continue
            kept_all = pd.concat(kept_parts, ignore_index=True) if kept_parts else pd.DataFrame()
        except Exception as e:
            print(f"[ERROR] Failed to read {name}: {e}")
            any_error = True
            continue

        base, ext = os.path.splitext(in_path)
        out_path = f"{base}{args.out_suffix}{ext}"

        kept_all.to_csv(out_path, index=False)

        print(f"[INFO] {name}: total rows = {n_total_rows}, cruise rows dropped = {n_cruise_total}, kept = {len(kept_all)}")
        print(f"[INFO] Wrote NOCRUISE file: {out_path}")

    if any_error:
        print("\n[DONE] Completed with errors. See messages above for missing headers or read failures.")
        sys.exit(1)
    else:
        print("\n[DONE] All files processed successfully.")


if __name__ == "__main__":
    main()
