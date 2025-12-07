#!/usr/bin/env python3
"""
DFCO-free cruise pedal scan.

Reads a single HP Tuners CSV, finds the header line, and computes cruise-ish pedal
stats without relying on DFCO flags (since DFCO is disabled).
"""
import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def find_header_row(path: Path, max_rows: int = 50) -> int:
    with path.open("r", errors="ignore") as f:
        for i in range(max_rows):
            line = f.readline()
            if not line:
                break
            if "," in line and "Time" in line:
                return i
    return 0


def load_log(path: Path) -> pd.DataFrame:
    header = find_header_row(path)
    df = pd.read_csv(path, skiprows=header, low_memory=False)
    return df


def summarize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    cols = ["speed_mph", "pedal_pct", "throttle_pct"]
    stats = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        stats[f"{label}__{c}_count"] = int(s.count())
        stats[f"{label}__{c}_mean"] = float(s.mean())
        stats[f"{label}__{c}_median"] = float(s.median())
        stats[f"{label}__{c}_min"] = float(s.min())
        stats[f"{label}__{c}_max"] = float(s.max())
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            stats[f"{label}__{c}_p{int(q*100)}"] = float(s.quantile(q))
    return pd.DataFrame([stats])


def main():
    ap = argparse.ArgumentParser(description="DFCO-free cruise pedal scan")
    ap.add_argument("--log", required=True, help="Path to raw HP Tuners CSV")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    args = ap.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_log(log_path)

    # Map columns
    col_map = {
        "Vehicle Speed (SAE)": "speed_mph",
        "Accelerator Pedal Position": "pedal_pct",
        "Throttle Position": "throttle_pct",
        "Brake Pressure": "brake_kpa",
    }
    df = pd.DataFrame()
    for src, dst in col_map.items():
        if src not in df_raw.columns:
            raise SystemExit(f"[ERROR] Missing required column '{src}' in {log_path}")
        df[dst] = pd.to_numeric(df_raw[src], errors="coerce")

    df["speed_diff"] = df["speed_mph"].diff()

    cruise_mask = (
        (df["speed_mph"] >= 30) & (df["speed_mph"] <= 50) &
        (df["brake_kpa"] < 15) &
        (df["pedal_pct"] > 2) & (df["pedal_pct"] < 30)
    )
    steady_mask = df["speed_diff"].abs() < 0.05
    use = df[cruise_mask & steady_mask].copy()

    bands = []
    for lo, hi in [(30, 40), (40, 50)]:
        band = use[(use["speed_mph"] >= lo) & (use["speed_mph"] < hi)].copy()
        band_stats = summarize(band, f"band_{lo}_{hi}")
        bands.append(band_stats)

    overall = summarize(use, "overall")
    summary_df = pd.concat([overall] + bands, axis=1)

    out_csv = out_dir / "CRUISE_PEDAL_STATS__HAHA1.csv"
    summary_df.to_csv(out_csv, index=False)

    lines = []
    lines.append(f"CRUISE PEDAL STATS (DFCO-free)\nLog: {log_path.name}\n")
    lines.append(f"Rows considered (cruise-ish & steady): {len(use)}")
    for prefix in ["overall", "band_30_40", "band_40_50"]:
        lines.append(f"\n[{prefix}]")
        for col in ["speed_mph", "pedal_pct", "throttle_pct"]:
            for k in ["count", "mean", "median", "min", "max", "p10", "p25", "p50", "p75", "p90"]:
                key = f"{prefix}__{col}_{k}"
                val = summary_df.iloc[0][key]
                lines.append(f"{key}: {val:.3f}" if isinstance(val, (int, float, np.floating)) else f"{key}: {val}")
    out_txt = out_dir / "CRUISE_PEDAL_STATS__HAHA1.txt"
    out_txt.write_text("\n".join(lines))

    print(f"[OK] Wrote {out_csv}")
    print(f"[OK] Wrote {out_txt}")


if __name__ == "__main__":
    main()
