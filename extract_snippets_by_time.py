#!/usr/bin/env python3
"""
extract_snippets_by_time.py - Extract time-window snippets from CLEAN_FULL files.

Usage examples:

  # Extract rows where  50.0 <= time_s <=  70.0
  python extract_snippets_by_time.py ^
      --in  .\newlogs\cleaned\__trans_focus__clean_FULL__foo.csv ^
      --out .\newlogs\cleaned\snippet_foo_50_70.csv ^
      --t-start 50.0 ^
      --t-end   70.0

This script:
  - Reads a CSV (typically CLEAN_FULL) that has a 'time_s' column.
  - Filters rows where t_start <= time_s <= t_end.
  - Writes the snippet to the requested output path.
"""

import argparse
import csv
import os
from typing import Optional


def extract_snippet(
    in_path: str, out_path: str, t_start: float, t_end: float
) -> None:
    if t_end < t_start:
        raise SystemExit(f"[ERROR] t_end ({t_end}) < t_start ({t_start})")

    if not os.path.exists(in_path):
        raise SystemExit(f"[ERROR] Input file not found: {in_path}")

    with open(in_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise SystemExit(f"[ERROR] Empty CSV: {in_path}")

        if "time_s" not in header:
            raise SystemExit(
                f"[ERROR] 'time_s' column not found in {in_path}; "
                "expected CLEAN_FULL-style input."
            )

        time_idx = header.index("time_s")

        rows_out = []
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            if time_idx >= len(row):
                continue
            try:
                t = float(row[time_idx])
            except (TypeError, ValueError):
                continue
            if t_start <= t <= t_end:
                rows_out.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)
        writer.writerows(rows_out)

    print(
        f"[OK] Wrote snippet {len(rows_out)} row(s) from {in_path} "
        f"into {out_path} for {t_start} <= time_s <= {t_end}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract a time_s window snippet from a CLEAN_FULL CSV."
    )
    ap.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input CLEAN_FULL CSV (must contain 'time_s' column).",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output CSV for the extracted snippet.",
    )
    ap.add_argument(
        "--t-start",
        type=float,
        required=True,
        help="Start time (seconds, inclusive).",
    )
    ap.add_argument(
        "--t-end",
        type=float,
        required=True,
        help="End time (seconds, inclusive).",
    )
    args = ap.parse_args()

    extract_snippet(args.in_path, args.out_path, args.t_start, args.t_end)


if __name__ == "__main__":
    main()

