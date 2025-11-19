#!/usr/bin/env python3
"""
hp_log_preclean.py - Stage -1 header pre-clean for raw HP Tuners CSV logs.

Usage:
  python hp_log_preclean.py --in-dir .\raw_logs --out-dir .\precleaned_logs

This script:
  - Detects HP Tuners CSV logs by header / section markers.
  - Builds a single CSV header row from [Channel Information] (PID + name).
  - Applies special handling for gear PIDs (14100 -> gear_actual, 4120 -> gear_cmd).
  - Ensures duplicate names are made unique by appending __PID<pid> (except special gear).
  - Writes data rows from [Channel Data] unchanged.
"""

import argparse
import csv
import os
from typing import List


def find_section(lines: List[str], marker: str) -> int:
    for i, line in enumerate(lines):
        if line.strip() == marker:
            return i
    raise ValueError(f"Marker {marker!r} not found")


def build_headers(pid_row: list, name_row: list) -> list:
    """
    Build canonical headers from PID + name rows.

    Special cases:
      - PID 14100 -> gear_actual
      - PID 4120  -> gear_cmd
    Other channels preserve the original name (stripped), with de-dup via __PID<pid>.
    """
    provisional = []
    for pid, raw_name in zip(pid_row, name_row):
        pid = pid.strip()
        raw_name = raw_name.strip()

        if pid == "14100":
            header = "gear_actual"
        elif pid == "4120":
            header = "gear_cmd"
        else:
            header = raw_name  # preserve original alpha name

        provisional.append((pid, header))

    seen = {}
    final_headers = []
    for pid, header in provisional:
        if header not in seen:
            seen[header] = 0
            final_headers.append(header)
        else:
            # Duplicate name: make unique by appending PID (except special gear headers).
            seen[header] += 1
            if header in ("gear_actual", "gear_cmd"):
                # For special gear channels, keep header as-is (these should already be unique PIDs).
                final_headers.append(header)
            else:
                final_headers.append(f"{header}__PID{pid}")
    return final_headers


def preclean_file(in_path: str, out_dir: str) -> None:
    with open(in_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if not lines or not lines[0].startswith("HP Tuners CSV Log File"):
        print(f"[SKIP] {in_path} (not an HP Tuners CSV log)")
        return

    try:
        ch_idx = find_section(lines, "[Channel Information]")
        data_idx = find_section(lines, "[Channel Data]")
    except ValueError as e:
        print(f"[SKIP] {in_path}: {e}")
        return

    # PID / Name / Units rows immediately after [Channel Information]
    if ch_idx + 2 >= len(lines):
        print(f"[SKIP] {in_path}: incomplete [Channel Information] section")
        return

    pid_line = lines[ch_idx + 1].strip()
    name_line = lines[ch_idx + 2].strip()
    # units_line = lines[ch_idx + 3].strip()  # currently unused

    pid_row = [p.strip() for p in pid_line.split(",")]
    name_row = [n.strip() for n in name_line.split(",")]

    if len(pid_row) != len(name_row):
        print(
            f"[WARN] {in_path}: PID and name row length mismatch "
            f"({len(pid_row)} vs {len(name_row)})"
        )

    headers = build_headers(pid_row, name_row)

    data_lines = lines[data_idx + 1 :]
    reader = csv.reader(data_lines)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(in_path))

    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(headers)
        for row in reader:
            # Skip completely empty rows.
            if not any(cell.strip() for cell in row):
                continue
            writer.writerow(row)

    print(f"[OK] Wrote precleaned file: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pre-clean HP Tuners CSV logs (header fix only, values unchanged)."
    )
    ap.add_argument(
        "--in-dir", required=True, help="Input directory with raw HPT CSV logs"
    )
    ap.add_argument(
        "--out-dir", required=True, help="Output directory for precleaned logs"
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for entry in os.scandir(args.in_dir):
        if not entry.is_file():
            continue
        if not entry.name.lower().endswith(".csv"):
            continue
        preclean_file(entry.path, args.out_dir)


if __name__ == "__main__":
    main()

