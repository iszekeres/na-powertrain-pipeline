#!/usr/bin/env python3
"""
Runner that cleans and analyzes the indy trip logs using the new NA Trans scripts.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run NA Trans cleaning for the indy trip session.")
    parser.add_argument("--session", default="indy trip", help="Session name under logs_raw/logs_processed.")
    args = parser.parse_args()

    repo_root = Path.cwd()
    session = args.session
    raw_dir = repo_root / "logs_raw" / session
    cleaned_dir = repo_root / "logs_processed" / session / "cleaned"
    analyze_dir = repo_root / "logs_processed" / session / "output" / "00_cleaner"

    print("[run] session =", session)
    print("      raw_dir     =", raw_dir)
    print("      cleaned_dir =", cleaned_dir)
    print("      analyze_out =", analyze_dir)

    if not raw_dir.exists():
        print(f"[error] missing raw directory: {raw_dir}")
        sys.exit(1)

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        print(f"[error] no CSV files found in {raw_dir}")
        sys.exit(1)

    cleaned_dir.mkdir(parents=True, exist_ok=True)
    analyze_dir.mkdir(parents=True, exist_ok=True)

    total = len(csv_files)
    for idx, raw_file in enumerate(csv_files, start=1):
        cleaned_name = f"{raw_file.stem}__clean_full.csv"
        cleaned_path = cleaned_dir / cleaned_name
        summary_target = analyze_dir / f"__trans_focus__summary__{cleaned_name}.txt"
        if cleaned_path.exists() and summary_target.exists():
            print(f"[{idx}/{total}] skip {raw_file.name} (already processed)")
            continue
        print(f"[{idx}/{total}] clean {raw_file.name} -> {cleaned_name}")
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "scripts.pipeline.clean_log_NA",
                    "--in-file",
                    str(raw_file),
                    "--out-file",
                    str(cleaned_path),
                ]
            )
        except subprocess.CalledProcessError as exc:
            print(f"[error] cleaner failed for {raw_file.name}: {exc}")
            sys.exit(exc.returncode)

        print(f"[{idx}/{total}] analyze {cleaned_name}")
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "scripts.pipeline.trans_clean_analyze",
                    "--in-file",
                    str(cleaned_path),
                    "--out-dir",
                    str(analyze_dir),
                ]
            )
        except subprocess.CalledProcessError as exc:
            print(f"[error] analyzer failed for {cleaned_name}: {exc}")
            sys.exit(exc.returncode)

    print("[done] indy trip logs cleaned and analyzed.")
    print(f"       FULL cleaned logs:   {cleaned_dir}")
    print(f"       Trans-focus outputs: {analyze_dir}")


if __name__ == "__main__":
    main()
