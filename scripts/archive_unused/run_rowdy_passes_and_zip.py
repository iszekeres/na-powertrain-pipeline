#!/usr/bin/env python3
"""Run the Rowdy refinement passes and bundle the resulting delta tables."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

PASS_SCRIPTS = [
    "tools/intent_pass_rowdy.py",
    "tools/stopgo_pass_rowdy.py",
    "tools/consist_pass_rowdy.py",
    "tools/kickdown_pass_rowdy.py",
    "tools/tcc_polish_rowdy.py",
]

OUTPUT_DIR = Path("newlogs/output")
ZIP_PATH = OUTPUT_DIR / "ROWDY__DELTAS__from_logs.zip"

DELTA_FILES = [
    Path("newlogs/output/02_passes/INTENT/INTENT__ROWDY__SHIFT_UP__DELTA.tsv"),
    Path("newlogs/output/02_passes/INTENT/INTENT__ROWDY__TCC_RELEASE__DELTA.tsv"),
    Path("newlogs/output/02_passes/STOPGO/STOPGO__ROWDY__SHIFT_DOWN__DELTA.tsv"),
    Path("newlogs/output/02_passes/CONSIST/CONSIST__ROWDY__SHIFT_UP__DELTA.tsv"),
    Path("newlogs/output/02_passes/CONSIST/CONSIST__ROWDY__SHIFT_DOWN__DELTA.tsv"),
    Path("newlogs/output/02_passes/KICKDOWN/KICKDOWN__ROWDY__SHIFT_DOWN__DELTA.tsv"),
    Path("newlogs/output/02_passes/TCC_POLISH/TCC_POLISH__ROWDY__APPLY__DELTA.tsv"),
    Path("newlogs/output/02_passes/TCC_POLISH/TCC_POLISH__ROWDY__RELEASE__DELTA.tsv"),
]


def run_passes() -> None:
    for script in PASS_SCRIPTS:
        print(f"Running {script} ...")
        subprocess.run(["python", script], check=True)


def bundle_deltas() -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    included: list[Path] = []
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in DELTA_FILES:
            if file.exists():
                zf.write(file, arcname=file.name)
                included.append(file)
    return included


def main() -> None:
    run_passes()
    included = bundle_deltas()
    if included:
        print(f"Packaged Rowdy deltas into {ZIP_PATH}")
        for path in included:
            print(f"  - {path}")
    else:
        print("No delta files found to include in the ZIP.")


if __name__ == "__main__":
    main()
