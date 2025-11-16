"""
tcc_pack_force_1dp.py

Final-pack TCC 1-decimal fixer for NA_Trans.

- Operates on a TCC tables directory (default:
    .\newlogs\output\03_flash_pack\LOGFIRST_MODEB_SMOOTH_XGEAR\01_tables\tcc
  )
- For each TSV file, leaves the header line as-is.
- For data rows:
    * Converts numeric-looking cells to floats.
    * If value is ~317 or ~318, writes "317" or "318" (no decimals).
    * Otherwise writes with exactly one decimal place (e.g. "29.7").
    * Empty cells remain empty.
"""

import os
import sys
import csv
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

DEFAULT_DIR = os.path.join(
    "newlogs",
    "output",
    "03_flash_pack",
    "LOGFIRST_MODEB_SMOOTH_XGEAR",
    "01_tables",
    "tcc",
)


def snap_val(txt: str) -> str:
    if txt is None:
        return ""
    s = str(txt).strip()
    if s == "":
        return ""
    # Preserve exact sentinels if already "317" or "318"
    if s in ("317", "318"):
        return s
    # Try to treat as Decimal for robust 1dp rounding
    try:
        d = Decimal(s)
    except InvalidOperation:
        # Non-numeric, return as-is
        return s
    # Sentinel detection with tolerance
    if (d - Decimal("317")).copy_abs() < Decimal("0.000001"):
        return "317"
    if (d - Decimal("318")).copy_abs() < Decimal("0.000001"):
        return "318"
    # Normal value: 1 decimal place
    d = d.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    # Always emit with one decimal place
    return f"{d:.1f}"


def fix_tsv(path: str) -> None:
    print(f"[TCC_1DP] Fixing {path}")
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))

    if not rows:
        print(f"[TCC_1DP] Empty file, skipping: {path}")
        return

    out_rows = []
    # Header: keep exactly as-is
    out_rows.append(rows[0])

    # Data rows: fix numeric cells
    for row in rows[1:]:
        if not row:
            out_rows.append(row)
            continue
        new_row = row[:]
        # Start from col 1 (col 0 is row label like "3rd Release")
        for i in range(1, len(new_row)):
            new_row[i] = snap_val(new_row[i])
        out_rows.append(new_row)

    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerows(out_rows)
    print(f"[TCC_1DP] Wrote fixed file: {path}")


def main(dir_path: str = None) -> None:
    if dir_path is None:
        dir_path = DEFAULT_DIR
    dir_path = os.path.abspath(dir_path)
    print(f"[TCC_1DP] Using TCC dir: {dir_path}")
    if not os.path.isdir(dir_path):
        print(f"[ERR] TCC dir not found: {dir_path}")
        sys.exit(1)

    # Fix any TSV in this directory
    any_fixed = False
    for name in os.listdir(dir_path):
        if not name.lower().endswith(".tsv"):
            continue
        path = os.path.join(dir_path, name)
        fix_tsv(path)
        any_fixed = True

    if not any_fixed:
        print(f"[WARN] No .tsv files found in {dir_path}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)

