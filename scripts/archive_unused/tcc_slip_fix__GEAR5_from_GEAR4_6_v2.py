#!/usr/bin/env python
"""
tcc_slip_fix__GEAR5_from_GEAR4_6_v2.py

Average the Gear 4 and Gear 6 EC3 slip tables to rebuild the Gear 5 table,
then export the headerless copy.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing slip table: {path}")
    return pd.read_csv(path, sep="\t", engine="python")


def ensure_compatible(a: pd.DataFrame, b: pd.DataFrame) -> None:
    if list(a.columns) != list(b.columns):
        raise ValueError("Columns differ between Gear4 and Gear6 slip tables.")
    if not np.allclose(a.iloc[:, 0].to_numpy(dtype=float), b.iloc[:, 0].to_numpy(dtype=float)):
        raise ValueError("First column (rpm axis) mismatch between Gear4 and Gear6 tables.")


def average_cells(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    out = a.copy()
    numeric_cols = a.columns[1:]
    avg = (a[numeric_cols].astype(float) + b[numeric_cols].astype(float)) / 2.0
    out[numeric_cols] = avg
    return out


def write_nhdr(df: pd.DataFrame, path: Path) -> None:
    numeric = df.iloc[:, 1:].copy()
    nohdr_path = path.with_name(path.stem + "__NOHDR.tsv")
    numeric.to_csv(nohdr_path, sep="\t", index=False, header=False, encoding="utf-8")
    print(f"[INFO] Wrote headerless slip table: {nohdr_path}")


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    slip_dir = repo / "newlogs" / "output" / "01_tables" / "tcc_slip"
    g4 = slip_dir / "TCC_SLIP_TABLE__GEAR4__EC3_FROM_LOGS.tsv"
    g5 = slip_dir / "TCC_SLIP_TABLE__GEAR5__EC3_FROM_LOGS.tsv"
    g6 = slip_dir / "TCC_SLIP_TABLE__GEAR6__EC3_FROM_LOGS.tsv"

    print(f"[INFO] Loading Gear4/6 tables from {slip_dir}")
    df4 = load_table(g4)
    df6 = load_table(g6)
    ensure_compatible(df4, df6)
    df5_new = average_cells(df4, df6)

    if g5.exists():
        backup = slip_dir / "TCC_SLIP_TABLE__GEAR5__EC3_FROM_LOGS__backup_before_avg.tsv"
        g5.replace(backup)
        print(f"[INFO] Backed up old Gear5 table to {backup.name}")

    df5_new.to_csv(g5, sep="\t", index=False, encoding="utf-8")
    stats = df5_new.iloc[:, 1:].to_numpy(dtype=float)
    print(
        "[INFO] Gear5 stats:",
        f"min={np.nanmin(stats):.1f}, mean={np.nanmean(stats):.1f}, max={np.nanmax(stats):.1f}",
    )
    write_nhdr(df5_new, g5)


if __name__ == "__main__":
    main()
