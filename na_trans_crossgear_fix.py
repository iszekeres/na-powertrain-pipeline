#!/usr/bin/env python
import pandas as pd
from pathlib import Path

TPS_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]
ROW_ORDER = [
    "1 -> 2 Shift",
    "2 -> 3 Shift",
    "3 -> 4 Shift",
    "4 -> 5 Shift",
    "5 -> 6 Shift",
]

# Minimum cross-gear gap in mph (2-3 at least GAP above 1-2, etc.)
CROSS_GEAR_MIN_GAP = 1.5


def enforce_crossgear(df: pd.DataFrame, gap: float) -> pd.DataFrame:
    # Find row indices for the 5 UP rows in the expected order
    idxs = []
    mph_col = df.columns[0]
    for label in ROW_ORDER:
        matches = df.index[df[mph_col].astype(str) == label].tolist()
        if not matches:
            raise SystemExit(f"[ERROR] Missing row '{label}' in SHIFT UP table.")
        idxs.append(matches[0])

    # For each TPS column, enforce 1-2 < 2-3 < 3-4 < 4-5 < 5-6 with min gap
    for tps in TPS_AXIS:
        col = str(tps)
        if col not in df.columns:
            continue

        prev_val = None
        for row_i in idxs:
            raw = df.at[row_i, col]
            if pd.isna(raw):
                continue
            try:
                v = float(raw)
            except (TypeError, ValueError):
                continue

            if prev_val is None:
                prev_val = v
                continue

            min_allowed = prev_val + gap
            if v < min_allowed:
                v = min_allowed
                df.at[row_i, col] = v
            prev_val = v

    # Round all numeric TPS columns back to 0.1 mph
    for col in df.columns:
        if col in (mph_col, "%"):
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(1)
        except Exception:
            # Non-numeric or special columns are left untouched
            pass

    return df


def main():
    base = (
        Path("newlogs")
        / "output"
        / "01_tables"
        / "BLENDED_LOGOVERBASE"
        / "AUDITFIX_SMOOTH"
    )
    src = base / "SHIFT_TABLES__UP__Throttle17.tsv"

    if not src.exists():
        raise SystemExit(f"[ERROR] SHIFT UP table not found: {src}")

    print(f"[INFO] Loading SHIFT UP table from: {src}")
    df = pd.read_csv(src, sep="\t")

    df = enforce_crossgear(df, CROSS_GEAR_MIN_GAP)

    tmp = src.with_suffix(".tmp")
    df.to_csv(tmp, sep="\t", index=False, float_format="%.1f")
    tmp.replace(src)

    print(f"[OK] Cross-gear smoothing applied and written back to: {src}")


if __name__ == "__main__":
    main()

