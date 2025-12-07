#!/usr/bin/env python
import sys
from pathlib import Path

import pandas as pd


ROOT = Path("newlogs")
SHIFT_BASE_DIR = ROOT / "output" / "01_tables" / "shift"
TCC_BASE_DIR = ROOT / "output" / "01_tables" / "tcc"

# Base COMFORT tables
BASE_SHIFT_UP = SHIFT_BASE_DIR / "SHIFT_TABLES__UP__Throttle17__COMFORT.tsv"
BASE_SHIFT_DOWN = SHIFT_BASE_DIR / "SHIFT_TABLES__DOWN__Throttle17__COMFORT.tsv"
BASE_TCC_APPLY = TCC_BASE_DIR / "TCC_APPLY__Throttle17__COMFORT.tsv"
BASE_TCC_REL = TCC_BASE_DIR / "TCC_RELEASE__Throttle17__COMFORT.tsv"

# Tahoe delta sources
PASS_DIR = ROOT / "output" / "02_passes"

DELTA_SHIFT_UP = [
    PASS_DIR / "CONSIST_FROM_BEST_TAHOE" / "CONSIST__SHIFT_UP__DELTA__TAHOE.tsv",
    PASS_DIR / "LAT_FROM_BEST_TAHOE" / "LAT__SHIFT_UP__DELTA__TAHOE.tsv",
    PASS_DIR / "INTENT_FROM_BEST_TAHOE" / "INTENT__SHIFT_UP__DELTA__TAHOE.tsv",
]

DELTA_SHIFT_DOWN = [
    PASS_DIR / "CONSIST_FROM_BEST_TAHOE" / "CONSIST__SHIFT_DOWN__DELTA__TAHOE.tsv",
    PASS_DIR / "STOPGO_FROM_BEST_TAHOE" / "STOPGO__SHIFT_DOWN__DELTA.tsv",
    PASS_DIR / "KICKDOWN_FROM_BEST_TAHOE" / "KICKDOWN__SHIFT_DOWN__DELTA.tsv",
    PASS_DIR / "CORNER_FROM_BEST_TAHOE" / "CORNER__SHIFT_DOWN__DELTA__COMBINED.tsv",
]

DELTA_TCC_APPLY = [
    PASS_DIR / "TCC_EDGE_FROM_BEST_TAHOE" / "TCC_EDGE__APPLY__DELTA__TAHOE.tsv",
]

DELTA_TCC_REL = [
    PASS_DIR / "TCC_EDGE_FROM_BEST_TAHOE" / "TCC_EDGE__RELEASE__DELTA__TAHOE.tsv",
    PASS_DIR / "INTENT_FROM_BEST_TAHOE" / "INTENT__TCC_RELEASE__DELTA__TAHOE.tsv",
]


OUT_DIR = ROOT / "output" / "01_tables" / "COMFORT_TAHOE"


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[ERROR] Base table missing: {path}")
        sys.exit(1)
    df = pd.read_csv(path, sep="\t")
    return df


def load_delta_like(base: pd.DataFrame, path: Path, label: str) -> pd.DataFrame:
    """
    Load a delta table, align to base, and return numeric deltas.
    Missing file -> zero delta.
    Non-numeric entries (blank, etc.) are treated as 0 delta.
    """
    cols = list(base.columns)
    num_cols = [c for c in cols if c not in ("mph", "%")]

    if not path.exists():
        print(f"[WARN] Delta file not found for {label}: {path} (treated as zeros)")
        return pd.DataFrame(0.0, index=base.index, columns=num_cols)

    d = pd.read_csv(path, sep="\t")
    # Align structure; ignore any extra cols in the delta file
    missing_cols = [c for c in num_cols if c not in d.columns]
    if missing_cols:
        print(f"[WARN] Delta {path} missing cols {missing_cols}, filling with zeros")

    out = {}
    for c in num_cols:
        if c in d.columns:
            out[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
        else:
            out[c] = 0.0

    df_delta = pd.DataFrame(out, index=base.index)
    return df_delta


def apply_deltas(base_path: Path, delta_paths, out_path: Path, label: str, is_tcc: bool = False):
    base = load_table(base_path)
    cols = list(base.columns)
    num_cols = [c for c in cols if c not in ("mph", "%")]

    # Build sum of deltas
    total_delta = pd.DataFrame(0.0, index=base.index, columns=num_cols)
    for p in delta_paths:
        df_d = load_delta_like(base, p, label)
        total_delta[num_cols] = total_delta[num_cols] + df_d[num_cols]

    # Apply to base
    out = base.copy()
    base_num = {}
    for c in num_cols:
        base_num[c] = pd.to_numeric(base[c], errors="coerce")

    for c in num_cols:
        out[c] = (base_num[c].fillna(0.0) + total_delta[c]).round(1)

    # For TCC tables, preserve 317/318 sentinels exactly from base
    if is_tcc:
        for c in num_cols:
            sentinel_mask = base[c].astype(str).isin(["317", "318", "317.0", "318.0"])
            out.loc[sentinel_mask, c] = base.loc[sentinel_mask, c]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False, lineterminator="\n")
    # Count nonzero changes vs base for info
    changed = 0
    total_cells = 0
    for c in num_cols:
        b = pd.to_numeric(base[c], errors="coerce").fillna(0.0)
        o = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        diff = (o - b).abs()
        changed += (diff > 0.0001).sum()
        total_cells += len(diff)

    print(f"[BLEND] {label}: wrote {out_path}")
    print(f"         changed cells: {changed} / {total_cells}")


def main():
    print("[BLEND] Tahoe COMFORT blend starting...")

    # SHIFT UP
    apply_deltas(
        BASE_SHIFT_UP,
        DELTA_SHIFT_UP,
        OUT_DIR / "SHIFT_TABLES__UP__Throttle17__COMFORT_TAHOE.tsv",
        label="SHIFT UP",
        is_tcc=False,
    )

    # SHIFT DOWN
    apply_deltas(
        BASE_SHIFT_DOWN,
        DELTA_SHIFT_DOWN,
        OUT_DIR / "SHIFT_TABLES__DOWN__Throttle17__COMFORT_TAHOE.tsv",
        label="SHIFT DOWN",
        is_tcc=False,
    )

    # TCC APPLY
    apply_deltas(
        BASE_TCC_APPLY,
        DELTA_TCC_APPLY,
        OUT_DIR / "TCC_APPLY__Throttle17__COMFORT_TAHOE.tsv",
        label="TCC APPLY",
        is_tcc=True,
    )

    # TCC RELEASE
    apply_deltas(
        BASE_TCC_REL,
        DELTA_TCC_REL,
        OUT_DIR / "TCC_RELEASE__Throttle17__COMFORT_TAHOE.tsv",
        label="TCC RELEASE",
        is_tcc=True,
    )

    print("[BLEND] Done. Remember to run your usual TSV guard + audit before flashing.")


if __name__ == "__main__":
    main()
