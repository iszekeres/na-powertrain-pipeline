#!/usr/bin/env python3
import argparse
import glob
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------
# I/O helpers
# ------------------------------
TPS_COLS = [str(x) for x in [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]]


def read_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    # Drop any unnamed junk columns while preserving order
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def write_table(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")


def zero_delta_like(base_df: pd.DataFrame) -> pd.DataFrame:
    out = base_df.copy()
    for c in out.columns:
        if c not in ("mph", "%"):
            out[c] = 0.0
    return out


# ------------------------------
# BESTINTERP -> core adapter
# ------------------------------
CANON_REQUIRED = [
    "time_s__canon",
    "speed_mph__canon",
    "throttle_pct__canon",
    "gear_actual__canon",
]


def build_tmp_from_bestinterp(logs_glob: str, tmp_dir: Path) -> int:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(logs_glob))
    written = 0
    for p in files:
        src = Path(p)
        try:
            df = pd.read_csv(src, low_memory=False)
        except Exception:
            continue
        # Ensure canonical quartet present
        if any(c not in df.columns for c in CANON_REQUIRED):
            continue
        # Coerce numerics and drop NaNs
        for c in CANON_REQUIRED:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=CANON_REQUIRED)
        if df.empty:
            continue
        # Optional pedal aliasing
        pedal = None
        for name in (
            "pedal_pct__canon",
            "Pedal Position",
            "Accelerator Pedal Position",
            "Accelerator Pedal Position (%)",
            "APP",
        ):
            if name in df.columns:
                pedal = pd.to_numeric(df[name], errors="coerce")
                break
        out = pd.DataFrame(
            {
                "time_s": df["time_s__canon"],
                "speed_mph": df["speed_mph__canon"],
                "throttle_pct": df["throttle_pct__canon"],
                "gear_actual": df["gear_actual__canon"],
                "__file": src.name,
            }
        )
        if pedal is not None:
            out["pedal_pct"] = pedal
        tmp_path = tmp_dir / src.name
        out.to_csv(tmp_path, index=False)
        written += 1
    return written


def run_core_into_tmp(tmp_glob: str, tmp_dir: Path) -> None:
    # Use the existing INTENT core as-is, directing outputs to tmp_dir.
    # We reuse the same call style the previous wrapper used.
    prefix = tmp_dir / "INTENT__"
    cmd = [
        os.sys.executable,
        str(Path("driver_intent_pass_weighted.py").resolve()),
        "--logs-glob",
        tmp_glob,
        "--out-prefix",
        str(prefix),
    ]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"driver_intent_pass_weighted.py exited with code {rc}")


def find_intent_summary(tmp_dir: Path) -> Path | None:
    # Flexible search for INTENT summary file emitted by core variants
    for pat in ("INTENT__DEBUG_SUMMARY*.csv", "INTENT*SUMMARY*.csv"):
        for p in tmp_dir.glob(pat):
            if p.is_file():
                return p
    return None


def load_summary_or_none(tmp_dir: Path) -> pd.DataFrame | None:
    path = find_intent_summary(tmp_dir)
    if not path:
        return None
    df = pd.read_csv(path)
    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    # Ensure required logical columns exist (case-insensitive)
    # Try to standardize access keys
    rename_map = {}
    for want, alts in (
        ("row", ("row",)),
        ("tps_bin", ("tps_bin", "tps", "tpsbin")),
        ("count", ("count", "n", "hits")),
        ("median_mph", ("median_mph", "median", "mph_median")),
        ("std_mph", ("std_mph", "std", "mph_std")),
    ):
        for a in alts:
            if a in cols:
                rename_map[cols[a]] = want
                break
    df = df.rename(columns=rename_map)
    if not {"row", "tps_bin", "count", "median_mph", "std_mph"}.issubset(df.columns):
        return None
    return df[["row", "tps_bin", "count", "median_mph", "std_mph"]].copy()


def load_suggested_or_none(tmp_dir: Path, kind: str) -> pd.DataFrame | None:
    # kind in {"SHIFT_UP", "TCC_RELEASE"}
    for name in (
        f"INTENT__{kind}__SUGGESTED.tsv",
        f"INTENT__{kind}__SUGGESTED__TAHOE.tsv",
    ):
        p = tmp_dir / name
        if p.exists():
            try:
                df = read_table(str(p))
                return df
            except Exception:
                pass
    return None


# ------------------------------
# Tahoe aggregation (heavier, more forgiving)
# ------------------------------
# 8. SUMMARY AGGREGATION (TAHOE)
INTENT_MIN_COUNT_PER_BIN = 3
INTENT_STD_MAX = 6.0
INTENT_DELTA_CAP = 0.4

# 9. TCC RELEASE INTENT (Tahoe)
# Earlier release for stability & responsiveness.
INTENT_TCC_RELEASE_DELTA_CAP = -0.3
INTENT_TCC_RELEASE_GAP_MIN = 0.25

MIN_COUNT_BY_ROW_UP = {
    "1 -> 2 Shift": INTENT_MIN_COUNT_PER_BIN,
    "2 -> 3 Shift": INTENT_MIN_COUNT_PER_BIN,
    "3 -> 4 Shift": INTENT_MIN_COUNT_PER_BIN,
    "4 -> 5 Shift": INTENT_MIN_COUNT_PER_BIN,
    "5 -> 6 Shift": INTENT_MIN_COUNT_PER_BIN,
}
STD_MAX_UP = INTENT_STD_MAX
BASE_GAP_MIN_UP = 0.3
DELTA_CAP_UP = INTENT_DELTA_CAP

MIN_COUNT_REL = INTENT_MIN_COUNT_PER_BIN
STD_MAX_REL = INTENT_STD_MAX
BASE_GAP_MIN_REL = INTENT_TCC_RELEASE_GAP_MIN


def compute_shift_up_delta(summary: pd.DataFrame, base_up_df: pd.DataFrame) -> pd.DataFrame:
    delta_df = zero_delta_like(base_up_df)
    if summary is None or summary.empty:
        return delta_df
    sum_up = summary[summary["row"].astype(str).str.contains("Shift", na=False)].copy()
    for _, r in sum_up.iterrows():
        row_name = str(r["row"]).strip()
        if row_name not in MIN_COUNT_BY_ROW_UP:
            continue
        try:
            tps_bin = int(r["tps_bin"]) if pd.notna(r["tps_bin"]) else None
            count = int(r["count"]) if pd.notna(r["count"]) else 0
            median = float(r["median_mph"]) if pd.notna(r["median_mph"]) else np.nan
            std = float(r["std_mph"]) if pd.notna(r["std_mph"]) else np.nan
        except Exception:
            continue
        if tps_bin is None or str(tps_bin) not in base_up_df.columns:
            continue
        if count < MIN_COUNT_BY_ROW_UP[row_name]:
            continue
        if not np.isfinite(std) or std > STD_MAX_UP:
            continue
        # baseline
        try:
            base = base_up_df.loc[base_up_df["mph"] == row_name, str(tps_bin)].iloc[0]
        except Exception:
            continue
        if not np.isfinite(base):
            continue
        gap = median - base
        if gap >= BASE_GAP_MIN_UP:
            delta_val = min(DELTA_CAP_UP, gap)
        else:
            delta_val = 0.0
        delta_df.loc[delta_df["mph"] == row_name, str(tps_bin)] = round(float(delta_val), 3)
    return delta_df


def compute_tcc_release_delta(summary: pd.DataFrame, base_rel_df: pd.DataFrame) -> pd.DataFrame:
    delta_df = zero_delta_like(base_rel_df)
    if summary is None or summary.empty:
        return delta_df
    sum_rel = summary[summary["row"].astype(str).str.contains("Release", na=False)].copy()
    for _, r in sum_rel.iterrows():
        row_name = str(r["row"]).strip()
        if not any(row_name.startswith(p) for p in ("3rd", "4th", "5th", "6th")):
            continue
        try:
            tps_bin = int(r["tps_bin"]) if pd.notna(r["tps_bin"]) else None
            count = int(r["count"]) if pd.notna(r["count"]) else 0
            median = float(r["median_mph"]) if pd.notna(r["median_mph"]) else np.nan
            std = float(r["std_mph"]) if pd.notna(r["std_mph"]) else np.nan
        except Exception:
            continue
        if tps_bin is None or str(tps_bin) not in base_rel_df.columns:
            continue
        if count < MIN_COUNT_REL:
            continue
        if not np.isfinite(std) or std > STD_MAX_REL:
            continue
        # baseline
        try:
            base = base_rel_df.loc[base_rel_df["mph"] == row_name, str(tps_bin)].iloc[0]
        except Exception:
            continue
        if not np.isfinite(base):
            continue
        # Lockout sentinel
        try:
            if float(base) == 317.0:
                continue
        except Exception:
            pass
        gap_rel = median - base
        if gap_rel >= BASE_GAP_MIN_REL:
            # Negative delta: earlier TCC release, capped by INTENT_TCC_RELEASE_DELTA_CAP
            raw_delta = -gap_rel
            delta_val = max(INTENT_TCC_RELEASE_DELTA_CAP, raw_delta)
        else:
            delta_val = 0.0
        delta_df.loc[delta_df["mph"] == row_name, str(tps_bin)] = round(float(delta_val), 3)
    return delta_df


def count_nonzero_cells(df: pd.DataFrame, row_prefixes=None) -> tuple[int, int]:
    if row_prefixes is None:
        rows = df["mph"].astype(str).tolist()
    else:
        rows = [r for r in df["mph"].astype(str).tolist() if any(r.startswith(p) for p in row_prefixes)]
    mask_rows = df["mph"].astype(str).isin(rows)
    nz = (df.loc[mask_rows, TPS_COLS].to_numpy() != 0).sum()
    total = len(rows) * len(TPS_COLS)
    return int(nz), int(total)


def main():
    ap = argparse.ArgumentParser(
        description="Tahoe INTENT aggregator: calls core for events/suggested, then builds Tahoe DELTAs vs COMFORT baselines."
    )
    ap.add_argument(
        "--logs-glob",
        required=True,
        help=(
            "Glob for BESTINTERP CSVs, e.g. newlogs\\cleaned_bestinterp\\__trans_focus__clean_FULL__*__BESTINTERP.csv"
        ),
    )
    ap.add_argument(
        "--baseline-shift-up",
        required=True,
        help="Path to COMFORT SHIFT UP baseline TSV (Throttle17)",
    )
    ap.add_argument(
        "--baseline-tcc-rel",
        required=True,
        help="Path to COMFORT TCC RELEASE baseline TSV (Throttle17)",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_INTENT_TAHOE_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1) Prepare inputs for core and run it into tmp_dir
    n = build_tmp_from_bestinterp(args.logs_glob, tmp_dir)
    if n == 0:
        raise SystemExit(f"No BESTINTERP files converted from glob: {args.logs_glob}")
    run_core_into_tmp(str(tmp_dir / "*.csv"), tmp_dir)

    # 2) Load SUGGESTED (optional copies)
    up_sugg = load_suggested_or_none(tmp_dir, "SHIFT_UP")
    tcc_sugg = load_suggested_or_none(tmp_dir, "TCC_RELEASE")
    if up_sugg is not None:
        write_table(up_sugg, str(out_dir / "INTENT__SHIFT_UP__SUGGESTED__TAHOE.tsv"))
    if tcc_sugg is not None:
        write_table(tcc_sugg, str(out_dir / "INTENT__TCC_RELEASE__SUGGESTED__TAHOE.tsv"))

    # 3) Load baselines
    base_up_df = read_table(args.baseline_shift_up)
    base_rel_df = read_table(args.baseline_tcc_rel)

    # 4) Load INTENT summary
    summary = load_summary_or_none(tmp_dir)
    if summary is None:
        print("[INTENT_TAHOE] WARNING: No INTENT summary found in tmp_dir; DELTAs will be all zeros.")

    # 5) Compute Tahoe DELTAs
    up_delta_df = compute_shift_up_delta(summary, base_up_df)
    tcc_delta_df = compute_tcc_release_delta(summary, base_rel_df)

    # 6) Write outputs
    up_delta_path = out_dir / "INTENT__SHIFT_UP__DELTA__TAHOE.tsv"
    tcc_delta_path = out_dir / "INTENT__TCC_RELEASE__DELTA__TAHOE.tsv"
    write_table(up_delta_df, str(up_delta_path))
    write_table(tcc_delta_df, str(tcc_delta_path))

    # 7) Console diagnostics
    up_nz, up_tot = count_nonzero_cells(up_delta_df, row_prefixes=None)  # 5*17 = 85
    tcc_nz, tcc_tot = count_nonzero_cells(tcc_delta_df, row_prefixes=("3rd", "4th", "5th", "6th"))  # 4*17 = 68
    print(f"[INTENT_TAHOE] SHIFT_UP nonzero cells: {up_nz} / {up_tot}")
    print(f"[INTENT_TAHOE] TCC_RELEASE nonzero cells: {tcc_nz} / {tcc_tot}")


if __name__ == "__main__":
    main()
