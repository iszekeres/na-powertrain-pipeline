#!/usr/bin/env python3
"""
High-resolution analysis of TCC slip and pressure vs. mode/gear.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SHIFT_MODE_CANDIDATES = [
    "shift_mode",
    "Shift Mode",
    "Trans Shift Pattern",
    "Shift Pattern",
    "Current Shift Pattern",
]

TCC_PRESSURE_CANDIDATES = [
    "tcc_apply_psi_raw",
    "tcc_apply_psi",
    "TCC Apply Pressure",
    "TCC Pressure",
    "TCC Pressure (PSI)",
]

TORQUE_CANDIDATES = [
    "Engine Torque",
    "Delivered Torque",
    "Engine Torque (Calculated)",
    "Engine Torque (Nm)",
]

TCC_DUTY_CANDIDATES = [
    "TCC Duty Cycle",
    "TCC PWM",
    "TCC Command",
]

LINE_PRESSURE_CANDIDATES = [
    "Line Pressure",
    "Trans Line Pressure",
]

SAMPLES_COLUMNS = [
    "file",
    "time_s",
    "speed_mph",
    "gear_actual__canon",
    "slip_abs",
    "tcc_psi",
    "shift_mode",
    "throttle_pct",
    "brake",
    "ect_c",
    "tft_c",
    "engine_torque",
    "tcc_duty",
    "line_pressure",
]

COARSE_QUANTILES = [5, 25, 50, 75, 90, 95]
QUANTILE_GRID = list(range(101))
SLIP_HIST_BINS = np.append(np.arange(0, 501, 10), np.inf)


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    cols = list(df.columns)
    for candidate in candidates:
        lower = candidate.lower()
        for col in cols:
            if lower in col.lower():
                return col
    return None


def normalize_mode(value: str | float | int) -> str:
    text = str(value).strip().lower()
    if "normal" in text:
        return "normal"
    if "pattern a" in text or "pattern_a" in text:
        return "pattern_a"
    return text or "unknown"


def percentiles(series: pd.Series, percents: Iterable[int]) -> dict[int, float]:
    if series.empty:
        return {p: float("nan") for p in percents}
    data = np.nanpercentile(series, list(percents))
    return {int(p): float(val) for p, val in zip(percents, data)}


def quantile_rows(
    group_type: str,
    group_key: str,
    series: pd.Series,
    metric: str,
    quantiles: Iterable[int] = QUANTILE_GRID,
) -> list[dict]:
    rows = []
    if series.empty:
        values = [float("nan")] * len(quantiles)
    else:
        values = np.nanpercentile(series, list(quantiles))
    for q, value in zip(quantiles, values):
        rows.append(
            {
                "group_type": group_type,
                "group_key": group_key,
                "metric": metric,
                "quantile": int(q),
                "value": float(value),
            }
        )
    return rows


def histogram(series: pd.Series, bins: np.ndarray) -> list[tuple[str, int]]:
    if series.empty:
        return []
    counts, edges = np.histogram(series.dropna(), bins=bins)
    hist = []
    for idx in range(len(counts)):
        low = edges[idx]
        high = edges[idx + 1]
        if np.isinf(high):
            label = f">{int(edges[-2])}"
        else:
            label = f"[{int(low)},{int(high)}):"
        hist.append((label, int(counts[idx])))
    return hist


def summarize_percentiles(series: pd.Series, prefix: str) -> str:
    data = percentiles(series.dropna(), COARSE_QUANTILES)
    values = "/".join(f"p{q}:{data[q]:.1f}" for q in COARSE_QUANTILES)
    return f"{prefix} {values}"


def format_coarse_stats(data: dict[int, float]) -> str:
    return "/".join(f"p{q}:{data.get(q, float('nan')):.1f}" for q in COARSE_QUANTILES)


def main():
    parser = argparse.ArgumentParser(description="High-res TCC slip scan.")
    parser.add_argument("--session", default="indy trip", help="Session name under logs_processed.")
    args = parser.parse_args()

    repo = Path.cwd()
    base = repo / "logs_processed" / args.session
    out00 = base / "output" / "00_cleaner"
    if not out00.exists():
        print(f"[error] missing analyzer output directory: {out00}")
        raise SystemExit(1)

    files = sorted(out00.glob("__trans_focus__clean_FULL__*.csv"))
    if not files:
        print(f"[error] no trans_focus clean_FULL files found under {out00}")
        raise SystemExit(1)

    all_samples: list[pd.DataFrame] = []
    per_file_stats: list[dict] = []
    duration_sum = 0.0
    usable_files = 0

    for file in files:
        df = pd.read_csv(file, low_memory=False)
        if "tcc_slip_rpm" not in df.columns and "tcc_slip_fused" not in df.columns:
            print(f"[warn] skipping {file.name}: missing slip column")
            continue
        if not {"time_s", "speed_mph", "gear_actual__canon", "brake"}.issubset(df.columns):
            print(f"[warn] skipping {file.name}: missing core columns")
            continue

        slip_col = "tcc_slip_rpm" if "tcc_slip_rpm" in df.columns else "tcc_slip_fused"
        df["slip_abs"] = df[slip_col].abs()

        mode_col = find_column(df, SHIFT_MODE_CANDIDATES)
        if mode_col:
            df["shift_mode"] = df[mode_col].astype(str).map(normalize_mode)
        else:
            df["shift_mode"] = "unknown"

        psi_col = find_column(df, TCC_PRESSURE_CANDIDATES)
        if psi_col:
            df["tcc_psi"] = pd.to_numeric(df[psi_col], errors="coerce")
        else:
            df["tcc_psi"] = np.nan

        torque_col = find_column(df, TORQUE_CANDIDATES)
        df["engine_torque"] = pd.to_numeric(df[torque_col], errors="coerce") if torque_col else np.nan

        duty_col = find_column(df, TCC_DUTY_CANDIDATES)
        df["tcc_duty"] = pd.to_numeric(df[duty_col], errors="coerce") if duty_col else np.nan

        line_col = find_column(df, LINE_PRESSURE_CANDIDATES)
        df["line_pressure"] = pd.to_numeric(df[line_col], errors="coerce") if line_col else np.nan

        eligible = (
            (df["gear_actual__canon"] >= 3)
            & (df["speed_mph"] >= 25)
            & (df["brake"] == 0)
        )
        highway = eligible & (df["speed_mph"] >= 55)

        eligible_df = df.loc[eligible].copy()
        eligible_df["file"] = file.name

        all_samples.append(eligible_df[SAMPLES_COLUMNS])

        slip_percentiles = percentiles(eligible_df["slip_abs"].dropna(), COARSE_QUANTILES)
        psi_percentiles = (
            percentiles(eligible_df["tcc_psi"].dropna(), COARSE_QUANTILES)
            if eligible_df["tcc_psi"].notna().any()
            else {}
        )

        per_file_stats.append(
            {
                "file": file.name,
                "total_samples": len(df),
                "eligible": len(eligible_df),
                "highway": int(highway.sum()),
                "slip_percentiles": slip_percentiles,
                "psi_percentiles": psi_percentiles,
            }
        )

        duration_sum += float(
            df["time_s"].max() - df["time_s"].min() if len(df) >= 2 else 0.0
        )
        usable_files += 1

    if not all_samples:
        print("[error] no usable files after validation")
        raise SystemExit(1)

    combined = pd.concat(all_samples, ignore_index=True)
    combined["gear_actual__canon"] = combined["gear_actual__canon"].astype(int)
    total_samples = len(combined)
    highway_samples = int((combined["speed_mph"] >= 55).sum())

    slip_series = combined["slip_abs"].dropna()
    psi_series = combined["tcc_psi"].dropna()

    hist = histogram(slip_series, SLIP_HIST_BINS)
    if not psi_series.empty:
        psi_min = float(psi_series.min())
        psi_max = float(psi_series.max())
        psi_edges = (
            np.linspace(psi_min, psi_max, 21)
            if psi_min != psi_max
            else np.linspace(psi_min, psi_min + 1, 21)
        )
        psi_hist = histogram(psi_series, psi_edges)
    else:
        psi_hist = []

    quantile_rows_list: list[dict] = []

    def add_group(group_type: str, group_key: str, subset: pd.Series, metric: str):
        quantile_rows_list.extend(
            quantile_rows(group_type, group_key, subset, metric)
        )

    add_group("overall", "all", slip_series, "slip_abs")
    if not psi_series.empty:
        add_group("overall", "all", psi_series, "tcc_psi")

    for gear in range(3, 7):
        mask = combined["gear_actual__canon"] == gear
        add_group("gear", f"gear={gear}", combined.loc[mask, "slip_abs"], "slip_abs")
        if not psi_series.empty:
            add_group("gear", f"gear={gear}", combined.loc[mask, "tcc_psi"], "tcc_psi")

    modes = combined["shift_mode"].dropna().unique()
    for mode in modes:
        mask = combined["shift_mode"] == mode
        add_group("mode", f"mode={mode}", combined.loc[mask, "slip_abs"], "slip_abs")
        if not psi_series.empty:
            add_group("mode", f"mode={mode}", combined.loc[mask, "tcc_psi"], "tcc_psi")
        for gear in range(3, 7):
            mask2 = mask & (combined["gear_actual__canon"] == gear)
            add_group(
                "mode_gear",
                f"mode={mode},gear={gear}",
                combined.loc[mask2, "slip_abs"],
                "slip_abs",
            )
            add_group(
                "mode_gear_highway",
                f"mode={mode},gear={gear}",
                combined.loc[mask2 & (combined["speed_mph"] >= 55), "slip_abs"],
                "slip_abs",
            )
            if not psi_series.empty:
                add_group(
                    "mode_gear",
                    f"mode={mode},gear={gear}",
                    combined.loc[mask2, "tcc_psi"],
                    "tcc_psi",
                )
                add_group(
                    "mode_gear_highway",
                    f"mode={mode},gear={gear}",
                    combined.loc[mask2 & (combined["speed_mph"] >= 55), "tcc_psi"],
                    "tcc_psi",
                )

    quantiles_df = pd.DataFrame(quantile_rows_list)
    quantiles_path = base / "TCC_SLIP_STATE_QUANTILES.csv"
    quantiles_df.to_csv(quantiles_path, index=False, encoding="utf-8")

    samples_df = combined[SAMPLES_COLUMNS].copy()
    max_rows = 20_000_000
    if len(samples_df) > max_rows:
        step = max(1, len(samples_df) // max_rows)
        samples_df = samples_df.iloc[::step]
    samples_path = base / "TCC_SLIP_STATE_SAMPLES.csv"
    samples_df.to_csv(samples_path, index=False, encoding="utf-8")

    summary_lines = [
        f"[tcc scan] session = {args.session}",
        f"  logs used: {usable_files}",
        f"  total eligible samples: {total_samples}",
        f"  total highway samples: {highway_samples}",
        f"  approx duration: {duration_sum/3600:.1f} hours",
        "",
        "Overall eligible slip_abs (rpm):",
        summarize_percentiles(slip_series, "slip"),
    ]
    if not psi_series.empty:
        summary_lines.extend(
            ["", "Overall eligible TCC psi:", summarize_percentiles(psi_series, "psi")]
        )
    summary_lines.append("")
    summary_lines.append("Per gear (eligible):")
    for gear in range(3, 7):
        mask = combined["gear_actual__canon"] == gear
        summary_lines.append(
            f"  gear {gear}: n={mask.sum()}, {summarize_percentiles(combined.loc[mask, 'slip_abs'], 'slip')}"
        )
    summary_lines.append("")
    summary_lines.append("Per mode (eligible):")
    for mode in modes:
        mask = combined["shift_mode"] == mode
        summary_lines.append(
            f"  mode={mode}: n={mask.sum()}, {summarize_percentiles(combined.loc[mask, 'slip_abs'], 'slip')}"
        )
    summary_lines.append("")
    summary_lines.append("Highway (>=55 mph), mode+gear:")
    for mode, gear in [("normal", 5), ("normal", 6), ("pattern_a", 5), ("pattern_a", 6)]:
        mask = (
            (combined["shift_mode"] == mode)
            & (combined["gear_actual__canon"] == gear)
            & (combined["speed_mph"] >= 55)
        )
        summary_lines.append(
            f"  {mode}, gear {gear}: n={mask.sum()}, {summarize_percentiles(combined.loc[mask, 'slip_abs'], 'slip')}"
        )
        if not psi_series.empty:
            summary_lines.append(
                f"    psi: {summarize_percentiles(combined.loc[mask, 'tcc_psi'], 'psi')}"
            )
    summary_lines.append("")
    summary_lines.append("Slip histogram (eligible):")
    for label, count in hist:
        summary_lines.append(f"  {label} count={count}")
    if psi_hist:
        summary_lines.append("")
        summary_lines.append("Psi histogram (eligible):")
        for label, count in psi_hist:
            summary_lines.append(f"  {label} count={count}")
    summary_lines.append("")
    summary_lines.append("Per-file summary:")
    for stats in per_file_stats:
        summary_lines.append(f"  {stats['file']}:")
        summary_lines.append(f"    n_total: {stats['total_samples']}")
        summary_lines.append(f"    n_eligible: {stats['eligible']}")
        summary_lines.append(f"    n_highway: {stats['highway']}")
        summary_lines.append(f"    slip: {format_coarse_stats(stats['slip_percentiles'])}")
        if stats["psi_percentiles"]:
            summary_lines.append(f"    psi: {format_coarse_stats(stats['psi_percentiles'])}")
    summary_lines.extend(
        [
            "",
            f"full 1% quantiles written to: {quantiles_path.name}",
            f"sample rows written to:       {samples_path.name}",
        ]
    )
    summary_text = "\n".join(summary_lines)

    summary_path = base / "TCC_SLIP_STATE_SCAN.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
