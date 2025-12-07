#!/usr/bin/env python3
"""
Session-level QC report for NA Trans logs (cleaned + trans_focus outputs).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_focus_files(out_dir: Path) -> list[Path]:
    return sorted(out_dir.glob("__trans_focus__clean_FULL__*.csv"))


def validate_columns(df: pd.DataFrame) -> bool:
    required = [
        "time_s",
        "speed_mph",
        "engine_rpm",
        "throttle_pct",
        "gear_actual__canon",
        "tcc_locked_built",
        "ect_c",
        "tft_c",
        "brake",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        return False
    slip_cols = {"tcc_slip_rpm", "tcc_slip_fused"}
    if not slip_cols & set(df.columns):
        return False
    return True


def build_metrics(df: pd.DataFrame) -> dict:
    if "tcc_slip_rpm" not in df.columns and "tcc_slip_fused" in df.columns:
        df["tcc_slip_rpm"] = df["tcc_slip_fused"]

    duration = float(df["time_s"].max() - df["time_s"].min()) if len(df) >= 2 else 0.0
    n_samples = len(df)
    avg_speed = float(df["speed_mph"].mean())
    max_speed = float(df["speed_mph"].max())

    gear_counts = df["gear_actual__canon"].fillna(0)
    total = len(gear_counts)
    gear_usage = {g: 0.0 for g in range(7)}
    if total:
        gear_usage = {g: float((gear_counts == g).sum() / total) for g in range(7)}

    locked_overall = float(df["tcc_locked_built"].mean())

    slip = df["tcc_slip_rpm"].abs()
    gear_tcc = {}
    for gear in range(3, 7):
        mask = gear_counts == gear
        if not mask.any():
            gear_tcc[gear] = {"coverage": 0.0, "locked": np.nan, "mean_abs_slip": np.nan}
            continue
        gear_tcc[gear] = {
            "coverage": float(mask.sum() / total),
            "locked": float(df.loc[mask, "tcc_locked_built"].mean()),
            "mean_abs_slip": float(slip.loc[mask].mean()),
        }

    ect = df["ect_c"].dropna()
    tft = df["tft_c"].dropna()
    temps = {
        "ect_min": float(ect.min()) if len(ect) else np.nan,
        "ect_max": float(ect.max()) if len(ect) else np.nan,
        "ect_mean": float(ect.mean()) if len(ect) else np.nan,
        "tft_min": float(tft.min()) if len(tft) else np.nan,
        "tft_max": float(tft.max()) if len(tft) else np.nan,
        "tft_mean": float(tft.mean()) if len(tft) else np.nan,
    }

    highway_mask = df["speed_mph"] >= 55
    highway_samples = int(highway_mask.sum())
    highway_locked = float(df.loc[highway_mask, "tcc_locked_built"].mean()) if highway_samples else np.nan
    highway_speed = float(df.loc[highway_mask, "speed_mph"].mean()) if highway_samples else np.nan

    return {
        "duration": duration,
        "samples": n_samples,
        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "gear_usage": gear_usage,
        "locked_overall": locked_overall,
        "gear_tcc": gear_tcc,
        "temps": temps,
        "highway_samples": highway_samples,
        "highway_locked": highway_locked,
        "highway_avg_speed": highway_speed,
        "df": df,
    }


def format_pct(value: float) -> str:
    return f"{value*100:.1f}%" if not pd.isna(value) else "N/A"


def summarize_metrics(session: str, per_log: list[tuple[str, dict]], combined: pd.DataFrame) -> str:
    total_samples = sum(entry["samples"] for _, entry in per_log)
    duration_sec = sum(entry["duration"] for _, entry in per_log)
    duration_h = int(duration_sec // 3600)
    duration_m = int((duration_sec % 3600) // 60)

    combined_duration = (
        float(combined["time_s"].max() - combined["time_s"].min())
        if len(combined) >= 2
        else duration_sec
    )

    gear_counts = combined["gear_actual__canon"]
    total_combined = len(gear_counts)
    gear_usage = {g: 0.0 for g in range(7)}
    if total_combined:
        gear_usage = {
            g: float((gear_counts == g).sum() / total_combined) for g in range(7)
        }

    locked_avg = float(combined["tcc_locked_built"].mean()) if len(combined) else np.nan

    combined_gear_tcc = {}
    for gear in range(3, 7):
        mask = combined["gear_actual__canon"] == gear
        if not mask.any():
            combined_gear_tcc[gear] = {"locked": np.nan}
            continue
        combined_gear_tcc[gear] = {
            "locked": float(combined.loc[mask, "tcc_locked_built"].mean()),
        }

    highway_mask = combined["speed_mph"] >= 55
    highway_samples = int(highway_mask.sum())
    highway_locked = (
        float(combined.loc[highway_mask, "tcc_locked_built"].mean()) if highway_samples else np.nan
    )
    highway_speed = (
        float(combined.loc[highway_mask, "speed_mph"].mean()) if highway_samples else np.nan
    )

    ect = combined["ect_c"].dropna()
    tft = combined["tft_c"].dropna()
    ect_stats = (
        float(ect.min()) if len(ect) else np.nan,
        float(ect.max()) if len(ect) else np.nan,
        float(ect.mean()) if len(ect) else np.nan,
    )
    tft_stats = (
        float(tft.min()) if len(tft) else np.nan,
        float(tft.max()) if len(tft) else np.nan,
        float(tft.mean()) if len(tft) else np.nan,
    )

    warnings = []
    if not pd.isna(tft_stats[1]) and tft_stats[1] > 110:
        warnings.append("high trans temp")
    if highway_samples and not pd.isna(highway_locked) and highway_locked < 0.3:
        warnings.append("low highway TCC lock fraction")
    if gear_usage[5] < 0.05:
        warnings.append("low gear 5 coverage")
    if gear_usage[6] < 0.05:
        warnings.append("low gear 6 coverage")

    lines = [
        f"[qc] session = {session}",
        f"      logs    = {len(per_log)}",
        f"      samples = {total_samples}",
        f"      duration ~ {duration_h}h{duration_m}m",
        "",
        "Per-log:",
    ]
    for filename, data in per_log:
        lines.append(f"  log: {filename}")
        lines.append(f"    samples: {data['samples']}")
        lines.append(f"    duration: {data['duration']:.1f}s")
        lines.append(f"    avg speed: {data['avg_speed']:.1f} mph")
        lines.append(f"    max speed: {data['max_speed']:.1f} mph")
        gear_line = ", ".join(f"{g}: {format_pct(data['gear_usage'][g])}" for g in range(7))
        lines.append(f"    gear usage: {gear_line}")
        lines.append(f"    TCC locked overall: {format_pct(data['locked_overall'])}")
        gear3_6 = "; ".join(
            f"{g}: locked {format_pct(data['gear_tcc'][g]['locked'])}, slip {data['gear_tcc'][g]['mean_abs_slip']:.1f} rpm"
            if not pd.isna(data['gear_tcc'][g]['locked'])
            else f"{g}: no samples"
            for g in range(3, 7)
        )
        lines.append(f"    TCC (gear 3-6): {gear3_6}")
        temps = data['temps']
        lines.append(
            f"    temps: ECT {temps['ect_min']:.1f}/{temps['ect_max']:.1f}/{temps['ect_mean']:.1f}, "
            f"TFT {temps['tft_min']:.1f}/{temps['tft_max']:.1f}/{temps['tft_mean']:.1f}"
        )
        highway_info = (
            "none"
            if data['highway_samples'] == 0
            else f"locked {format_pct(data['highway_locked'])}, avg speed {data['highway_avg_speed']:.1f}"
        )
        lines.append(f"    highway: {data['highway_samples']} samples, {highway_info}")
        lines.append("")

    lines.extend(
        [
            "Combined (session):",
            f"  total samples: {total_samples}",
            f"  approximate duration: {combined_duration:.1f}s",
            "  gear usage (combined): "
            + ", ".join(f"{g}: {format_pct(gear_usage[g])}" for g in range(7)),
            f"  TCC locked overall: {format_pct(locked_avg)}",
            "  TCC locked (gear 3-6): "
            + "; ".join(
                f"{g}: {format_pct(combined_gear_tcc[g]['locked'])}"
                if not pd.isna(combined_gear_tcc[g]['locked'])
                else f"{g}: no samples"
                for g in range(3, 7)
            ),
            f"  highway samples: {highway_samples}",
            f"  highway locked: {format_pct(highway_locked)}",
            (
                f"  highway avg speed: {highway_speed:.1f} mph"
                if highway_samples
                else "  highway avg speed: n/a"
            ),
            f"  ECT min/max/mean: {ect_stats[0]:.1f}/{ect_stats[1]:.1f}/{ect_stats[2]:.1f}",
            f"  TFT min/max/mean: {tft_stats[0]:.1f}/{tft_stats[1]:.1f}/{tft_stats[2]:.1f}",
            "",
            "Warnings:",
            *([f"  - {msg}" for msg in warnings] if warnings else ["  - none"]),
        ]
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="QC session-level transmission data.")
    parser.add_argument("--session", default="indy trip", help="Session name in logs_processed.")
    args = parser.parse_args()

    repo = Path.cwd()
    base = repo / "logs_processed" / args.session
    out00 = base / "output" / "00_cleaner"
    if not out00.exists():
        print(f"[error] missing analyzer output directory: {out00}")
        raise SystemExit(1)

    focus_files = load_focus_files(out00)
    if not focus_files:
        print(f"[error] no trans_focus clean_FULL files found under {out00}")
        raise SystemExit(1)

    per_log: list[tuple[str, dict]] = []
    combined_frames: list[pd.DataFrame] = []
    for file in focus_files:
        df = pd.read_csv(file, low_memory=False)
        if not validate_columns(df):
            print(f"[warn] skipping {file.name}: missing critical columns")
            continue
        metrics = build_metrics(df)
        per_log.append((file.name, metrics))
        subset = df[
            [
                "time_s",
                "speed_mph",
                "gear_actual__canon",
                "tcc_locked_built",
                "tcc_slip_rpm",
                "ect_c",
                "tft_c",
            ]
        ]
        combined_frames.append(subset)

    if not per_log:
        print("[error] no usable logs after validation")
        raise SystemExit(1)

    combined = pd.concat(combined_frames, ignore_index=True)
    report = summarize_metrics(args.session, per_log, combined)

    print(report)

    summary_path = base / "SESSION_QC_SUMMARY.txt"
    summary_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
