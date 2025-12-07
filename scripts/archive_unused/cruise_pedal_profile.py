import numpy as np
import pandas as pd
from pathlib import Path

CSV_PATH = Path("newlogs") / "*.csv"
SPEED_BANDS = [
    ("35_50", 35.0, 50.0),
    ("70_75", 70.0, 75.0),
]
TIME_CANDIDATES = ["time_s", "offset", "Time", "time"]
GEAR_PRIMARY = "gear_actual__canon"
GEAR_FALLBACK = "gear_actual"


def pick_time_column(df: pd.DataFrame) -> str:
    for name in TIME_CANDIDATES:
        if name in df.columns:
            print(f"[INFO] Using time column: {name}")
            return name
    raise SystemExit(f"[ERROR] None of the time columns {TIME_CANDIDATES} were found.")


def load_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"speed_mph", "pedal_pct", "brake"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise SystemExit(f"[ERROR] Missing columns: {missing}")
    if GEAR_PRIMARY not in df.columns and GEAR_FALLBACK not in df.columns:
        raise SystemExit(
            f"[ERROR] Missing gear column; expected {GEAR_PRIMARY} or {GEAR_FALLBACK}"
        )
    return df


def compute_stability(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.sort_values(time_col).reset_index(drop=True)
    df["dt"] = df[time_col].diff()
    avg_dt = df["dt"].iloc[1:].mean()
    if not np.isfinite(avg_dt) or avg_dt <= 0:
        avg_dt = 0.1
    window = max(3, int(round(2.0 / avg_dt)))
    print(f"[INFO] avg_dt={avg_dt:.3f}s window={window} samples")

    df["speed_d"] = df["speed_mph"].diff() / df["dt"]
    df["pedal_d"] = df["pedal_pct"].diff() / df["dt"]

    df["speed_std"] = df["speed_mph"].rolling(window=window, center=True, min_periods=1).std()
    df["pedal_std"] = df["pedal_pct"].rolling(window=window, center=True, min_periods=1).std()
    return df


def build_cruise_mask(df: pd.DataFrame) -> tuple[pd.Series, str]:
    gear_col = GEAR_PRIMARY if GEAR_PRIMARY in df.columns else GEAR_FALLBACK
    if gear_col != GEAR_PRIMARY:
        print(f"[INFO] Falling back to gear column: {gear_col}")
    mask = (
        (df["brake"].fillna(0) <= 0.5)
        & (df["speed_std"] <= 1.0)
        & (df["pedal_std"] <= 2.0)
        & (df["speed_d"].abs() <= 0.3)
        & (df["pedal_d"].abs() <= 1.0)
        & df[gear_col].between(1, 6)
    )
    return mask, gear_col


def summarize_band(df: pd.DataFrame, mask: pd.Series, gear_col: str, label: str, lo: float, hi: float) -> None:
    band_mask = mask & df["speed_mph"].between(lo, hi)
    band_df = df[band_mask].copy()
    print("\n" + "=" * 60)
    print(f"===== SPEED BAND {label} ({lo}–{hi} mph) =====")
    total = len(band_df)
    if total == 0:
        print("No cruise samples in this band.")
        return
    print(f"Total cruise samples in band: {total}")
    stats = band_df["pedal_pct"].describe(percentiles=[0.25, 0.5, 0.75, 0.9])
    print(f"Overall pedal_pct → mean: {stats['mean']:.2f}, median: {stats['50%']:.2f}, "
          f"p25: {stats['25%']:.2f}, p75: {stats['75%']:.2f}, p90: {stats['90%']:.2f}")
    for gear, group in band_df.groupby(gear_col):
        cnt = len(group)
        if cnt == 0:
            continue
        gear_stats = group["pedal_pct"].describe(percentiles=[0.25, 0.5, 0.75, 0.9])
        print(f"Gear {gear}: n = {cnt}")
        print(f"  mean: {gear_stats['mean']:.2f}, median: {gear_stats['50%']:.2f}, "
              f"p25: {gear_stats['25%']:.2f}, p75: {gear_stats['75%']:.2f}, p90: {gear_stats['90%']:.2f}")


def main() -> None:
    csv_files = sorted((CSV_PATH.parent).glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"[ERROR] No CSV files found under {CSV_PATH.parent}")
    df = load_log(csv_files[-1])
    time_col = pick_time_column(df)
    df = compute_stability(df, time_col)
    mask, gear_col = build_cruise_mask(df)
    for label, lo, hi in SPEED_BANDS:
        summarize_band(df, mask, gear_col, label, lo, hi)


if __name__ == "__main__":
    main()
