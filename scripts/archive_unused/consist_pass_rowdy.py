#!/usr/bin/env python3
"""Rowdy consistency pass for Pattern A up/down tables."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from rowdy_pass_utils import (
    ROW_DOWN_ORDER,
    ROW_UP_ORDER,
    TPS_COLS,
    assign_tps_idx,
    load_shift_events,
    read_rowdy_table,
)

MIN_HITS = 4
THRESH_UP = 0.3
THRESH_DOWN = 0.3
CLAMP = 0.2

OUTPUT_DIR = Path("newlogs/output/02_passes/CONSIST")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHIFT_UP_OUT = OUTPUT_DIR / "CONSIST__ROWDY__SHIFT_UP__DELTA.tsv"
SHIFT_DOWN_OUT = OUTPUT_DIR / "CONSIST__ROWDY__SHIFT_DOWN__DELTA.tsv"


def filter_events(df: pd.DataFrame, gear_pairs: List[str]) -> pd.DataFrame:
    df = df[
        (df["mode"] == "Pattern A")
        & (df["speed_mph"] <= 80)
        & (df["gear_pair"].isin(gear_pairs))
    ].copy()
    if {"ect_f", "tft_f"}.issubset(df.columns):
        df = df[(df["ect_f"] >= 100) & (df["tft_f"] >= 100)]
    df["tps_idx"] = df["pedal"].apply(assign_tps_idx)
    return df


def build_delta(
    table: pd.DataFrame,
    events: pd.DataFrame,
    shift_type: str,
    threshold: float,
    clamp: float,
    allow_negative: bool,
) -> pd.DataFrame:
    delta = pd.DataFrame(0.0, index=table.index, columns=TPS_COLS)
    subset = events[events["type"] == shift_type]
    if subset.empty:
        return delta
    for gear_name in table.index:
        gear_events = subset[subset["gear_pair"] == gear_name]
        if gear_events.empty:
            continue
        for idx, col in enumerate(TPS_COLS):
            bucket = gear_events[gear_events["tps_idx"] == idx]
            if len(bucket) < MIN_HITS:
                continue
            median_speed = bucket["speed_mph"].median()
            current = float(table.at[gear_name, col])
            diff = median_speed - current
            if abs(diff) >= threshold:
                val = max(-clamp, min(clamp, diff))
                if not allow_negative:
                    val = max(0.0, val)
                delta.at[gear_name, col] = round(val, 1)
    return delta


def main() -> None:
    gear_pairs = ROW_UP_ORDER + ROW_DOWN_ORDER
    events = filter_events(load_shift_events(), gear_pairs)
    up_table = read_rowdy_table(Path("newlogs/rowdy11_WOT6200_upshift.tsv"), ROW_UP_ORDER)
    down_table = read_rowdy_table(Path("newlogs/rowdy11_WOT6200_downshift.tsv"), ROW_DOWN_ORDER)
    delta_up = build_delta(up_table, events, "up", THRESH_UP, CLAMP, allow_negative=True)
    delta_down = build_delta(down_table, events, "down", THRESH_DOWN, CLAMP, allow_negative=False)
    delta_up.to_csv(SHIFT_UP_OUT, sep="\t", float_format="%.1f", index=True)
    delta_down.to_csv(SHIFT_DOWN_OUT, sep="\t", float_format="%.1f", index=True)
    print(f"Wrote shift up delta to {SHIFT_UP_OUT}")
    print(f"Wrote shift down delta to {SHIFT_DOWN_OUT}")
    print(f"Up cells changed: {(delta_up != 0).sum().sum():.0f}")
    print(f"Down cells changed: {(delta_down != 0).sum().sum():.0f}")


if __name__ == "__main__":
    main()
