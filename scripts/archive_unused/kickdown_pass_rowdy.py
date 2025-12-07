#!/usr/bin/env python3
"""Rowdy kickdown pass that nudges the 3->2/4->3/5->4/6->5 downshifts at higher pedal."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rowdy_pass_utils import (
    ROW_DOWN_ORDER,
    TPS_COLS,
    assign_tps_idx,
    load_shift_events,
    read_rowdy_table,
)

TABLE_PATH = Path("newlogs/rowdy11_WOT6200_downshift.tsv")
OUTPUT_DIR = Path("newlogs/output/02_passes/KICKDOWN")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "KICKDOWN__ROWDY__SHIFT_DOWN__DELTA.tsv"

MIN_SPEED = 30.0
MAX_SPEED = 100.0
MIN_PEDAL = 40.0
MIN_HITS = 3
THRESH = 0.3
MAX_DELTA = 0.3
TARGET_GEARS = {"3->2", "4->3", "5->4", "6->5"}


def filter_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df[
        (df["mode"] == "Pattern A")
        & (df["type"] == "down")
        & (df["gear_pair"].isin(TARGET_GEARS))
        & (df["speed_mph"].between(MIN_SPEED, MAX_SPEED))
        & (df["pedal"] >= MIN_PEDAL)
    ].copy()
    if {"ect_f", "tft_f"}.issubset(df.columns):
        df = df[(df["ect_f"] >= 100) & (df["tft_f"] >= 100)]
    df["tps_idx"] = df["pedal"].apply(assign_tps_idx)
    return df


def build_delta(table: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    delta = pd.DataFrame(0.0, index=table.index, columns=TPS_COLS)
    if events.empty:
        return delta
    grouped = events.groupby(["gear_pair", "tps_idx"])["speed_mph"].agg(["count", "median"])
    for (gear_pair, idx), row in grouped.iterrows():
        if row["count"] < MIN_HITS:
            continue
        if gear_pair not in table.index:
            continue
        col = TPS_COLS[idx]
        current = float(table.at[gear_pair, col])
        diff = row["median"] - current
        if diff >= THRESH:
            delta.at[gear_pair, col] = round(min(MAX_DELTA, diff), 1)
    return delta


def main() -> None:
    table = read_rowdy_table(TABLE_PATH, ROW_DOWN_ORDER)
    events = filter_events(load_shift_events())
    delta = build_delta(table, events)
    delta.to_csv(OUTPUT_PATH, sep="\t", float_format="%.1f", index=True)
    non_zero = (delta > 0.0).sum().sum()
    print(f"Wrote KICKDOWN Rowdy delta table to {OUTPUT_PATH}")
    print(f"Non-zero cells: {int(non_zero)}")


if __name__ == "__main__":
    main()
