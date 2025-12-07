#!/usr/bin/env python3
"""Rowdy stop/go pass that nudges low-speed downshifts in Pattern A."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rowdy_pass_utils import (
    ROW_DOWN_ORDER,
    TPS_AXIS,
    TPS_COLS,
    assign_tps_idx,
    load_shift_events,
    read_rowdy_table,
)

TABLE_PATH = Path("newlogs/rowdy11_WOT6200_downshift.tsv")
OUTPUT_DIR = Path("newlogs/output/02_passes/STOPGO")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "STOPGO__ROWDY__SHIFT_DOWN__DELTA.tsv"

MAX_SPEED = 30.0
MAX_PEDAL = 35.0
MIN_HITS = 3
THRESH = 0.3
MAX_DELTA = 0.3
TARGET_GEARS = {"2->1", "3->2"}


def filter_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df[
        (df["mode"] == "Pattern A")
        & (df["type"] == "down")
        & (df["gear_pair"].isin(TARGET_GEARS))
        & (df["speed_mph"] <= MAX_SPEED)
        & (df["pedal"] <= MAX_PEDAL)
    ].copy()
    if {"ect_f", "tft_f"}.issubset(df.columns):
        df = df[(df["ect_f"] >= 100) & (df["tft_f"] >= 100)]
    df["tps_idx"] = df["pedal"].apply(assign_tps_idx)
    df["tps_idx"] = df["tps_idx"].clip(0, len(TPS_AXIS) - 1)
    return df


def build_delta(events: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    delta = pd.DataFrame(0.0, index=table.index, columns=TPS_COLS)
    if events.empty:
        return delta
    grouped = events.groupby(["gear_pair", "tps_idx"])["speed_mph"].agg(["count", "median"])
    for (gear_pair, tps_idx), row in grouped.iterrows():
        if row["count"] < MIN_HITS:
            continue
        if TPS_AXIS[tps_idx] > 19.0:
            continue
        if gear_pair not in table.index:
            continue
        col = TPS_COLS[tps_idx]
        current = float(table.at[gear_pair, col])
        diff = row["median"] - current
        if diff > THRESH:
            delta.at[gear_pair, col] = round(min(MAX_DELTA, diff), 1)
    return delta


def main() -> None:
    table = read_rowdy_table(TABLE_PATH, ROW_DOWN_ORDER)
    events = filter_events(load_shift_events())
    delta = build_delta(events, table)
    delta.to_csv(OUTPUT_PATH, sep="\t", float_format="%.1f", index=True)
    nonzero = (delta > 0.0).sum().sum()
    print(f"Wrote STOPGO Rowdy delta table to {OUTPUT_PATH}")
    print(f"Non-zero low-TPS cells: {int(nonzero)}")


if __name__ == "__main__":
    main()
