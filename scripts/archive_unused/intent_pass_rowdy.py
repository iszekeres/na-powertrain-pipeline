#!/usr/bin/env python3
"""Rowdy intent pass that nudges Pattern A upshift tables based on high-intent events."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rowdy_pass_utils import (
    ROW_UP_ORDER,
    TPS_COLS,
    assign_tps_idx,
    load_shift_events,
    read_rowdy_table,
)

SPEED_WINDOW = (30.0, 80.0)
PEDAL_MIN = 20.0
HIGH_INTENT_PEDAL_RATE = 12.0
HIGH_INTENT_THROTTLE_RATE = 9.0
MIN_HITS = 3
THRESH = 0.3
MAX_DELTA = 0.2

TABLE_PATH = Path("newlogs/rowdy11_WOT6200_upshift.tsv")
TCC_RELEASE_PATH = Path("newlogs/rowdy11_WOT6200_tcc_release.tsv")
OUTPUT_DIR = Path("newlogs/output/02_passes/INTENT")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHIFT_OUT = OUTPUT_DIR / "INTENT__ROWDY__SHIFT_UP__DELTA.tsv"
TCC_OUT = OUTPUT_DIR / "INTENT__ROWDY__TCC_RELEASE__DELTA.tsv"


def prepare_events() -> pd.DataFrame:
    df = load_shift_events()
    df = df[df["mode"] == "Pattern A"].copy()
    if {"ect_f", "tft_f"}.issubset(df.columns):
        df = df[(df["ect_f"] >= 100) & (df["tft_f"] >= 100)]
    df["dt"] = df.groupby("file")["time_s"].diff().fillna(0.1)
    df["dt"] = df["dt"].replace(0.0, 0.1)
    df["pedal_prev"] = df.groupby("file")["pedal"].shift(1).fillna(df["pedal"])
    df["throttle_prev"] = df.groupby("file")["throttle"].shift(1).fillna(df["throttle"])
    df["pedal_rate"] = (df["pedal"] - df["pedal_prev"]) / df["dt"]
    df["throttle_rate"] = (df["throttle"] - df["throttle_prev"]) / df["dt"]
    df = df[
        (df["speed_mph"] >= SPEED_WINDOW[0])
        & (df["speed_mph"] <= SPEED_WINDOW[1])
        & (df["pedal"] >= PEDAL_MIN)
    ]
    df["high_intent"] = (
        (df["pedal_rate"] >= HIGH_INTENT_PEDAL_RATE)
        | (df["throttle_rate"] >= HIGH_INTENT_THROTTLE_RATE)
    )
    df["tps_idx"] = df["throttle"].apply(assign_tps_idx)
    df["tps_col"] = df["tps_idx"].apply(lambda idx: TPS_COLS[idx])
    return df


def compute_shift_delta(events: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    delta = pd.DataFrame(0.0, index=table.index, columns=TPS_COLS)
    high_intent = events[events["high_intent"]]
    if high_intent.empty:
        return delta
    grouped = high_intent.groupby(["gear_pair", "tps_col"])["speed_mph"].agg(["count", "median"])
    for (gear_pair, tps_col), row in grouped.iterrows():
        if row["count"] < MIN_HITS:
            continue
        if gear_pair not in delta.index:
            continue
        if tps_col not in TPS_COLS:
            continue
        table_value = float(table.at[gear_pair, tps_col])
        diff = row["median"] - table_value
        if diff >= THRESH:
            delta.at[gear_pair, tps_col] = round(min(MAX_DELTA, diff), 1)
    return delta


def zero_tcc_delta(table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=table.index, columns=TPS_COLS)


def main() -> None:
    table = read_rowdy_table(TABLE_PATH, ROW_UP_ORDER)
    events = prepare_events()
    shift_delta = compute_shift_delta(events, table)
    tcc_table = read_rowdy_table(TCC_RELEASE_PATH, ["3rd", "4th", "5th", "6th"])
    tcc_delta = zero_tcc_delta(tcc_table)
    shift_delta.to_csv(SHIFT_OUT, sep="\t", float_format="%.1f", index=True)
    tcc_delta.to_csv(TCC_OUT, sep="\t", float_format="%.1f", index=True)
    print(f"Wrote shift delta table to {SHIFT_OUT}")
    print(f"Wrote TCC delta table to {TCC_OUT}")
    print(f"Shift delta non-zero cells: {(shift_delta > 0.0).sum().sum():.0f}")
    print(f"TCC delta non-zero cells: {(tcc_delta != 0.0).sum().sum():.0f}")


if __name__ == "__main__":
    main()
