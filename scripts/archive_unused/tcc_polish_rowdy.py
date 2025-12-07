#!/usr/bin/env python3
"""Rowdy TCC polish pass that tightens Pattern A highway lock behavior."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from rowdy_pass_utils import (
    TCC_GEAR_ORDER,
    TPS_COLS,
    read_rowdy_table,
)

SUMMARY_PATH = Path("newlogs/tcc_summary__lock_soft_unlock.txt")
APPLY_TABLE = Path("newlogs/rowdy11_WOT6200_tcc_apply.tsv")
RELEASE_TABLE = Path("newlogs/rowdy11_WOT6200_tcc_release.tsv")
OUTPUT_DIR = Path("newlogs/output/02_passes/TCC_POLISH")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
APPLY_OUT = OUTPUT_DIR / "TCC_POLISH__ROWDY__APPLY__DELTA.tsv"
RELEASE_OUT = OUTPUT_DIR / "TCC_POLISH__ROWDY__RELEASE__DELTA.tsv"

MID_TPS_MIN = 25.0
MID_TPS_MAX = 69.0
APPLY_DELTA_VALUE = -2.0
RELEASE_DELTA_VALUE = -2.0
LOCKED_SOFT_TARGET = 0.70
SLIP_TARGET = 60.0


def parse_summary() -> dict[int, dict[str, float]]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError("TCC summary missing.")
    sections = {"cruise": {}, "highway": {}}
    current_section = None
    in_block = False
    block_section = None
    pattern = re.compile(
        r"Gear\s+(?P<gear>\d+):.*?locked=(?P<locked>[\d.]+).*?"
        r"transition=(?P<soft>[\d.]+).*?avg_slip=(?P<slip>[-\d.]+)"
    )
    for line in SUMMARY_PATH.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("Cruise"):
            current_section = "cruise"
            continue
        if stripped.startswith("Highway"):
            current_section = "highway"
            continue
        if "Mode 'Pattern A' per gear" in stripped:
            in_block = True
            block_section = current_section or "cruise"
            continue
        if stripped.startswith("Mode ") and "per gear" not in stripped:
            in_block = False
        if stripped == "":
            in_block = False
        if in_block and stripped.startswith("Gear"):
            match = pattern.search(stripped)
            if not match or block_section is None:
                continue
            gear = int(match.group("gear"))
            sections[block_section][gear] = {
                "locked": float(match.group("locked")),
                "soft": float(match.group("soft")),
                "avg_slip": abs(float(match.group("slip"))),
            }
    merged: dict[int, dict[str, float]] = {}
    preferred = {
        4: ["cruise"],
        5: ["highway", "cruise"],
        6: ["highway", "cruise"],
    }
    for gear, sources in preferred.items():
        for source in sources:
            if gear in sections.get(source, {}):
                merged[gear] = sections[source][gear]
                break
    return merged


def needs_polish(stats: dict[str, float]) -> bool:
    locked_soft = stats["locked"] + stats["soft"]
    return not (locked_soft >= LOCKED_SOFT_TARGET and stats["avg_slip"] <= SLIP_TARGET)


def apply_deltas(
    apply_table: pd.DataFrame,
    release_table: pd.DataFrame,
    stats: dict[int, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    apply_delta = pd.DataFrame(0.0, index=apply_table.index, columns=TPS_COLS)
    release_delta = pd.DataFrame(0.0, index=release_table.index, columns=TPS_COLS)
    for gear, label in TCC_GEAR_ORDER.items():
        if gear not in stats:
            print(f"Gear {gear}: no stats found, skipping.")
            continue
        data = stats[gear]
        print(
            f"Gear {gear}: locked+soft={data['locked']+data['soft']:.3f}, avg_slip={data['avg_slip']:.1f}"
        )
        if not needs_polish(data):
            continue
        if label not in apply_table.index or label not in release_table.index:
            continue
        for col in TPS_COLS:
            val = float(col)
            if not (MID_TPS_MIN <= val <= MID_TPS_MAX):
                continue
            base_apply = float(apply_table.at[label, col])
            base_release = float(release_table.at[label, col])
            if np.isnan(base_apply) or np.isnan(base_release):
                continue
            if base_apply in (317.0, 318.0) or base_release in (317.0, 318.0):
                continue
            apply_adj = APPLY_DELTA_VALUE
            release_adj = RELEASE_DELTA_VALUE
            if base_release + release_adj > base_apply + apply_adj - 1.1:
                release_adj = min(release_adj, base_apply + apply_adj - 1.1 - base_release)
            release_adj = min(0.0, release_adj)
            apply_delta.at[label, col] = round(apply_adj, 1)
            release_delta.at[label, col] = round(release_adj, 1)
    return apply_delta, release_delta


def main() -> None:
    stats = parse_summary()
    apply_table = read_rowdy_table(APPLY_TABLE, ["3rd", "4th", "5th", "6th"])
    release_table = read_rowdy_table(RELEASE_TABLE, ["3rd", "4th", "5th", "6th"])
    apply_delta, release_delta = apply_deltas(apply_table, release_table, stats)
    apply_delta.to_csv(APPLY_OUT, sep="\t", float_format="%.1f", index=True)
    release_delta.to_csv(RELEASE_OUT, sep="\t", float_format="%.1f", index=True)
    print(f"Wrote apply delta to {APPLY_OUT}")
    print(f"Wrote release delta to {RELEASE_OUT}")


if __name__ == "__main__":
    main()
