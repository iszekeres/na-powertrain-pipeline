#!/usr/bin/env python3
"""
Highway intent + harshness analysis.

Modes:
  --do-intent    : annotate intent episodes with driver demand + torque context
  --do-harshness : build shift harshness events + heatmap from prepped logs

Dependencies: pandas, numpy, argparse, pathlib (standard lib otherwise).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# Thresholds for intent/kickdown logic
MIN_PEDAL_FOR_KICKDOWN = 20.0  # percent
MIN_SPEED_FOR_HIGHWAY = 45.0  # mph
MIN_GAIN_FRACTION = 0.25  # 25% of requested torque
MIN_GAIN_ABS = 40.0  # absolute torque delta (same units as driver-demand map)


# -----------------------------------------------------------------------------
# Common utilities
# -----------------------------------------------------------------------------


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for cand in candidates:
        if cand in df.columns:
            return cand
    return None


def load_driver_demand_table(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=None, comment="%")
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    header_idx = None
    for i in range(len(df)):
        row_vals = df.iloc[i, 1:]
        if pd.to_numeric(row_vals, errors="coerce").notna().all():
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not locate speed axis row in driver demand table.")
    speed_axis = pd.to_numeric(df.iloc[header_idx, 1:], errors="coerce").to_numpy(dtype=float)
    data = df.iloc[header_idx + 1 :].reset_index(drop=True)
    pedal_axis = pd.to_numeric(data.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    values = data.iloc[:, 1:].to_numpy(dtype=float)
    if values.shape != (len(pedal_axis), len(speed_axis)):
        raise ValueError(
            f"Driver demand table shape mismatch: values {values.shape}, "
            f"expected ({len(pedal_axis)}, {len(speed_axis)})"
        )
    return pedal_axis, speed_axis, values


def dd_lookup(pedal: float, speed: float, pedal_axis: np.ndarray, speed_axis: np.ndarray, values: np.ndarray) -> float:
    if pd.isna(pedal) or pd.isna(speed):
        return math.nan
    p = float(np.clip(pedal, pedal_axis.min(), pedal_axis.max()))
    s = float(np.clip(speed, speed_axis.min(), speed_axis.max()))
    pi = np.searchsorted(pedal_axis, p, side="right") - 1
    si = np.searchsorted(speed_axis, s, side="right") - 1
    pi = np.clip(pi, 0, len(pedal_axis) - 2)
    si = np.clip(si, 0, len(speed_axis) - 2)
    p0, p1 = pedal_axis[pi], pedal_axis[pi + 1]
    s0, s1 = speed_axis[si], speed_axis[si + 1]
    t00 = values[pi, si]
    t01 = values[pi, si + 1]
    t10 = values[pi + 1, si]
    t11 = values[pi + 1, si + 1]
    wp = 0 if p1 == p0 else (p - p0) / (p1 - p0)
    ws = 0 if s1 == s0 else (s - s0) / (s1 - s0)
    return float(
        (1 - wp) * (1 - ws) * t00
        + (1 - wp) * ws * t01
        + wp * (1 - ws) * t10
        + wp * ws * t11
    )


def load_torque_surface(path: Path) -> dict:
    df = pd.read_csv(path)
    gear_col = pick_col(df, ["gear", "gear_actual", "gear_mode"])
    speed_col = pick_col(df, ["speed_mph_center", "speed_mph", "speed_bin_center", "speed_center"])
    pedal_col = pick_col(df, ["pedal_pct_center", "pedal_pct", "pedal_bin_center", "pedal_center"])
    torque_col = pick_col(
        df,
        [
            "torque_engine",
            "torque_axle",
            "torque_wheel",
            "torque_nm",
            "hybrid_engine_torque",
            "physics_engine_torque_median",
            "eng_torque_mean",
            "axle_torque_mean",
        ],
    )
    missing = []
    if gear_col is None:
        missing.append("gear column (gear/gear_actual/gear_mode)")
    if speed_col is None:
        missing.append("speed column (speed_mph_center/speed_mph/speed_bin_center/speed_center)")
    if pedal_col is None:
        missing.append("pedal column (pedal_pct_center/pedal_pct/pedal_bin_center/pedal_center)")
    if torque_col is None:
        missing.append("torque column (torque_engine/torque_axle/torque_wheel/torque_nm/...)")
    if missing:
        raise ValueError(f"Torque surface missing required headers: {missing}")
    return {
        "df": df,
        "gear_col": gear_col,
        "speed_col": speed_col,
        "pedal_col": pedal_col,
        "torque_col": torque_col,
    }


def torque_lookup(surface: dict, gear: int, speed_mph: float, pedal_pct: float) -> float:
    df = surface["df"]
    gcol, scol, pcol, tcol = surface["gear_col"], surface["speed_col"], surface["pedal_col"], surface["torque_col"]
    sub = df[df[gcol] == gear]
    if sub.empty or pd.isna(speed_mph) or pd.isna(pedal_pct):
        return math.nan
    idx_speed = (sub[scol].astype(float) - float(speed_mph)).abs().idxmin()
    speed_val = sub.loc[idx_speed, scol]
    sub2 = sub[sub[scol] == speed_val]
    idx_pedal = (sub2[pcol].astype(float) - float(pedal_pct)).abs().idxmin()
    row = sub2.loc[idx_pedal]
    return float(row[tcol]) if pd.notna(row[tcol]) else math.nan


# -----------------------------------------------------------------------------
# Intent mode
# -----------------------------------------------------------------------------


def run_intent_mode(episodes_path: Path, dd_path: Path, surface_path: Path, out_path: Optional[Path]) -> None:
    df_ep = pd.read_csv(episodes_path)
    gear_col = pick_col(df_ep, ["gear_entry", "gear_mode", "gear", "gear_actual__canon", "gear_start"])
    speed_col = pick_col(
        df_ep,
        [
            "speed_mph_center",
            "speed_mph_med",
            "speed_mph_entry",
            "speed_mph_mean",
            "speed_mph",
            "speed_start_mph",
        ],
    )
    pedal_col = pick_col(
        df_ep,
        [
            "pedal_pct_center",
            "pedal_pct_med",
            "pedal_pct_entry",
            "pedal_pct_mean",
            "pedal_pct",
            "pedal_start_pct",
        ],
    )
    mode_col = pick_col(df_ep, ["mode", "pattern", "mode_name"])
    missing = []
    if gear_col is None:
        missing.append("gear column in episodes")
    if speed_col is None:
        missing.append("speed column in episodes")
    if pedal_col is None:
        missing.append("pedal column in episodes")
    if missing:
        raise SystemExit(f"[ERROR] {missing}")

    pedal_axis, speed_axis, dd_vals = load_driver_demand_table(dd_path)
    surface = load_torque_surface(surface_path)
    gears_surface = sorted(surface["df"][surface["gear_col"]].dropna().unique().astype(int))

    rows = []
    skipped = 0
    for _, r in df_ep.iterrows():
        g = r.get(gear_col)
        v = r.get(speed_col)
        p = r.get(pedal_col)
        if pd.isna(g) or pd.isna(v) or pd.isna(p):
            skipped += 1
            continue
        g = int(g)
        v = float(v)
        p = float(p)
        T_req = dd_lookup(p, v, pedal_axis, speed_axis, dd_vals)
        T_act = torque_lookup(surface, g, v, p)
        best_gear = math.nan
        T_best = math.nan
        best_val = -np.inf
        for gtest in gears_surface:
            val = torque_lookup(surface, gtest, v, p)
            if pd.isna(val):
                continue
            if val > best_val:
                best_val = val
                T_best = val
                best_gear = gtest
        torque_gain = T_best - T_act if pd.notna(T_best) and pd.notna(T_act) else math.nan
        deficit_req_vs_act = T_req - T_act if pd.notna(T_req) and pd.notna(T_act) else math.nan
        interesting = (p >= MIN_PEDAL_FOR_KICKDOWN) and (v >= MIN_SPEED_FOR_HIGHWAY)
        should_downshift = (
            interesting
            and pd.notna(best_gear)
            and best_gear != g
            and pd.notna(torque_gain)
            and (
                (torque_gain >= MIN_GAIN_ABS)
                and (torque_gain >= MIN_GAIN_FRACTION * max(T_req if pd.notna(T_req) else 0, 1.0))
            )
        )
        rows.append(
            {
                "speed_mph_ref": v,
                "pedal_pct_ref": p,
                "gear_episode": g,
                "best_gear": best_gear,
                "T_req": T_req,
                "T_act": T_act,
                "T_best": T_best,
                "torque_gain_best_vs_act": torque_gain,
                "deficit_req_vs_act": deficit_req_vs_act,
                "should_have_downshifted": bool(should_downshift),
                "mode": r.get(mode_col) if mode_col else math.nan,
            }
        )

    df_rows = pd.DataFrame(rows)
    df_out = pd.concat([df_ep.reset_index(drop=True), df_rows.reset_index(drop=True)], axis=1)
    if out_path is None:
        out_path = episodes_path.with_name(f"{episodes_path.stem}__intent_kickdown_annotated.csv")
    df_out.to_csv(out_path, index=False)

    flagged = df_out[df_out["should_have_downshifted"]].shape[0]
    total = len(df_rows)
    print(f"[INFO] Intent episodes processed: {total} (skipped {skipped}), flagged {flagged} should-have-downshifted.")
    if flagged:
        by_gear = df_out[df_out["should_have_downshifted"]]["gear_episode"].value_counts()
        print("[INFO] Flagged by gear:")
        for g, c in by_gear.items():
            print(f"  Gear {int(g)}: {c}")
    print(f"[OK] Wrote annotated episodes to {out_path}")


# -----------------------------------------------------------------------------
# Harshness mode
# -----------------------------------------------------------------------------


def compute_dt(time_s: pd.Series) -> pd.Series:
    dt = time_s.diff()
    dt[dt < 0] = np.nan
    dt = dt.ffill()
    dt = dt.fillna(dt.median(skipna=True))
    dt = dt.clip(0, 1.0)
    return dt


def run_harshness_mode(prepped_dir: Path, out_events: Optional[Path], out_heatmap: Optional[Path], out_dir: Optional[Path]) -> None:
    csvs = [p for p in prepped_dir.glob("*.csv") if p.is_file() and p.stat().st_size > 1024]
    if not csvs:
        raise SystemExit(f"[ERROR] No prepped CSVs found in {prepped_dir}")

    events = []
    for csv in csvs:
        df = pd.read_csv(csv, low_memory=False)
        time_col = pick_col(df, ["time_s", "Time", "time"])
        speed_col = pick_col(df, ["speed_mph", "Vehicle Speed (SAE)", "Vehicle Speed"])
        gear_col = pick_col(df, ["gear_actual__canon", "Trans Current Gear", "gear"])
        pedal_col = pick_col(df, ["pedal_pct", "Accelerator Pedal Position"])
        if not all([time_col, speed_col, gear_col]):
            print(f"[WARN] Skipping {csv.name} (missing time/speed/gear).")
            continue
        df = df.sort_values(time_col).reset_index(drop=True)
        time_s = pd.to_numeric(df[time_col], errors="coerce")
        speed_mph = pd.to_numeric(df[speed_col], errors="coerce")
        gear = pd.to_numeric(df[gear_col], errors="coerce").astype("Int64")
        dt_s = compute_dt(time_s)
        speed_mps = speed_mph * 0.44704
        accel = speed_mps.diff() / dt_s
        jerk = accel.diff() / dt_s

        for i in range(1, len(df)):
            if gear.iloc[i] != gear.iloc[i - 1]:
                from_g = gear.iloc[i - 1]
                to_g = gear.iloc[i]
                t_shift = time_s.iloc[i]
                t_start = t_shift - 0.3
                t_end = t_shift + 0.7
                window = df[(time_s >= t_start) & (time_s <= t_end)]
                accel_win = accel.loc[window.index]
                jerk_win = jerk.loc[window.index]
                speed_pre = speed_mph.loc[window.index].iloc[0] if not window.empty else np.nan
                pedal_ref = (
                    pd.to_numeric(df[pedal_col], errors="coerce").loc[window.index].iloc[0]
                    if pedal_col and not window.empty
                    else np.nan
                )
                max_abs_jerk = np.nanmax(np.abs(jerk_win)) if len(jerk_win.dropna()) else np.nan
                rms_jerk = math.sqrt(np.nanmean(np.square(jerk_win))) if len(jerk_win.dropna()) else np.nan
                max_accel = np.nanmax(accel_win) if len(accel_win.dropna()) else np.nan
                min_accel = np.nanmin(accel_win) if len(accel_win.dropna()) else np.nan
                events.append(
                    {
                        "log_name": csv.name,
                        "from_gear": from_g,
                        "to_gear": to_g,
                        "t_shift": t_shift,
                        "speed_mph_pre": speed_pre,
                        "pedal_pct_ref": pedal_ref,
                        "max_abs_jerk": max_abs_jerk,
                        "rms_jerk": rms_jerk,
                        "max_accel": max_accel,
                        "min_accel": min_accel,
                    }
                )

    if not events:
        print("[WARN] No shift events detected.")
        return

    df_events = pd.DataFrame(events)
    base_dir = out_dir if out_dir else prepped_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    def_speed = base_dir / "shift_harshness_events.csv"
    out_events = out_events or def_speed
    df_events.to_csv(out_events, index=False)

    def bin_center(val, width, max_cap=None):
        if pd.isna(val):
            return math.nan
        if max_cap is not None and val >= max_cap:
            return max_cap + width / 2
        return width * (math.floor(val / width) + 0.5)

    df_events["speed_bin_center"] = df_events["speed_mph_pre"].apply(lambda v: bin_center(v, 10, 100))
    df_events["pedal_bin_center"] = df_events["pedal_pct_ref"].apply(lambda v: bin_center(v, 10, 100))

    grp = df_events.groupby(
        ["from_gear", "to_gear", "speed_bin_center", "pedal_bin_center"], dropna=False, observed=False
    )
    heatmap = grp.agg(
        n_events=("max_abs_jerk", "count"),
        mean_max_abs_jerk=("max_abs_jerk", "mean"),
        p90_max_abs_jerk=("max_abs_jerk", lambda x: np.nanpercentile(x.dropna(), 90) if len(x.dropna()) else np.nan),
        mean_rms_jerk=("rms_jerk", "mean"),
        mean_speed_mph_pre=("speed_mph_pre", "mean"),
        mean_pedal_pct_ref=("pedal_pct_ref", "mean"),
    ).reset_index()

    def_heat = base_dir / "shift_harshness_heatmap.csv"
    out_heatmap = out_heatmap or def_heat
    heatmap.to_csv(out_heatmap, index=False)

    print(f"[INFO] Total shift events: {len(df_events)}")
    pair_counts = df_events.groupby(["from_gear", "to_gear"]).size()
    for (f, t), c in pair_counts.items():
        print(f"  {int(f)} -> {int(t)}: {c} events")
    print(f"[OK] Wrote events to {out_events}")
    print(f"[OK] Wrote heatmap to {out_heatmap}")
    print("Use the heatmap to spot gear pairs / speed/pedal bins with highest max_abs_jerk (harshness).")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Highway intent + harshness analysis")
    parser.add_argument("--do-intent", action="store_true", help="Annotate intent episodes")
    parser.add_argument("--episodes", type=str, help="Path to episodes CSV")
    parser.add_argument("--driver-demand", type=str, help="Path to driver_demand_normal.csv")
    parser.add_argument("--torque-surface", type=str, help="Path to SPEEDSPACE torque surface CSV")
    parser.add_argument("--out-intent", type=str, help="Optional output CSV for annotated episodes")

    parser.add_argument("--do-harshness", action="store_true", help="Build shift harshness events + heatmap")
    parser.add_argument(
        "--prepped-dir",
        type=str,
        default="newlogs/highway_MAX_analysis/prepped",
        help="Directory containing prepped logs",
    )
    parser.add_argument("--out-dir", type=str, help="Base output directory for harshness artifacts")
    parser.add_argument("--out-harsh-events", type=str, help="Optional output CSV for events")
    parser.add_argument("--out-harsh-heatmap", type=str, help="Optional output CSV for heatmap")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.do_intent and not args.do_harshness:
        print("Nothing to do; specify --do-intent and/or --do-harshness.", file=sys.stderr)
        return

    if args.do_intent:
        if not (args.episodes and args.driver_demand and args.torque_surface):
            raise SystemExit("[ERROR] --do-intent requires --episodes, --driver-demand, and --torque-surface.")
        run_intent_mode(
            Path(args.episodes),
            Path(args.driver_demand),
            Path(args.torque_surface),
            Path(args.out_intent) if args.out_intent else None,
        )

    if args.do_harshness:
        run_harshness_mode(
            Path(args.prepped_dir),
            Path(args.out_harsh_events) if args.out_harsh_events else None,
            Path(args.out_harsh_heatmap) if args.out_harsh_heatmap else None,
            Path(args.out_dir) if args.out_dir else None,
        )


if __name__ == "__main__":
    main()
