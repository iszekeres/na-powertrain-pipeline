import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_DIR = Path("newlogs") / "cleaned"
OUT_DIR = Path("newlogs") / "output" / "01_tables" / "shift"
TORQUE_DIR = Path("newlogs") / "output" / "02_passes" / "TORQUE"
WEAK_DIR = Path("newlogs") / "output" / "02_passes" / "COMFORT_WEAK"
TORQUE_CURVE_PATH = TORQUE_DIR / "TORQUE_CURVE__GLOBAL_WOT.tsv"
WEAK_SUMMARY_PATH = WEAK_DIR / "COMFORT_WEAKSPOTS__SHIFT_SUMMARY.tsv"

TPS_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]
TPS_AXIS_STR = [str(x) for x in TPS_AXIS]
GEARS = [1, 2, 3, 4, 5, 6]
SHIFT_PAIRS = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
SHIFT_LABELS = {
    (1, 2): "1 -> 2 Shift",
    (2, 3): "2 -> 3 Shift",
    (3, 4): "3 -> 4 Shift",
    (4, 5): "4 -> 5 Shift",
    (5, 6): "5 -> 6 Shift",
}
DOWN_LABELS = {
    (2, 1): "2 -> 1 Shift",
    (3, 2): "3 -> 2 Shift",
    (4, 3): "4 -> 3 Shift",
    (5, 4): "5 -> 4 Shift",
    (6, 5): "6 -> 5 Shift",
}

TORQUE_PEAK_APPROX = 4350.0
FINAL_DRIVE = 3.08
TIRE_DIAMETER_INCH = 32.5

REQUIRED_COLS = [
    "time_s",
    "speed_mph",
    "gear_actual__canon",
    "throttle_pct",
    "engine_rpm__canon",
    "tcc_locked_built",
    "shift_mode_canon",
    "mode_profile",
]


def error_exit(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def assign_tps_bin(value):
    try:
        pct = float(value)
    except (ValueError, TypeError):
        return None
    pct = max(0.0, pct)
    return min(TPS_AXIS, key=lambda x: abs(x - pct))


def load_torque_curve():
    if not TORQUE_CURVE_PATH.exists():
        error_exit(f"Missing torque curve at {TORQUE_CURVE_PATH}")
    try:
        df = pd.read_csv(TORQUE_CURVE_PATH, sep="\t")
    except Exception as exc:
        error_exit(f"Failed to load torque curve: {exc}")
    if df.empty:
        error_exit("Torque curve file is empty")
    if "rpm_bin_center" not in df.columns:
        error_exit("Torque curve missing rpm_bin_center")
    torque_col = None
    for col in ("torque_p90", "torque_mean", "torque"):
        if col in df.columns:
            torque_col = col
            break
    if torque_col is None:
        candidates = [c for c in df.columns if c != "rpm_bin_center"]
        if not candidates:
            error_exit("Torque curve lacks a torque-like column")
        torque_col = candidates[0]
    df = df.dropna(subset=[torque_col, "rpm_bin_center"]).copy()
    if df.empty:
        error_exit("Torque curve emptied after dropping NaNs")
    peak = df[torque_col].max()
    if not np.isfinite(peak) or peak <= 0:
        error_exit("Invalid torque peak in curve")
    df["torque_rel"] = df[torque_col] / peak

    def classify(val):
        if not np.isfinite(val):
            return ("unknown", 0)
        if val < 0.5:
            return ("lug", 1)
        if val < 0.65:
            return ("weak", 2)
        if val < 0.8:
            return ("okay", 3)
        if val < 0.92:
            return ("strong", 4)
        return ("peak", 5)

    bands = df["torque_rel"].apply(classify)
    df["torque_band"] = bands.apply(lambda t: t[0])
    df["torque_band_id"] = bands.apply(lambda t: t[1])
    print(f"[INFO] Loaded torque curve from {TORQUE_CURVE_PATH}")
    return df, peak


def lookup_torque_band(curve_df, rpm_value):
    if curve_df is None or rpm_value is None or not np.isfinite(rpm_value):
        return {"rpm_center": float("nan"), "torque_rel": float("nan"), "band": "unknown", "band_id": 0}
    diffs = (curve_df["rpm_bin_center"] - rpm_value).abs()
    idx = diffs.idxmin()
    row = curve_df.loc[idx]
    return {
        "rpm_center": float(row["rpm_bin_center"]),
        "torque_rel": float(row["torque_rel"]),
        "band": str(row["torque_band"]),
        "band_id": int(row["torque_band_id"]),
    }


def load_weak_summary():
    if not WEAK_SUMMARY_PATH.exists():
        return {}
    df = pd.read_csv(WEAK_SUMMARY_PATH, sep="\t")
    info = {}
    for _, row in df.iterrows():
        key = (row["shift_label"], int(row["tps_bin"]))
        info[key] = {
            "weak_frac": float(row.get("weak_frac", 0.0)),
            "rpm_post_rel_to_peak_mean": float(row.get("rpm_post_rel_to_peak_mean", 0.0)),
            "avg_tcc_locked_post_frac": float(row.get("avg_tcc_locked_post_frac", 0.0)),
        }
    return info


def full_logs():
    files = sorted(CLEAN_DIR.glob("__trans_focus__clean_FULL__*.csv"))
    if not files:
        error_exit("No FULL logs found under newlogs/cleaned")
    return files


def comfort_mask(df):
    return (df["mode_profile"] == "comfort") | (df["shift_mode_canon"] == "normal")


def detect_upshifts(paths, torque_curve):
    events = []
    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            error_exit(f"{path.name} missing columns: {missing}")
        df = df[comfort_mask(df)].copy()
        if df.empty:
            continue
        df = df.sort_values("time_s").reset_index(drop=True)
        df = df.dropna(
            subset=[
                "time_s",
                "speed_mph",
                "gear_actual__canon",
                "engine_rpm__canon",
                "throttle_pct",
            ]
        )
        if df.empty:
            continue
        gear = df["gear_actual__canon"].round().astype("Int64")
        gear_prev = gear.shift(1)
        for idx in df.index[gear_prev.notna()]:
            g_prev = int(gear_prev.loc[idx])
            g_curr = int(gear.loc[idx])
            if (g_prev, g_curr) not in SHIFT_PAIRS:
                continue
            time_shift = float(df.loc[idx, "time_s"])
            pre = df[
                (df["time_s"] >= time_shift - 0.7)
                & (df["time_s"] < time_shift)
                & (gear == g_prev)
            ]
            post = df[
                (df["time_s"] >= time_shift)
                & (df["time_s"] <= time_shift + 1.0)
                & (gear == g_curr)
            ]
            if len(pre) < 4 or len(post) < 4:
                continue
            rpm_pre_max = pre["engine_rpm__canon"].max()
            speed_pre_avg = pre["speed_mph"].mean()
            rpm_post_avg = post["engine_rpm__canon"].mean()
            rpm_post_min = post["engine_rpm__canon"].min()
            speed_post_avg = post["speed_mph"].mean()
            if any(np.isnan(v) for v in (rpm_pre_max, rpm_post_avg, speed_post_avg)):
                continue
            tcc_locked_pre = (pre["tcc_locked_built"] == 1).mean()
            tcc_locked_post = (post["tcc_locked_built"] == 1).mean()
            throttle_at_shift = float(df.loc[idx, "throttle_pct"])
            tps_bin = assign_tps_bin(throttle_at_shift)
            if tps_bin is None:
                continue
            torque = lookup_torque_band(torque_curve, rpm_post_avg)
            weak_flag = int(
                rpm_post_avg < 0.75 * TORQUE_PEAK_APPROX
                and 20.0 <= throttle_at_shift <= 70.0
            )
            shift_label = f"{g_prev}->{g_curr}"
            events.append(
                {
                    "log_name": path.name,
                    "shift_label": shift_label,
                    "gear_from": g_prev,
                    "gear_to": g_curr,
                    "time_s_shift": time_shift,
                    "speed_pre_avg": speed_pre_avg,
                    "speed_post_avg": speed_post_avg,
                    "rpm_pre_max": rpm_pre_max,
                    "rpm_post_min": rpm_post_min,
                    "rpm_post_avg": rpm_post_avg,
                    "rpm_drop": rpm_pre_max - rpm_post_min,
                    "throttle_pct_at_shift": throttle_at_shift,
                    "tps_bin": tps_bin,
                    "tcc_locked_pre_frac": tcc_locked_pre,
                    "tcc_locked_post_frac": tcc_locked_post,
                    "torque_band_post": torque["band"],
                    "torque_band_id_post": torque["band_id"],
                    "weak_shift_flag": weak_flag,
                }
            )
    if not events:
        error_exit("No upshift events detected")
    return pd.DataFrame(events)


def base_rpm_target(tps_bin, peak):
    if tps_bin <= 6:
        rel = 0.55
    elif tps_bin <= 25:
        rel = 0.65
    elif tps_bin <= 44:
        rel = 0.75
    elif tps_bin <= 75:
        rel = 0.85
    else:
        rel = 0.9
    return rel * peak


def rpm_to_mph_in_gear(rpm, gear_to):
    gear_ratios = {1: 4.94, 2: 2.86, 3: 1.91, 4: 1.41, 5: 1.00, 6: 0.75}
    if gear_to not in gear_ratios:
        return float("nan")
    circumference = math.pi * TIRE_DIAMETER_INCH
    speed_ips = rpm * circumference / (gear_ratios[gear_to] * FINAL_DRIVE)
    return speed_ips * 60 / 63360


def compute_bin_stats(events_df, peak, weak_info):
    stats = {}
    for (label, tps), group in events_df.groupby(["shift_label", "tps_bin"]):
        if len(group) == 0:
            continue
        weights = group["torque_band_id_post"].astype(float) * (
            1.0 - 0.5 * group["weak_shift_flag"]
        ).clip(lower=0.1)
        total_weight = weights.sum()
        rpm_post_avg_mean = float(group["rpm_post_avg"].mean())
        rpm_post_weighted = (
            float((group["rpm_post_avg"] * weights).sum() / total_weight)
            if total_weight > 0
            else rpm_post_avg_mean
        )
        speed_post_weighted = (
            float((group["speed_post_avg"] * weights).sum() / total_weight)
            if total_weight > 0
            else float(group["speed_post_avg"].mean())
        )
        stats[(label, tps)] = {
            "n": len(group),
            "n_weak": int(group["weak_shift_flag"].sum()),
            "weak_frac": float(group["weak_shift_flag"].mean()),
            "rpm_post_avg_mean": rpm_post_avg_mean,
            "rpm_post_weighted": rpm_post_weighted,
            "rpm_post_rel_to_peak_mean": rpm_post_avg_mean / peak,
            "speed_post_weighted": speed_post_weighted,
            "avg_tcc_locked_post_frac": float(
                group["tcc_locked_post_frac"].mean()
            ),
        }
        if (label, tps) in weak_info:
            stats[(label, tps)]["weak_frac_ext"] = weak_info[(label, tps)]["weak_frac"]
    return stats


def determine_targets(stats, peak, weak_info):
    targets = {}
    for (g_from, g_to) in SHIFT_PAIRS:
        shift_label = f"{g_from}->{g_to}"
        for tps in TPS_AXIS:
            base = base_rpm_target(tps, peak)
            entry = stats.get((shift_label, tps))
            alpha_data = 0.0
            if entry:
                n = entry["n"]
                if n >= 8:
                    alpha_data = 0.7
                elif n >= 3:
                    alpha_data = 0.5
                else:
                    alpha_data = 0.3
            weak_local = entry["weak_frac"] if entry else 0.0
            weak_ext = weak_info.get((shift_label, tps), {}).get("weak_frac", 0.0)
            weak_combined = max(weak_local, weak_ext)
            if weak_combined > 0.5:
                alpha_data *= 0.6
            data_rpm = entry["rpm_post_weighted"] if entry else base
            rpm_target = alpha_data * data_rpm + (1 - alpha_data) * base
            rpm_target = np.clip(rpm_target, 0.45 * peak, 1.05 * peak)
            targets[(shift_label, tps)] = {"rpm_target": rpm_target, "gear_to": g_to}
    return targets


def rpm_to_mph_targets(targets):
    mph_targets = {}
    for (shift_label, tps), info in targets.items():
        g_to = info["gear_to"]
        rpm_target = info["rpm_target"]
        mph = rpm_to_mph_in_gear(rpm_target, g_to)
        mph_targets[(shift_label, tps)] = mph
    return mph_targets


def build_shift_table(mph_targets):
    rows = []
    for (g_from, g_to) in SHIFT_PAIRS:
        shift_label = f"{g_from}->{g_to}"
        row_label = SHIFT_LABELS[(g_from, g_to)]
        row = {"mph": row_label}
        for tps in TPS_AXIS:
            val = mph_targets.get((shift_label, tps), np.nan)
            row[str(tps)] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    for tps in TPS_AXIS_STR:
        series = df[tps].astype(float)
        if series.notna().any():
            series = series.interpolate(limit_direction="both")
            series = series.ffill().bfill()
            for i in range(1, len(series)):
                if series.iloc[i] < series.iloc[i - 1]:
                    series.iloc[i] = series.iloc[i - 1]
            df[tps] = series
    return df


def build_down_table(up_df):
    rows = []
    for (to_gear, from_gear) in [(2, 1), (3, 2), (4, 3), (5, 4), (6, 5)]:
        row_label = DOWN_LABELS[(to_gear, from_gear)]
        up_label = SHIFT_LABELS[(from_gear, to_gear)]
        row = {"mph": row_label}
        for tps in TPS_AXIS:
            up_val = up_df.loc[up_df["mph"] == up_label, str(tps)]
            if up_val.empty or pd.isna(up_val.iloc[0]):
                row[str(tps)] = np.nan
                continue
            val = float(up_val.iloc[0])
            if tps <= 6:
                gap = 5.0
            elif tps <= 25:
                gap = 4.0
            elif tps <= 50:
                gap = 3.5
            elif tps <= 75:
                gap = 3.0
            else:
                gap = 2.5
            down = val - gap
            down = max(down, 0.0)
            if down > val - 1.0:
                down = val - 1.0
            row[str(tps)] = down
        rows.append(row)
    df = pd.DataFrame(rows)
    for tps in TPS_AXIS_STR:
        series = df[tps].astype(float)
        if series.notna().any():
            series = series.interpolate(limit_direction="both")
            series = series.ffill().bfill()
            df[tps] = series
    return df


def round_table(df):
    out = df.copy()
    for col in TPS_AXIS_STR:
        out[col] = out[col].apply(lambda v: round(v, 1) if pd.notna(v) else v)
    return out


def main():
    ensure_dirs()
    torque_curve, torque_peak = load_torque_curve()
    weak_info = load_weak_summary()
    paths = full_logs()
    events_df = detect_upshifts(paths, torque_curve)
    stats = compute_bin_stats(events_df, torque_peak, weak_info)
    targets = determine_targets(stats, torque_peak, weak_info)
    mph_targets = rpm_to_mph_targets(targets)
    up_df = build_shift_table(mph_targets)
    down_df = build_down_table(up_df)
    up_df = round_table(up_df)
    down_df = round_table(down_df)
    up_path = OUT_DIR / "SHIFT_TABLES__UP__Throttle17__COMFORT_TORQUESEED.tsv"
    down_path = OUT_DIR / "SHIFT_TABLES__DOWN__Throttle17__COMFORT_TORQUESEED.tsv"
    up_df.to_csv(up_path, sep="\t", index=False)
    down_df.to_csv(down_path, sep="\t", index=False)
    print(f"[INFO] Logs processed: {len(paths)}")
    print(f"[INFO] Upshift events: {len(events_df)}")
    if not events_df.empty:
        summary = events_df.groupby("shift_label")["weak_shift_flag"].agg(
            total="count", weak="sum"
        )
        for shift_label, row in summary.iterrows():
            frac = row["weak"] / row["total"]
            print(
                f"[INFO] {shift_label}: n={int(row['total'])}, weak={int(row['weak'])}, weak_frac={frac:.2f}"
            )
    for row_label in up_df["mph"]:
        values = up_df.loc[up_df["mph"] == row_label, TPS_AXIS_STR].iloc[0]
        valid = values[values.notna()]
        if valid.empty:
            continue
        print(f"[INFO] UP {row_label} min/max = {valid.min():.1f}/{valid.max():.1f}")
    for row_label in down_df["mph"]:
        values = down_df.loc[down_df["mph"] == row_label, TPS_AXIS_STR].iloc[0]
        valid = values[values.notna()]
        if valid.empty:
            continue
        print(f"[INFO] DOWN {row_label} min/max = {valid.min():.1f}/{valid.max():.1f}")
    print(f"[INFO] Wrote UP seed to {up_path}")
    print(f"[INFO] Wrote DOWN seed to {down_path}")


if __name__ == "__main__":
    main()
