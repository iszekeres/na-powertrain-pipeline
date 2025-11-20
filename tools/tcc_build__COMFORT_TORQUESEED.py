import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_DIR = Path("newlogs") / "cleaned"
OUT_DIR = Path("newlogs") / "output" / "01_tables" / "tcc"
TORQUE_DIR = Path("newlogs") / "output" / "02_passes" / "TORQUE"
TORQUE_CURVE_PATH = TORQUE_DIR / "TORQUE_CURVE__GLOBAL_WOT.tsv"

TPS_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]
TPS_AXIS_STR = [str(x) for x in TPS_AXIS]
GEARS = [3, 4, 5, 6]

TORQUE_PEAK_APPROX = 4350.0
APPLY_SENTINEL = 318
RELEASE_SENTINEL = 317

REQUIRED_COLS = [
    "time_s",
    "speed_mph",
    "gear_actual__canon",
    "throttle_pct",
    "engine_rpm__canon",
    "tcc_locked_built",
    "tcc_slip_fused",
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
    closest = min(TPS_AXIS, key=lambda v: abs(v - pct))
    return int(closest)


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
        error_exit("Torque curve missing rpm_bin_center column")
    torque_col = None
    for c in ("torque_p90", "torque_mean", "torque"):
        if c in df.columns:
            torque_col = c
            break
    if torque_col is None:
        error_exit("Torque curve lacks a torque column (torque_p90/torque_mean/torque)")
    df = df.dropna(subset=[torque_col, "rpm_bin_center"]).copy()
    if df.empty:
        error_exit("Torque curve is empty after dropping NaNs")
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
    return df


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


def comfort_mask(df):
    return (df["mode_profile"] == "comfort") | (df["shift_mode_canon"] == "normal")


def full_log_paths():
    files = sorted(CLEAN_DIR.glob("__trans_focus__clean_FULL__*.csv"))
    if not files:
        error_exit("No FULL logs found under newlogs/cleaned")
    return files


def detect_events(files, torque_curve):
    events = []
    apply_count = 0
    release_count = 0
    for path in files:
        df = pd.read_csv(path, low_memory=False)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            error_exit(f"{path.name} missing columns: {missing}")
        df = df[comfort_mask(df)].copy()
        if df.empty:
            continue
        gear_mask = df["gear_actual__canon"].isin(GEARS)
        speed_mask = df["speed_mph"] >= 25.0
        base = df[gear_mask & speed_mask].copy()
        if base.empty:
            continue
        base = base.sort_values("time_s").reset_index(drop=True)
        base["gear_round"] = base["gear_actual__canon"].round().astype("Int64")
        base["locked"] = base["tcc_locked_built"].fillna(0).astype(int)
        base["locked_prev"] = base["locked"].shift(1).fillna(0).astype(int)
        for gear in GEARS:
            gear_df = base[base["gear_round"] == gear].copy()
            if gear_df.empty:
                continue
            gear_df = gear_df.dropna(
                subset=["time_s", "speed_mph", "engine_rpm__canon", "tcc_slip_fused", "throttle_pct"]
            )
            if gear_df.empty:
                continue
            gear_df = gear_df.reset_index(drop=True)
            apply_mask = (gear_df["locked_prev"] == 0) & (gear_df["locked"] == 1)
            release_mask = (gear_df["locked_prev"] == 1) & (gear_df["locked"] == 0)
            for idx in gear_df[apply_mask].index:
                record = process_event(gear_df, idx, "apply", torque_curve, path.name)
                if record:
                    events.append(record)
                    apply_count += 1
            for idx in gear_df[release_mask].index:
                record = process_event(gear_df, idx, "release", torque_curve, path.name)
                if record:
                    events.append(record)
                    release_count += 1
    if not events:
        error_exit("No TCC events found after filtering")
    return pd.DataFrame(events), apply_count, release_count


def process_event(df, idx, event_type, curve, log_name):
    row = df.loc[idx]
    time_edge = float(row["time_s"])
    mph_edge = float(row["speed_mph"])
    rpm_edge = float(row["engine_rpm__canon"])
    thr_edge = float(row["throttle_pct"])
    slip_edge = float(row["tcc_slip_fused"])
    if any(np.isnan(v) for v in (mph_edge, rpm_edge, thr_edge, slip_edge)):
        return None
    pre = df[(df["time_s"] >= time_edge - 0.5) & (df["time_s"] < time_edge)]
    post = df[(df["time_s"] >= time_edge) & (df["time_s"] <= time_edge + 0.5)]
    rpm_pre_mean = pre["engine_rpm__canon"].mean()
    rpm_post_mean = post["engine_rpm__canon"].mean()
    slip_pre_mean = pre["tcc_slip_fused"].mean()
    slip_post_mean = post["tcc_slip_fused"].mean()
    if any(np.isnan(v) for v in (rpm_pre_mean, rpm_post_mean, slip_pre_mean, slip_post_mean)):
        return None
    quality_flag = 0
    if event_type == "apply":
        if slip_pre_mean > 80 and slip_post_mean < 60:
            quality_flag = 1
    else:
        if slip_post_mean > 80:
            quality_flag = 1
    torque = lookup_torque_band(curve, rpm_edge)
    tps_bin = assign_tps_bin(thr_edge)
    if tps_bin is None:
        return None
    return {
        "log_name": log_name,
        "gear": int(row["gear_round"]),
        "event_type": event_type,
        "time_s_edge": time_edge,
        "mph_edge": mph_edge,
        "rpm_edge": rpm_edge,
        "throttle_pct": thr_edge,
        "tps_bin": tps_bin,
        "slip_edge": slip_edge,
        "rpm_pre_mean": rpm_pre_mean,
        "rpm_post_mean": rpm_post_mean,
        "slip_pre_mean": slip_pre_mean,
        "slip_post_mean": slip_post_mean,
        "quality_flag": quality_flag,
        "torque_rel": torque["torque_rel"],
        "torque_band": torque["band"],
        "torque_band_id": torque["band_id"],
    }


def new_table(kind):
    suffix = {3: "3rd", 4: "4th", 5: "5th", 6: "6th"}
    rows = []
    for gear in GEARS:
        row = {"row_label": f"{suffix[gear]} {kind}"}
        for col in TPS_AXIS_STR:
            row[col] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def weighted_seed(events_df, event_type, table):
    subset = events_df[events_df["event_type"] == event_type]
    for (gear, tps), group in subset.groupby(["gear", "tps_bin"]):
        if gear not in GEARS or tps not in TPS_AXIS:
            continue
        label = f"{gear_label(gear)} {event_type.capitalize()}"
        weights = []
        mphs = []
        for _, row in group.iterrows():
            torque_w = max(1, int(row["torque_band_id"]))
            quality_w = 1.5 if int(row["quality_flag"]) == 1 else 1.0
            w = torque_w * quality_w
            weights.append(w)
            mphs.append(float(row["mph_edge"]))
        total_weight = sum(weights)
        if total_weight <= 0:
            continue
        mph_avg = sum(m * w for m, w in zip(mphs, weights)) / total_weight
        table.loc[table["row_label"] == label, str(tps)] = mph_avg


def gear_label(gear):
    return {3: "3rd", 4: "4th", 5: "5th", 6: "6th"}[gear]


def fill_monotonic(table):
    for gear in GEARS:
        row_label = f"{gear_label(gear)} Apply"
        idx = table.index[table["row_label"] == row_label]
        if idx.empty:
            continue
        series = table.loc[idx[0], TPS_AXIS_STR].astype(float)
        if series.isna().all():
            if not backfill_from_neighbor(table, row_label, "Apply", gear):
                continue
            idx = table.index[table["row_label"] == row_label]
            series = table.loc[idx[0], TPS_AXIS_STR].astype(float)
        series = series.interpolate(limit_direction="both")
        series = series.ffill().bfill()
        for i in range(1, len(series)):
            if series.iloc[i] < series.iloc[i - 1]:
                series.iloc[i] = series.iloc[i - 1]
        table.loc[table["row_label"] == row_label, TPS_AXIS_STR] = series.tolist()
    for gear in GEARS:
        row_label = f"{gear_label(gear)} Release"
        idx = table.index[table["row_label"] == row_label]
        if idx.empty:
            continue
        series = table.loc[idx[0], TPS_AXIS_STR].astype(float)
        if series.isna().all():
            if not backfill_from_neighbor(table, row_label, "Release", gear):
                continue
            idx = table.index[table["row_label"] == row_label]
            series = table.loc[idx[0], TPS_AXIS_STR].astype(float)
        series = series.interpolate(limit_direction="both")
        series = series.ffill().bfill()
        table.loc[table["row_label"] == row_label, TPS_AXIS_STR] = series.tolist()


def backfill_from_neighbor(table, label, kind, gear):
    for delta in (-1, 1):
        neighbor = gear + delta
        if neighbor not in GEARS:
            continue
        neighbor_label = f"{gear_label(neighbor)} {kind}"
        neighbor_row = table.loc[table["row_label"] == neighbor_label, TPS_AXIS_STR]
        if neighbor_row.empty:
            continue
        neighbor_vals = neighbor_row.iloc[0]
        if neighbor_vals.isna().all():
            continue
        table.loc[table["row_label"] == label, TPS_AXIS_STR] = neighbor_vals.values
        return True
    return False


def enforce_lockout(apply_df, release_df):
    for gear in GEARS:
        apply_label = f"{gear_label(gear)} Apply"
        release_label = f"{gear_label(gear)} Release"
        for tps in TPS_AXIS_STR:
            if int(tps) >= 81:
                apply_df.loc[apply_df["row_label"] == apply_label, tps] = APPLY_SENTINEL
                release_df.loc[release_df["row_label"] == release_label, tps] = RELEASE_SENTINEL


def recompute_release(apply_df, release_df):
    for gear in GEARS:
        apply_label = f"{gear_label(gear)} Apply"
        release_label = f"{gear_label(gear)} Release"
        for tps in TPS_AXIS_STR:
            val = apply_df.loc[apply_df["row_label"] == apply_label, tps].iloc[0]
            if pd.isna(val):
                continue
            if is_sentinel(val):
                release_df.loc[release_df["row_label"] == release_label, tps] = RELEASE_SENTINEL
                continue
            val = float(val)
            if int(val) >= 300:
                release_df.loc[release_df["row_label"] == release_label, tps] = RELEASE_SENTINEL
                continue
            tps_bin = int(float(tps))
            if tps_bin <= 12:
                gap = 3.0
            elif tps_bin <= 37:
                gap = 4.0
            elif tps_bin <= 75:
                gap = 5.0
            else:
                gap = 6.0
            rel = val - gap
            rel = max(rel, 0.0)
            if rel > val - 1.1:
                rel = val - 1.1
            release_df.loc[release_df["row_label"] == release_label, tps] = rel


def is_sentinel(value):
    if pd.isna(value):
        return False
    try:
        num = float(value)
    except Exception:
        return False
    return abs(num - 317) < 1e-6 or abs(num - 318) < 1e-6


def normalize_table(df):
    out = df.copy()
    for col in TPS_AXIS_STR:
        out[col] = out[col].apply(lambda v: round(v, 1) if pd.notna(v) and not is_sentinel(v) else v)
    return out


def main():
    ensure_dirs()
    torque_curve = load_torque_curve()
    files = full_log_paths()
    events_df, apply_events, release_events = detect_events(files, torque_curve)
    apply_table = new_table("Apply")
    release_table = new_table("Release")
    weighted_seed(events_df, "apply", apply_table)
    weighted_seed(events_df, "release", release_table)
    fill_monotonic(apply_table)
    fill_monotonic(release_table)
    enforce_lockout(apply_table, release_table)
    recompute_release(apply_table, release_table)
    norm_apply = normalize_table(apply_table)
    norm_release = normalize_table(release_table)
    norm_apply = norm_apply.rename(columns={"row_label": "mph"})
    norm_release = norm_release.rename(columns={"row_label": "mph"})
    norm_apply.to_csv(OUT_DIR / f"TCC_APPLY__Throttle17__COMFORT_TORQUESEED.tsv", sep="\t", index=False)
    norm_release.to_csv(OUT_DIR / f"TCC_RELEASE__Throttle17__COMFORT_TORQUESEED.tsv", sep="\t", index=False)
    print(f"[INFO] Logs processed: {len(files)}")
    print(f"[INFO] APPLY events: {apply_events}, RELEASE events: {release_events}")
    for gear in GEARS:
        apply_vals = norm_apply.loc[norm_apply["mph"] == f"{gear_label(gear)} Apply", TPS_AXIS_STR].iloc[0]
        release_vals = norm_release.loc[norm_release["mph"] == f"{gear_label(gear)} Release", TPS_AXIS_STR].iloc[0]
        apply_vals = apply_vals[apply_vals.apply(lambda v: pd.notna(v) and not is_sentinel(v))]
        release_vals = release_vals[release_vals.apply(lambda v: pd.notna(v) and not is_sentinel(v))]
        if not apply_vals.empty:
            print(
                f"[INFO] Gear {gear} APPLY min/max = {apply_vals.min():.1f}/{apply_vals.max():.1f}"
            )
        if not release_vals.empty:
            print(
                f"[INFO] Gear {gear} RELEASE min/max = {release_vals.min():.1f}/{release_vals.max():.1f}"
            )
    print(f"[INFO] Wrote APPLY seed to {OUT_DIR / 'TCC_APPLY__Throttle17__COMFORT_TORQUESEED.tsv'}")
    print(f"[INFO] Wrote RELEASE seed to {OUT_DIR / 'TCC_RELEASE__Throttle17__COMFORT_TORQUESEED.tsv'}")


if __name__ == "__main__":
    main()
