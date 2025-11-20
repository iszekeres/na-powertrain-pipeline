import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

FULL_DIR = Path("newlogs") / "cleaned"
OUT_DIR = Path("newlogs") / "output" / "02_passes" / "COMFORT_WEAK"
EVENT_CSV = OUT_DIR / "COMFORT_WEAKSPOTS__SHIFT_EVENTS.csv"
SHIFT_SUMMARY = OUT_DIR / "COMFORT_WEAKSPOTS__SHIFT_SUMMARY.tsv"
CRUISE_TCC = OUT_DIR / "COMFORT_WEAKSPOTS__CRUISE_TCC.tsv"
SUMMARY_TXT = OUT_DIR / "SUMMARY.txt"

SHIFT_FINAL_UP = (
    Path("newlogs")
    / "output"
    / "01_tables"
    / "shift"
    / "SHIFT_TABLES__UP__Throttle17__COMFORT_FINAL.tsv"
)
TPS_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]
SPEED_BANDS = [(55, 65), (65, 75), (75, 85), (85, 95), (95, 105)]
GEARS_CRUISE = [3, 4, 5, 6]
RPM_PEAK = 4350.0

REQUIRED_COLS = [
    "time_s",
    "speed_mph",
    "gear_actual__canon",
    "throttle_pct",
    "engine_rpm__canon",
    "tcc_locked_built",
    "tcc_slip_fused",
    "shift_mode_canon",
    "shift_mode_state_canon",
    "mode_profile",
]

TORQUE_CURVE_GLOBAL_PATH = os.path.join(
    "newlogs", "output", "02_passes", "TORQUE", "TORQUE_CURVE__GLOBAL_WOT.tsv"
)


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def comfort_mask(df):
    return (df["mode_profile"] == "comfort") | (df["shift_mode_canon"] == "normal")


def assign_tps_bin(value):
    try:
        pct = float(value)
    except (ValueError, TypeError):
        return None
    pct = max(0.0, pct)
    for b in TPS_AXIS:
        if pct <= b:
            return int(b)
    return 100


def find_speed_band(speed):
    for lo, hi in SPEED_BANDS:
        if lo <= speed < hi:
            return lo, hi
    return None


def load_shift_table():
    if not SHIFT_FINAL_UP.exists():
        return pd.DataFrame()
    df = pd.read_csv(SHIFT_FINAL_UP, sep="\t")
    df = df.rename(columns={df.columns[0]: "row_label"})
    df.columns = ["row_label"] + [int(float(c)) for c in df.columns[1:]]
    return df


def load_global_torque_curve(path=TORQUE_CURVE_GLOBAL_PATH):
    if not os.path.exists(path):
        print(f"[WARN] Global torque curve not found at {path}; running in RPM-only mode.")
        return None
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as exc:
        print(f"[WARN] Failed to read torque curve at {path}: {exc}; running RPM-only.")
        return None
    if "rpm_bin_center" not in df.columns:
        print(f"[WARN] rpm_bin_center missing in {path}; torque-based bands disabled.")
        return None
    torque_col = None
    for c in ("torque_p90", "torque_mean", "torque"):
        if c in df.columns:
            torque_col = c
            break
    if torque_col is None:
        print(f"[WARN] No torque column in {path}; torque-based bands disabled.")
        return None
    df = df.copy()
    peak = df[torque_col].max()
    if not np.isfinite(peak) or peak <= 0:
        print(f"[WARN] Invalid global torque peak in {path}; torque bands disabled.")
        return None
    df["torque_rel"] = df[torque_col] / peak

    def classify_band(torque_rel):
        if not np.isfinite(torque_rel):
            return ("unknown", 0)
        if torque_rel < 0.50:
            return ("lug", 1)
        if torque_rel < 0.65:
            return ("weak", 2)
        if torque_rel < 0.80:
            return ("okay", 3)
        if torque_rel < 0.92:
            return ("strong", 4)
        return ("peak", 5)

    bands = df["torque_rel"].apply(classify_band)
    df["torque_band"] = bands.apply(lambda t: t[0])
    df["torque_band_id"] = bands.apply(lambda t: t[1])
    print(f"[INFO] Loaded global torque curve from {path}")
    return df


def lookup_torque_from_curve(curve_df, rpm_value):
    if curve_df is None:
        return (float("nan"), "unknown", 0)
    if rpm_value is None or not np.isfinite(rpm_value):
        return (float("nan"), "unknown", 0)
    diffs = (curve_df["rpm_bin_center"] - rpm_value).abs()
    idx = diffs.idxmin()
    row = curve_df.loc[idx]
    return (row["torque_rel"], row["torque_band"], int(row["torque_band_id"]))


def derive_log_name(path: Path):
    stem = path.stem.replace("__trans_focus__clean_FULL__", "")
    return stem.split("__")[0]


def process_shift_events(path: Path, shift_table: pd.DataFrame, torque_curve):
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns in {path.name}: {missing}")
        return pd.DataFrame()

    df = df[comfort_mask(df)].copy()
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(
        subset=[
            "time_s",
            "speed_mph",
            "gear_actual__canon",
            "throttle_pct",
            "engine_rpm__canon",
            "tcc_locked_built",
            "tcc_slip_fused",
        ]
    )
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("time_s").reset_index(drop=True)
    df["gear_round"] = df["gear_actual__canon"].round().astype(int)
    events = []
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        if prev["gear_round"] + 1 != curr["gear_round"]:
            continue
        gear_from = int(prev["gear_round"])
        gear_to = int(curr["gear_round"])
        if not (1 <= gear_from <= 5 and gear_to == gear_from + 1):
            continue
        t_shift = curr["time_s"]
        pre = df[(df["time_s"] >= t_shift - 0.5) & (df["time_s"] < t_shift)]
        post = df[(df["time_s"] >= t_shift) & (df["time_s"] <= t_shift + 1.0)]
        if pre.empty or post.empty:
            continue
        rpm_pre_max = pre["engine_rpm__canon"].max()
        rpm_post_min = post["engine_rpm__canon"].min()
        rpm_post_avg = post["engine_rpm__canon"].mean()
        speed_pre_avg = pre["speed_mph"].mean()
        speed_post_avg = post["speed_mph"].mean()
        locked_pre = (pre["tcc_locked_built"] == 1).mean()
        locked_post = (post["tcc_locked_built"] == 1).mean()
        shift_tps = df[
            (df["time_s"] >= t_shift - 0.1) & (df["time_s"] <= t_shift + 0.1)
        ]["throttle_pct"]
        throttle_at_shift = shift_tps.mean() if not shift_tps.empty else curr["throttle_pct"]
        tps_bin = assign_tps_bin(throttle_at_shift)
        rpm_drop = rpm_pre_max - rpm_post_min
        rpm_ratio = rpm_post_avg / RPM_PEAK if RPM_PEAK else float("nan")
        weak_flag = int(
            rpm_post_avg < 0.75 * RPM_PEAK and 20 <= throttle_at_shift <= 70
        )
        torque_rel_pre, torque_band_pre, torque_band_id_pre = lookup_torque_from_curve(
            torque_curve, rpm_pre_max
        )
        torque_rel_post, torque_band_post, torque_band_id_post = lookup_torque_from_curve(
            torque_curve, rpm_post_avg
        )
        score = 0
        tps = throttle_at_shift
        if 20 <= tps <= 70:
            if torque_band_post == "lug":
                score += 3
            elif torque_band_post == "weak":
                score += 2
            elif torque_band_post == "okay" and tps >= 40:
                score += 1
            if torque_band_pre in ("okay", "strong", "peak") and torque_band_post in ("lug", "weak"):
                score += 1
            if locked_pre >= 0.8 and torque_band_post in ("lug", "weak"):
                score += 1
        weak_score_flag = 1 if score >= 2 else 0
        shift_label = f"{gear_from}->{gear_to}"
        events.append(
            {
                "log_name": derive_log_name(path),
                "shift_label": shift_label,
                "time_s_shift": t_shift,
                "speed_pre_avg": speed_pre_avg,
                "speed_post_avg": speed_post_avg,
                "rpm_pre_max": rpm_pre_max,
                "rpm_post_min": rpm_post_min,
                "rpm_post_avg": rpm_post_avg,
                "rpm_drop": rpm_drop,
                "rpm_post_rel_to_peak": rpm_ratio,
                "throttle_pct_at_shift": throttle_at_shift,
                "tps_bin": tps_bin,
                "tcc_locked_pre_frac": locked_pre,
                "tcc_locked_post_frac": locked_post,
                "weak_shift_flag": weak_flag,
                "torque_rel_pre": torque_rel_pre,
                "torque_band_pre": torque_band_pre,
                "torque_band_id_pre": torque_band_id_pre,
                "torque_rel_post": torque_rel_post,
                "torque_band_post": torque_band_post,
                "torque_band_id_post": torque_band_id_post,
                "weak_score": score,
                "weak_shift_flag_score": weak_score_flag,
            }
        )
    return pd.DataFrame(events)


def summarize_shifts(events: pd.DataFrame, shift_table: pd.DataFrame):
    if events.empty:
        return pd.DataFrame()
    rows = []
    for (label, bin_), group in events.groupby(["shift_label", "tps_bin"]):
        if bin_ is None:
            continue
        n_events = len(group)
        n_weak = group["weak_shift_flag_score"].sum()
        weak_frac = n_weak / n_events if n_events else 0.0
        rpm_post_avg = group["rpm_post_avg"].mean()
        rpm_ratio = group["rpm_post_rel_to_peak"].mean()
        locked_post = group["tcc_locked_post_frac"].mean()
        avg_score = group["weak_score"].mean()
        max_score = group["weak_score"].max()
        torque_rel_post_mean = group["torque_rel_post"].mean()
        table_speed = np.nan
        label_clean = label.replace("->", " -> ")
        if not shift_table.empty and label_clean in shift_table["row_label"].values:
            row = shift_table[shift_table["row_label"] == label_clean]
            if not row.empty and bin_ in row.columns:
                table_speed = row.iloc[0][bin_]
        rows.append(
            {
                "shift_label": label,
                "tps_bin": int(bin_),
                "n_events": n_events,
                "n_weak": int(n_weak),
                "weak_frac": weak_frac,
                "rpm_post_avg_mean": rpm_post_avg,
                "rpm_post_rel_to_peak_mean": rpm_ratio,
                "avg_tcc_locked_post_frac": locked_post,
                "avg_weak_score": avg_score,
                "max_weak_score": max_score,
                "torque_rel_post_mean": torque_rel_post_mean,
                "table_shift_speed": table_speed,
            }
        )
    return pd.DataFrame(rows)


def process_cruise_tcc(path: Path, torque_curve):
    df = pd.read_csv(path, low_memory=False)
    cf = df[comfort_mask(df)].copy()
    if cf.empty:
        return pd.DataFrame()
    cf = cf.dropna(
        subset=[
            "speed_mph",
            "gear_actual__canon",
            "throttle_pct",
            "tcc_locked_built",
            "tcc_slip_fused",
            "engine_rpm__canon",
        ]
    )
    cf = cf[
        (cf["speed_mph"] >= 55)
        & (cf["gear_actual__canon"].isin(GEARS_CRUISE))
        & (cf["throttle_pct"].between(5, 35))
    ]
    if cf.empty:
        return pd.DataFrame()
    cf["gear"] = cf["gear_actual__canon"].round().astype(int)
    cf["tps_bin"] = cf["throttle_pct"].apply(assign_tps_bin)
    cf["band"] = cf["speed_mph"].apply(find_speed_band)
    cf = cf.dropna(subset=["band"])
    if cf.empty:
        return pd.DataFrame()
    cf["band_lo"] = cf["band"].apply(lambda b: b[0])
    cf["band_hi"] = cf["band"].apply(lambda b: b[1])
    records = []
    for (gear, lo, hi, bin_), group in cf.groupby(["gear", "band_lo", "band_hi", "tps_bin"]):
        n_rows = len(group)
        if n_rows == 0:
            continue
        locked_frac = (group["tcc_locked_built"] == 1).mean()
        rpm_mean = group["engine_rpm__canon"].mean()
        torque_rel_cruise, torque_band_cruise, torque_band_id_cruise = lookup_torque_from_curve(
            torque_curve, rpm_mean
        )
        mpg_flag = 0
        if lo >= 60 and bin_ <= 25 and locked_frac < 0.7:
            if torque_band_cruise in ("okay", "strong", "peak"):
                mpg_flag = 1
        comfort_flag = 0
        if bin_ >= 31 and locked_frac >= 0.8 and torque_band_cruise in ("lug", "weak"):
            comfort_flag = 1
        records.append(
            {
                "log_name": derive_log_name(path),
                "gear": int(gear),
                "speed_band_lo": float(lo),
                "speed_band_hi": float(hi),
                "tps_bin": int(bin_),
                "n_rows": n_rows,
                "tcc_locked_frac": locked_frac,
                "tcc_mean_slip": group["tcc_slip_fused"].mean(),
                "rpm_mean": rpm_mean,
                "throttle_mean": group["throttle_pct"].mean(),
                "mpg_weakspot_flag": mpg_flag,
                "comfort_issue_flag": comfort_flag,
                "torque_rel_cruise": torque_rel_cruise,
                "torque_band_cruise": torque_band_cruise,
                "torque_band_id_cruise": torque_band_id_cruise,
            }
        )
    return pd.DataFrame(records)


def write_summary(events_df, shift_summary_df, cruise_df, logs_count):
    total_events = len(events_df)
    score_events = (
        int(events_df["weak_shift_flag_score"].sum()) if "weak_shift_flag_score" in events_df else 0
    )
    lines = [
        f"[INFO] Shift events: {total_events}",
        f"[INFO] Score-based weak events: {score_events}",
        f"[INFO] Logs processed: {logs_count}",
        "",
    ]
    shift_bad = (
        shift_summary_df[shift_summary_df["weak_frac"] > 0.3]
        if not shift_summary_df.empty
        else pd.DataFrame()
    )
    lines.append("Shifts with weak_frac > 0.3:")
    if shift_bad.empty:
        lines.append("  None.")
    else:
        for _, row in shift_bad.iterrows():
            lines.append(
                f"  {row['shift_label']} @ TPS {int(row['tps_bin'])}: weak_frac={row['weak_frac']:.2f}"
            )
    lines.append("")
    cruise_bad = (
        cruise_df[cruise_df["mpg_weakspot_flag"] == 1]
        if not cruise_df.empty
        else pd.DataFrame()
    )
    lines.append("Cruise TCC weak spots:")
    if cruise_bad.empty:
        lines.append("  None.")
    else:
        for _, row in cruise_bad.iterrows():
            lines.append(
                f"  gear={row['gear']}, band={int(row['speed_band_lo'])}-{int(row['speed_band_hi'])}, "
                f"TPS={row['tps_bin']}, locked_frac={row['tcc_locked_frac']:.2f}"
            )
    lines.append("")
    lines.append("[TORQUE-BASED SHIFT WEAKSPOTS]")
    high_score = (
        shift_summary_df[shift_summary_df["avg_weak_score"] > 1.5]
        if not shift_summary_df.empty
        else pd.DataFrame()
    )
    if high_score.empty:
        lines.append("  None.")
    else:
        for _, row in high_score.iterrows():
            lines.append(
                f"  {row['shift_label']} @ TPS {int(row['tps_bin'])}: avg_score={row['avg_weak_score']:.2f}, "
                f"torque_rel={row['torque_rel_post_mean']:.2f}"
            )
    lines.append("")
    lines.append("[TORQUE-BASED CRUISE HOTSPOTS]")
    cruise_hotspots = cruise_df[
        (cruise_df["mpg_weakspot_flag"] == 1) | (cruise_df["comfort_issue_flag"] == 1)
    ] if not cruise_df.empty else pd.DataFrame()
    if cruise_hotspots.empty:
        lines.append("  None.")
    else:
        for _, row in cruise_hotspots.iterrows():
            lines.append(
                f"  log={row['log_name']}, gear={row['gear']}, band={int(row['speed_band_lo'])}-{int(row['speed_band_hi'])}, "
                f"TPS={row['tps_bin']}, locked_frac={row['tcc_locked_frac']:.2f}, "
                f"torque_band={row['torque_band_cruise']}, torque_rel={row['torque_rel_cruise']:.2f}"
            )
    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_dirs()
    shift_table = load_shift_table()
    torque_curve = load_global_torque_curve()
    files = sorted(FULL_DIR.glob("__trans_focus__clean_FULL__*.csv"))
    shift_events = []
    shift_stats = []
    cruise_stats = []
    for path in files:
        events = process_shift_events(path, shift_table, torque_curve)
        if not events.empty:
            shift_events.append(events)
        summary = summarize_shifts(events, shift_table)
        if not summary.empty:
            shift_stats.append(summary)
        cruise = process_cruise_tcc(path, torque_curve)
        if not cruise.empty:
            cruise_stats.append(cruise)
    all_events = pd.concat(shift_events, ignore_index=True) if shift_events else pd.DataFrame()
    shift_summary = (
        pd.concat(shift_stats, ignore_index=True) if shift_stats else pd.DataFrame()
    )
    cruise_df = pd.concat(cruise_stats, ignore_index=True) if cruise_stats else pd.DataFrame()
    if not all_events.empty:
        all_events.to_csv(EVENT_CSV, index=False)
    if not shift_summary.empty:
        shift_summary.to_csv(SHIFT_SUMMARY, sep="\t", index=False)
    if not cruise_df.empty:
        cruise_df.to_csv(CRUISE_TCC, sep="\t", index=False)
    write_summary(all_events, shift_summary, cruise_df, len(files))
    print(f"[INFO] Logs processed: {len(files)}")
    print(f"[INFO] Shift events file: {EVENT_CSV}")
    print(f"[INFO] Shift summary file: {SHIFT_SUMMARY}")
    print(f"[INFO] Cruise TCC file: {CRUISE_TCC}")
    print(f"[INFO] Text summary: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
