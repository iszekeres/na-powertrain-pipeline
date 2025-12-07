import math
from pathlib import Path

import numpy as np
import pandas as pd

FULL_DIR = Path("newlogs") / "cleaned"
OUT_DIR = Path("newlogs") / "output" / "02_passes" / "DIAG_HIGHWAY"
PER_LOG_PATH = OUT_DIR / "HIGHWAY_CRUISE_PROFILE__PER_LOG.tsv"
AGG_PATH = OUT_DIR / "HIGHWAY_CRUISE_PROFILE__AGG.tsv"
SUMMARY_PATH = OUT_DIR / "SUMMARY.txt"

REQUIRED_COLS = [
    "time_s",
    "speed_mph",
    "gear_actual__canon",
    "throttle_pct",
    "pedal_pct",
    "engine_rpm__canon",
    "tcc_locked_built",
    "tcc_slip_fused",
    "shift_mode_canon",
    "mode_profile",
]

TPS_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]
SPEED_BANDS = [
    (40, 50),
    (50, 60),
    (60, 70),
    (70, 80),
    (80, 90),
]
GEARS = [3, 4, 5, 6]


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_full_logs():
    return sorted(FULL_DIR.glob("__trans_focus__clean_FULL__*.csv"))


def derive_log_name(path: Path):
    stem = path.stem
    prefix = "__trans_focus__clean_FULL__"
    if stem.startswith(prefix):
        remainder = stem[len(prefix) :]
        parts = remainder.split("__")
        if parts:
            return parts[0]
    return stem


def assign_tps_bin(value):
    try:
        pct = float(value)
    except (ValueError, TypeError):
        return None
    pct = max(0.0, pct)
    for b in TPS_AXIS:
        if pct <= b:
            return b
    return 100


def find_speed_band(speed):
    for lo, hi in SPEED_BANDS:
        if lo < speed <= hi:
            return lo, hi
    return None


def process_log(path: Path):
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] Missing required columns in {path.name}: {', '.join(missing)}")
        return None

    mask = (df["mode_profile"] == "comfort") | (df["shift_mode_canon"] == "normal")
    dfc = df.loc[mask].copy()
    if dfc.empty:
        print(f"[WARN] No comfort rows in {path.name}")
        return None

    subset = [
        "speed_mph",
        "gear_actual__canon",
        "throttle_pct",
        "pedal_pct",
        "engine_rpm__canon",
        "tcc_locked_built",
        "tcc_slip_fused",
    ]
    dfc = dfc.dropna(subset=subset)
    if dfc.empty:
        print(f"[WARN] No complete comfort rows in {path.name}")
        return None

    dfc["speed_mph"] = dfc["speed_mph"].astype(float)
    dfc["gear_actual__canon"] = dfc["gear_actual__canon"].astype(float)
    dfc["throttle_pct"] = dfc["throttle_pct"].astype(float)
    dfc["pedal_pct"] = dfc["pedal_pct"].astype(float)
    dfc["engine_rpm__canon"] = dfc["engine_rpm__canon"].astype(float)
    dfc["tcc_locked_built"] = dfc["tcc_locked_built"].astype(float)
    dfc["tcc_slip_fused"] = dfc["tcc_slip_fused"].astype(float)

    dfc = dfc[
        (dfc["speed_mph"] > 40)
        & (dfc["gear_actual__canon"].between(min(GEARS), max(GEARS)))
    ]
    if dfc.empty:
        print(f"[WARN] No comfort cruise rows >40 mph in {path.name}")
        return None

    dfc["gear"] = dfc["gear_actual__canon"].round().astype(int)
    dfc["tps_bin"] = dfc["throttle_pct"].apply(assign_tps_bin)

    bands = dfc["speed_mph"].apply(find_speed_band)
    dfc["band_lo"] = bands.apply(lambda b: b[0] if b else np.nan)
    dfc["band_hi"] = bands.apply(lambda b: b[1] if b else np.nan)
    dfc = dfc.dropna(subset=["band_lo", "band_hi"])
    if dfc.empty:
        print(f"[WARN] No rows fell into defined speed bands for {path.name}")
        return None

    dfc["log_name"] = derive_log_name(path)

    dfc["tcc_locked_flag"] = (dfc["tcc_locked_built"] == 1).astype(int)

    groups = []
    for name, group in dfc.groupby(["log_name", "gear", "band_lo", "band_hi", "tps_bin"]):
        log_name, gear, band_lo, band_hi, tps_bin = name
        n_rows = len(group)
        if n_rows == 0:
            continue
        locked_frac = group["tcc_locked_flag"].mean()
        tcc_slip = group["tcc_slip_fused"]
        mpg_lean = int(band_lo >= 60 and tps_bin <= 25 and locked_frac >= 0.8)
        mpg_weak = int(band_lo >= 60 and tps_bin <= 25 and locked_frac < 0.7)
        groups.append(
            {
                "log_name": log_name,
                "gear": int(gear),
                "band_lo": float(band_lo),
                "band_hi": float(band_hi),
                "tps_bin": int(tps_bin),
                "n_rows": n_rows,
                "speed_mean": group["speed_mph"].mean(),
                "speed_min": group["speed_mph"].min(),
                "speed_max": group["speed_mph"].max(),
                "pedal_mean": group["pedal_pct"].mean(),
                "pedal_min": group["pedal_pct"].min(),
                "pedal_max": group["pedal_pct"].max(),
                "throttle_mean": group["throttle_pct"].mean(),
                "throttle_min": group["throttle_pct"].min(),
                "throttle_max": group["throttle_pct"].max(),
                "rpm_mean": group["engine_rpm__canon"].mean(),
                "rpm_min": group["engine_rpm__canon"].min(),
                "rpm_max": group["engine_rpm__canon"].max(),
                "tcc_locked_frac": locked_frac,
                "tcc_slip_mean": tcc_slip.mean(),
                "tcc_slip_min": tcc_slip.min(),
                "tcc_slip_max": tcc_slip.max(),
                "mpg_lean_flag": mpg_lean,
                "mpg_weakspot_flag": mpg_weak,
            }
        )
    if not groups:
        print(f"[WARN] No grouped cruise data for {path.name}")
        return None
    return pd.DataFrame(groups)


def make_aggregate(per_log):
    if per_log.empty:
        return per_log.iloc[:0]
    agg_list = []
    grp = per_log.groupby(["gear", "band_lo", "band_hi", "tps_bin"])
    for name, group in grp:
        gear, band_lo, band_hi, tps_bin = name
        total_rows = group["n_rows"].sum()
        if total_rows == 0:
            continue
        weight = group["n_rows"]
        def weighted_mean(field):
            return (group[field] * weight).sum() / total_rows
        log_count = group["log_name"].nunique()
        agg_list.append(
            {
                "gear": gear,
                "band_lo": float(band_lo),
                "band_hi": float(band_hi),
                "tps_bin": int(tps_bin),
                "total_rows": int(total_rows),
                "log_count": int(log_count),
                "speed_mean": weighted_mean("speed_mean"),
                "pedal_mean": weighted_mean("pedal_mean"),
                "throttle_mean": weighted_mean("throttle_mean"),
                "rpm_mean": weighted_mean("rpm_mean"),
                "tcc_locked_frac_mean": weighted_mean("tcc_locked_frac"),
                "tcc_slip_mean_mean": weighted_mean("tcc_slip_mean"),
                "mpg_lean_flag_any": int(group["mpg_lean_flag"].any()),
                "mpg_weakspot_flag_any": int(group["mpg_weakspot_flag"].any()),
            }
        )
    return pd.DataFrame(agg_list)


def write_summary(total_logs, total_rows, agg_summary):
    lines = [
        f"[INFO] Logs processed: {total_logs}",
        f"[INFO] Total comfort cruise rows: {total_rows}",
        "",
    ]
    weak = agg_summary[agg_summary["mpg_weakspot_flag_any"] == 1]
    lean = agg_summary[agg_summary["mpg_lean_flag_any"] == 1]

    lines.append("Potential MPG weak spots (comfort cruise):")
    if weak.empty:
        lines.append("  None detected.")
    else:
        for _, row in weak.iterrows():
            lines.append(
                f"  gear={int(row['gear'])}, band={int(row['band_lo'])}-{int(row['band_hi'])}, "
                f"tps_bin={int(row['tps_bin'])}: "
                f"tcc_locked_frac_mean={row['tcc_locked_frac_mean']:.2f}, "
                f"rpm_mean={row['rpm_mean']:.1f}, throttle_mean={row['throttle_mean']:.1f}"
            )

    lines.append("")
    lines.append("MPG-friendly bands:")
    if lean.empty:
        lines.append("  None detected.")
    else:
        for _, row in lean.iterrows():
            lines.append(
                f"  gear={int(row['gear'])}, band={int(row['band_lo'])}-{int(row['band_hi'])}, "
                f"tps_bin={int(row['tps_bin'])}: "
                f"tcc_locked_frac_mean={row['tcc_locked_frac_mean']:.2f}, "
                f"rpm_mean={row['rpm_mean']:.1f}, throttle_mean={row['throttle_mean']:.1f}"
            )

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_dirs()
    files = find_full_logs()
    total_rows = 0
    summaries = []
    processed = 0
    for path in files:
        df = process_log(path)
        if df is None:
            continue
        processed += 1
        total_rows += df["n_rows"].sum()
        summaries.append(df)
    if not summaries:
        print("[WARN] No comfort cruise profiles generated.")
        return
    per_log_df = pd.concat(summaries, ignore_index=True)
    per_log_df.to_csv(PER_LOG_PATH, sep="\t", index=False)
    agg_df = make_aggregate(per_log_df)
    agg_df.to_csv(AGG_PATH, sep="\t", index=False)
    write_summary(processed, total_rows, agg_df)
    print(f"[INFO] Logs processed: {processed}")
    print(f"[INFO] Per-log summary: {PER_LOG_PATH}")
    print(f"[INFO] Aggregate summary: {AGG_PATH}")
    print(f"[INFO] Text summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
