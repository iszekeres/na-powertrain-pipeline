import glob
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

TORQUE_OUT_DIR = Path("newlogs") / "output" / "02_passes" / "TORQUE"
RPM_BIN_WIDTH = 100
RPM_MIN = 1500
RPM_MAX = 6800
TORQUE_PEAK_APPROX_RPM = 4350.0


def ensure_output_dir():
    os.makedirs(TORQUE_OUT_DIR, exist_ok=True)


def pick_torque_column(df, log_name):
    best_col = None
    best_count = -1
    for col in ("engine_torque_ecm", "engine_torque", "engine_torque_trans"):
        if col not in df.columns:
            continue
        nonna = int(df[col].notna().sum())
        if nonna > best_count:
            best_count = nonna
            best_col = col
    if best_col:
        print(f"[INFO] {log_name} torque column -> {best_col}")
        return best_col
    print(f"[WARN] {log_name} missing torque columns")
    return None


def classify_mode(row):
    state = str(row.get("shift_mode_state_canon", "")).lower()
    if "pattern_a" in state:
        return "pattern_a"
    if "normal" in state:
        return "normal"
    mode = str(row.get("shift_mode_canon", "")).lower()
    if "pattern_a" in mode:
        return "pattern_a"
    if "normal" in mode:
        return "normal"
    return "unknown"


def classify_lock_state(row):
    return "locked" if row.get("tcc_locked_built", 0) == 1 else "unlocked"


def compute_rpm_bin(rpm):
    try:
        rpm = float(rpm)
    except (TypeError, ValueError):
        return np.nan
    if math.isnan(rpm) or rpm < RPM_MIN or rpm > RPM_MAX:
        return np.nan
    idx = math.floor((rpm - RPM_MIN) / RPM_BIN_WIDTH)
    center = RPM_MIN + (idx + 0.5) * RPM_BIN_WIDTH
    return center


def derive_log_name(path):
    base = os.path.basename(path)
    clean = base.replace("__trans_focus__clean_FULL__", "")
    if "__" in clean:
        return clean.split("__")[0]
    return os.path.splitext(clean)[0]


def process_log(path):
    df = pd.read_csv(path, low_memory=False)
    log_name = derive_log_name(path)
    required = {"engine_rpm__canon", "gear_actual__canon", "throttle_pct", "tcc_locked_built", "time_s"}
    if not required.issubset(df.columns):
        print(f"[WARN] {log_name} missing required columns")
        return None
    torque_col = pick_torque_column(df, log_name)
    if torque_col is None:
        return None

    if "throttle_pct" in df.columns:
        hi_load_mask = df["throttle_pct"] >= 40
    else:
        hi_load_mask = pd.Series(True, index=df.index)

    if "gear_actual__canon" in df.columns:
        gear_mask = df["gear_actual__canon"].between(1, 6, inclusive="both")
    else:
        gear_mask = pd.Series(True, index=df.index)

    if "brake" in df.columns:
        brake_mask = df["brake"] == 0
    else:
        brake_mask = pd.Series(True, index=df.index)

    final_mask = hi_load_mask & gear_mask & brake_mask
    n_after_final = int(final_mask.sum())
    print(f"[DEBUG] {log_name}: rows after hi-load/gear/brake mask = {n_after_final}")

    if n_after_final == 0:
        print(f"[WARN] {log_name}: no rows after main hi-load/gear/brake mask, applying FALLBACK.")
        if "throttle_pct" in df.columns:
            fallback_mask = df["throttle_pct"] >= 30
        else:
            fallback_mask = pd.Series(True, index=df.index)
        if "gear_actual__canon" in df.columns:
            fallback_mask &= df["gear_actual__canon"].between(1, 6, inclusive="both")
        if "brake" in df.columns:
            fallback_mask &= df["brake"] == 0
        final_mask = fallback_mask
        n_after_final = int(final_mask.sum())
        print(f"[DEBUG] {log_name}: rows after FALLBACK hi-load/gear/brake mask = {n_after_final}")

    dfw = df.loc[final_mask].copy()

    dfw = dfw[dfw["engine_rpm__canon"].notna()].copy()
    dfw = dfw[dfw[torque_col].notna()].copy()
    n_after_rpm_torque = int(dfw.shape[0])
    print(f"[DEBUG] {log_name}: rows after dropping NaN rpm/torque = {n_after_rpm_torque}")

    if dfw.empty:
        print(f"[WARN] {log_name}: no usable samples after rpm/torque/binning; skipping this log.")
        return None

    dfw["log_name"] = log_name
    dfw["mode_group"] = dfw.apply(classify_mode, axis=1)
    dfw["lock_state"] = dfw.apply(classify_lock_state, axis=1)
    dfw["gear"] = dfw["gear_actual__canon"].round().astype(int)
    dfw["rpm"] = dfw["engine_rpm__canon"]
    dfw["torque"] = dfw[torque_col]

    dfw["rpm_bin_center"] = dfw["rpm"].apply(compute_rpm_bin)
    dfw = dfw[dfw["rpm_bin_center"].notna()].copy()
    n_after_bin = int(dfw.shape[0])
    print(f"[DEBUG] {log_name}: rows after rpm binning = {n_after_bin}")

    if dfw.empty:
        print(f"[WARN] {log_name}: no usable samples after rpm/torque/binning; skipping this log.")
        return None

    return dfw[["log_name", "mode_group", "lock_state", "gear", "rpm", "torque", "rpm_bin_center"]]


def aggregate(df, groups):
    return (
        df.groupby(groups)["torque"]
        .agg(n_samples="count", torque_p90=lambda s: s.quantile(0.9), torque_mean="mean")
        .reset_index()
    )


def normalize(df, group_keys):
    peaks = df.groupby(group_keys)["torque_p90"].transform("max")
    df["torque_rel"] = df["torque_p90"] / peaks
    return df


def summarize_peaks(df, keys):
    rows = []
    grouped = df.groupby(keys)
    for name, group in grouped:
        idx = group["torque_p90"].idxmax()
        peak = group.loc[idx]
        rows.append((*name, peak["rpm_bin_center"], peak["torque_p90"]))
    return rows


def main():
    ensure_output_dir()
    files = glob.glob(os.path.join("newlogs", "cleaned", "__trans_focus__clean_FULL__*.csv"))
    if not files:
        print("[INFO] No FULL logs found.")
        return
    all_dfs = []
    for path in files:
        dfw = process_log(path)
        if dfw is not None:
            all_dfs.append(dfw)
    if not all_dfs:
        print("[WARN] No WOT samples collected.")
        return
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_global = aggregate(df_all, ["rpm_bin_center"])
    df_global = normalize(df_global, ["rpm_bin_center"])
    df_mode_lock = aggregate(df_all, ["mode_group", "lock_state", "rpm_bin_center"])
    df_mode_lock = normalize(df_mode_lock, ["mode_group", "lock_state"])
    df_mode_lock_gear = aggregate(
        df_all, ["mode_group", "lock_state", "gear", "rpm_bin_center"]
    )
    df_mode_lock_gear = normalize(df_mode_lock_gear, ["mode_group", "lock_state", "gear"])
    df_global.sort_values("rpm_bin_center", inplace=True)
    df_mode_lock.sort_values(["mode_group", "lock_state", "rpm_bin_center"], inplace=True)
    df_mode_lock_gear.sort_values(["mode_group", "lock_state", "gear", "rpm_bin_center"], inplace=True)
    df_global.to_csv(TORQUE_OUT_DIR / "TORQUE_CURVE__GLOBAL_WOT.tsv", sep="\t", index=False)
    df_mode_lock.to_csv(TORQUE_OUT_DIR / "TORQUE_CURVES__MODE_LOCK_WOT.tsv", sep="\t", index=False)
    df_mode_lock_gear.to_csv(TORQUE_OUT_DIR / "TORQUE_CURVES__MODE_LOCK_GEAR_WOT.tsv", sep="\t", index=False)
    global_peak = df_global.loc[df_global["torque_p90"].idxmax()]
    mode_lock_peaks = summarize_peaks(df_mode_lock, ["mode_group", "lock_state"])
    mode_lock_gear_peaks = summarize_peaks(df_mode_lock_gear, ["mode_group", "lock_state", "gear"])
    summary_lines = [
        f"[INFO] FULL logs processed: {len(files)}",
        f"[INFO] Total WOT samples: {len(df_all)}",
        "",
        "[GLOBAL]",
        f"  peak_rpm = {global_peak['rpm_bin_center']:.0f}, torque_p90 = {global_peak['torque_p90']:.1f}",
        "",
        "[BY MODE+LOCK]",
    ]
    for mode_group, lock_state, rpm_peak, torque_peak in mode_lock_peaks:
        summary_lines.append(
            f"  mode={mode_group}, lock={lock_state}: peak_rpm={rpm_peak:.0f}, torque_p90={torque_peak:.1f}"
        )
    summary_lines.append("")
    summary_lines.append("[BY MODE+LOCK+GEAR]")
    for mode_group, lock_state, gear, rpm_peak, torque_peak in mode_lock_gear_peaks:
        summary_lines.append(
            f"  mode={mode_group}, lock={lock_state}, gear={gear}: peak_rpm={rpm_peak:.0f}, torque_p90={torque_peak:.1f}"
        )
    (TORQUE_OUT_DIR / "SUMMARY__TORQUE_CURVES_WOT.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    print(f"[INFO] Processed {len(files)} FULL logs")
    print(f"[INFO] Global curve: {TORQUE_OUT_DIR / 'TORQUE_CURVE__GLOBAL_WOT.tsv'}")
    print(f"[INFO] Mode+lock curves: {TORQUE_OUT_DIR / 'TORQUE_CURVES__MODE_LOCK_WOT.tsv'}")
    print(f"[INFO] Mode+lock+gear curves: {TORQUE_OUT_DIR / 'TORQUE_CURVES__MODE_LOCK_GEAR_WOT.tsv'}")
    print(f"[INFO] Summary: {TORQUE_OUT_DIR / 'SUMMARY__TORQUE_CURVES_WOT.txt'}")


if __name__ == "__main__":
    main()
