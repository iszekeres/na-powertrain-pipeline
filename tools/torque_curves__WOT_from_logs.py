import glob
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

TORQUE_OUT_DIR = Path("newlogs") / "output" / "02_passes" / "TORQUE"
RPM_BIN_WIDTH = 100
RPM_MIN = 1500
RPM_MAX = 6800


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


def classify_mode_label(row):
    mode_profile = str(row.get("mode_profile", "")).lower()
    shift_mode = str(row.get("shift_mode_canon", "")).lower()
    if mode_profile == "comfort" or shift_mode == "normal":
        return "normal"
    if "pattern_a" in shift_mode:
        return "pattern_a"
    return "unknown"


def classify_lock_state(row):
    slip = row.get("tcc_slip_fused", np.nan)
    locked = row.get("tcc_locked_built", np.nan)
    if pd.isna(slip) or pd.isna(locked):
        return "unknown"
    if (locked == 1) and (abs(slip) <= 50):
        return "locked"
    if abs(slip) > 1500:
        return "bad_slip"
    return "unlocked"


def compute_rpm_bin(rpm):
    try:
        rpm = float(rpm)
    except (TypeError, ValueError):
        return np.nan
    if math.isnan(rpm) or rpm < RPM_MIN or rpm > RPM_MAX:
        return np.nan
    idx = math.floor((rpm - RPM_MIN) / RPM_BIN_WIDTH)
    return RPM_MIN + (idx + 0.5) * RPM_BIN_WIDTH


def derive_log_name(path):
    base = os.path.basename(path)
    clean = base.replace("__trans_focus__clean_FULL__", "")
    if "__" in clean:
        return clean.split("__")[0]
    return os.path.splitext(clean)[0]


def build_masks(df):
    gear_mask = df["gear_actual__canon"].between(1, 6, inclusive="both")
    brake_col = df["brake"] if "brake" in df.columns else pd.Series(0, index=df.index)
    brake_mask = brake_col == 0
    mode_is_wot = df.get("mode_is_wot", pd.Series(0, index=df.index)).fillna(0)
    pedal = df.get("pedal_pct", pd.Series(np.nan, index=df.index)).fillna(-np.inf)
    thr = df.get("throttle_pct", pd.Series(np.nan, index=df.index)).fillna(-np.inf)

    mask1 = gear_mask & brake_mask & ((mode_is_wot == 1) | (pedal >= 90))
    mask2 = gear_mask & brake_mask & pedal.between(60, 90, inclusive="left") & thr.between(60, 90, inclusive="left")
    mask3 = gear_mask & brake_mask & pedal.between(35, 60, inclusive="left") & thr.between(35, 60, inclusive="left")
    return mask1, mask2, mask3


def apply_tier_weights(df, mask1, mask2, mask3):
    tier = pd.Series(0, index=df.index)
    weight = pd.Series(0.0, index=df.index)
    tier.loc[mask3] = 3
    weight.loc[mask3] = 0.3
    tier.loc[mask2] = 2
    weight.loc[mask2] = 0.6
    tier.loc[mask1] = 1
    weight.loc[mask1] = 1.0
    df = df.copy()
    df["tier"] = tier
    df["tier_weight"] = weight
    return df


def process_log(path):
    df = pd.read_csv(path, low_memory=False)
    log_name = derive_log_name(path)
    required = {
        "engine_rpm__canon",
        "gear_actual__canon",
        "throttle_pct",
        "pedal_pct",
        "mode_is_wot",
        "brake",
        "tcc_slip_fused",
        "tcc_locked_built",
        "time_s",
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] {log_name} missing required columns: {missing}")
        return None
    torque_col = pick_torque_column(df, log_name)
    if torque_col is None:
        return None

    mask1, mask2, mask3 = build_masks(df)
    print(f"[INFO] {log_name}: tier1={int(mask1.sum())}, tier2={int(mask2.sum())}, tier3={int(mask3.sum())}")
    df = apply_tier_weights(df, mask1, mask2, mask3)
    df = df[df["tier"] > 0].copy()
    if df.empty:
        print(f"[WARN] {log_name}: no rows after tier masks")
        return None

    df["lock_state"] = df.apply(classify_lock_state, axis=1)
    df = df[df["lock_state"] != "bad_slip"].copy()
    df["mode_label"] = df.apply(classify_mode_label, axis=1)

    df = df.dropna(subset=["engine_rpm__canon", torque_col])
    if df.empty:
        print(f"[WARN] {log_name}: no usable rows after RPM/torque drop")
        return None

    df["rpm_bin_center"] = df["engine_rpm__canon"].apply(compute_rpm_bin)
    df = df[df["rpm_bin_center"].notna()].copy()
    if df.empty:
        print(f"[WARN] {log_name}: no rows after rpm binning")
        return None

    df = df.loc[
        :,
        [
            "mode_label",
            "lock_state",
            "gear_actual__canon",
            "rpm_bin_center",
            torque_col,
            "tier",
            "tier_weight",
        ],
    ].rename(columns={torque_col: "torque"})
    return df


def apply_mad_filter(group):
    torques = group["torque"].values
    median = np.median(torques)
    diff = np.abs(torques - median)
    mad = np.median(diff)
    if mad > 0:
        keep = diff <= (3.0 * mad)
    else:
        keep = np.ones_like(diff, dtype=bool)
    return group.loc[keep]


def aggregate_weighted(df, group_cols):
    rows = []
    for keys, grp in df.groupby(group_cols):
        grp = apply_mad_filter(grp)
        if grp.empty:
            continue
        weight_sum = grp["tier_weight"].sum()
        if weight_sum <= 0:
            continue
        torque_w = (grp["torque"] * grp["tier_weight"]).sum() / weight_sum
        record = {}
        if isinstance(keys, tuple):
            for col, val in zip(group_cols, keys):
                record[col] = val
        else:
            record[group_cols[0]] = keys
        record["torque_abs_raw"] = torque_w
        record["weighted_samples"] = weight_sum
        rows.append(record)
    return pd.DataFrame(rows)


def smooth_curve(df, group_cols, rpm_col="rpm_bin_center", torque_col="torque_abs_raw"):
    if df.empty:
        df["torque_abs_smoothed"] = df.get(torque_col, np.nan)
        return df
    out = []
    groups = df.groupby(group_cols, dropna=False) if group_cols else [(None, df)]
    for keys, grp in groups:
        grp = grp.sort_values(rpm_col)
        rpm_vals = grp[rpm_col].values
        torque_vals = grp[torque_col].values.astype(float)
        series = pd.Series(torque_vals, index=rpm_vals)
        series = series.interpolate(limit_direction="both")
        vals = series.values
        n = len(vals)
        smoothed = np.array(vals, dtype=float)
        if n == 1:
            smoothed[0] = vals[0]
        elif n == 2:
            smoothed[0] = (2 * vals[0] + vals[1]) / 3
            smoothed[1] = (vals[0] + 2 * vals[1]) / 3
        else:
            for i in range(n):
                if i == 0:
                    smoothed[i] = (2 * vals[i] + vals[i + 1]) / 3
                elif i == n - 1:
                    smoothed[i] = (vals[i - 1] + 2 * vals[i]) / 3
                else:
                    smoothed[i] = (vals[i - 1] + 2 * vals[i] + vals[i + 1]) / 4
        grp = grp.copy()
        grp["torque_abs_smoothed"] = smoothed
        out.append(grp)
    return pd.concat(out, ignore_index=True)


def normalize_global(df):
    if df.empty:
        df["torque_rel"] = np.nan
        return df
    peak = df["torque_abs_smoothed"].max()
    df["torque_rel"] = df["torque_abs_smoothed"] / peak if peak and peak > 0 else np.nan
    return df


def summarize_bins(df_global, tier_counts, df_samples, df_global_smoothed, threshold=10):
    lines = []
    lines.append(
        f"[INFO] Tier samples -> Tier1: {tier_counts.get(1,0)}, Tier2: {tier_counts.get(2,0)}, Tier3: {tier_counts.get(3,0)}"
    )
    if not df_global.empty:
        peak_row = df_global.loc[df_global["torque_abs_smoothed"].idxmax()]
        lines.append(
            f"[INFO] Global peak: rpm={peak_row['rpm_bin_center']:.0f}, torque={peak_row['torque_abs_smoothed']:.1f}"
        )
        rich_bins = (df_global["weighted_samples"] >= threshold).sum()
        lines.append(f"[INFO] Global bins with >= {threshold} weighted samples: {rich_bins}")

    if not df_samples.empty:
        tier_weight_by_rpm = df_samples.groupby("rpm_bin_center")["tier_weight"].sum()
        tier1_weight_by_rpm = df_samples[df_samples["tier"] == 1].groupby("rpm_bin_center")[
            "tier_weight"
        ].sum()
        heavy_bins = []
        for rpm, total_w in tier_weight_by_rpm.items():
            t1 = tier1_weight_by_rpm.get(rpm, 0.0)
            frac = t1 / total_w if total_w > 0 else 0
            if frac < 0.5:
                heavy_bins.append((rpm, frac))
        if heavy_bins:
            lines.append("[INFO] Bins relying on Tier2/3 (tier1 fraction < 0.5):")
            lines.extend([f"  rpm {rpm:.0f}: tier1_frac={frac:.2f}" for rpm, frac in heavy_bins])

    if not df_global_smoothed.empty:
        total_max = df_global_smoothed["torque_abs_smoothed"].max()
        diff_bins = []
        for _, row in df_global_smoothed.iterrows():
            raw = row.get("torque_abs_raw", np.nan)
            sm = row.get("torque_abs_smoothed", np.nan)
            if (
                np.isfinite(raw)
                and np.isfinite(sm)
                and total_max > 0
                and abs(sm - raw) > 0.1 * total_max
            ):
                diff_bins.append(row["rpm_bin_center"])
        if diff_bins:
            lines.append("[INFO] Bins with >10% smoothing adjustment:")
            lines.extend([f"  rpm {rpm:.0f}" for rpm in diff_bins])
    return lines


def main():
    ensure_output_dir()
    files = glob.glob(os.path.join("newlogs", "cleaned", "__trans_focus__clean_FULL__*.csv"))
    if not files:
        print("[ERROR] No FULL logs found.")
        sys.exit(1)

    all_rows = []
    tier_counts = defaultdict(int)
    for path in files:
        dfw = process_log(path)
        if dfw is None:
            continue
        all_rows.append(dfw)
        for t, cnt in dfw["tier"].value_counts().items():
            tier_counts[int(t)] += int(cnt)

    if not all_rows:
        print("[ERROR] No usable torque samples collected.")
        sys.exit(1)

    df_all = pd.concat(all_rows, ignore_index=True)

    # Aggregations
    df_mode_lock_gear = aggregate_weighted(
        df_all, ["mode_label", "lock_state", "gear_actual__canon", "rpm_bin_center"]
    )
    df_mode_lock = aggregate_weighted(df_all, ["mode_label", "lock_state", "rpm_bin_center"])
    df_global = aggregate_weighted(df_all, ["rpm_bin_center"])

    # Smoothing
    df_mode_lock_gear = smooth_curve(
        df_mode_lock_gear, ["mode_label", "lock_state", "gear_actual__canon"]
    )
    df_mode_lock = smooth_curve(df_mode_lock, ["mode_label", "lock_state"])
    df_global = smooth_curve(df_global, [])

    # Normalize global
    df_global = normalize_global(df_global)

    # Sort
    df_global.sort_values("rpm_bin_center", inplace=True)
    df_mode_lock.sort_values(["mode_label", "lock_state", "rpm_bin_center"], inplace=True)
    df_mode_lock_gear.sort_values(
        ["mode_label", "lock_state", "gear_actual__canon", "rpm_bin_center"], inplace=True
    )

    # Write outputs
    df_global_out = df_global[
        ["rpm_bin_center", "torque_abs_smoothed", "torque_rel", "weighted_samples"]
    ].rename(columns={"torque_abs_smoothed": "torque_abs"})
    df_global_out.to_csv(TORQUE_OUT_DIR / "TORQUE_CURVE__GLOBAL_WOT.tsv", sep="\t", index=False)

    df_mode_lock_out = df_mode_lock.rename(
        columns={"gear_actual__canon": "gear", "torque_abs_smoothed": "torque_abs"}
    )
    df_mode_lock_out.to_csv(TORQUE_OUT_DIR / "TORQUE_CURVES__MODE_LOCK_WOT.tsv", sep="\t", index=False)

    df_mode_lock_gear_out = df_mode_lock_gear.rename(
        columns={"gear_actual__canon": "gear", "torque_abs_smoothed": "torque_abs"}
    )
    df_mode_lock_gear_out.to_csv(
        TORQUE_OUT_DIR / "TORQUE_CURVES__MODE_LOCK_GEAR_WOT.tsv", sep="\t", index=False
    )

    # Summary
    summary_lines = [
        f"[INFO] FULL logs processed: {len(files)}",
        f"[INFO] Total samples: {len(df_all)}",
    ]
    summary_lines += summarize_bins(df_global, tier_counts, df_all, df_global)

    if not df_mode_lock.empty:
        summary_lines.append("[INFO] Mode+lock bins with >=10 weighted samples:")
        for (mode, lock), grp in df_mode_lock.groupby(["mode_label", "lock_state"]):
            count = (grp["weighted_samples"] >= 10).sum()
            summary_lines.append(f"  mode={mode}, lock={lock}: bins={count}")
    if not df_mode_lock_gear.empty:
        summary_lines.append("[INFO] Mode+lock+gear bins with >=10 weighted samples:")
        for (mode, lock, gear), grp in df_mode_lock_gear.groupby(
            ["mode_label", "lock_state", "gear_actual__canon"]
        ):
            count = (grp["weighted_samples"] >= 10).sum()
            summary_lines.append(f"  mode={mode}, lock={lock}, gear={gear}: bins={count}")

    (TORQUE_OUT_DIR / "SUMMARY__TORQUE_CURVES_WOT.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print("\n".join(summary_lines))
    print(f"[INFO] Global curve: {TORQUE_OUT_DIR / 'TORQUE_CURVE__GLOBAL_WOT.tsv'}")
    print(f"[INFO] Mode+lock curves: {TORQUE_OUT_DIR / 'TORQUE_CURVES__MODE_LOCK_WOT.tsv'}")
    print(f"[INFO] Mode+lock+gear curves: {TORQUE_OUT_DIR / 'TORQUE_CURVES__MODE_LOCK_GEAR_WOT.tsv'}")
    print(f"[INFO] Summary: {TORQUE_OUT_DIR / 'SUMMARY__TORQUE_CURVES_WOT.txt'}")


if __name__ == "__main__":
    main()
