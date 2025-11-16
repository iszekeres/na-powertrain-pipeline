# na_trans_scan_harsh_shifts.py  (RELAXED GATING)
#
# Scan CLEAN_FULL logs for "harsh" shifts using chassis signals or derived accel.
# Outputs a ranked TSV so we can inspect the worst events.

import glob
import os

import numpy as np
import pandas as pd

CLEAN_DIR = os.path.join("newlogs", "cleaned")
OUT_DEBUG = os.path.join("newlogs", "output", "DEBUG")

# Shift detection / gating (relaxed a bit so we actually see events)
MIN_SPEED = 3.0  # mph – ignore nearly-stopped creeping
MAX_SPEED = 90.0  # mph
MIN_TPS = 2.0  # % – allow light throttle, but ignore pure lift-off
MAX_TPS = 95.0

# Corner gating (more permissive)
CORNER_LAT_G = 0.35  # |lat| above this is considered a real corner
CORNER_YAW_DEG = 30.0  # |yaw| above this is considered a real corner

# Harshness scoring thresholds (used only to flag is_harsh; do not gate output)
HARSH_PEAK_MIN = 0.5   # m/s^2 above baseline accel
HARSH_JERK_MIN = 1.0   # m/s^3 approximate; adjust later as needed

# Time windows around shift edge
PRE_WIN = (-0.30, -0.10)  # baseline accel
FULL_WIN = (-0.30, +0.70)  # window to search for spikes

LONG_ALIASES = [
    "Long Accel",
    "Longitudinal Accel",
    "G Force Longitudinal",
    "Accel Longitudinal",
    "Longitudinal Acceleration",
    "long_accel",
]
LAT_ALIASES = [
    "Lat Accel",
    "Lateral Accel",
    "G Force Lateral",
    "Accel Lateral",
    "Lateral Acceleration",
    "lat_accel",
]
YAW_ALIASES = [
    "Yaw Rate",
    "Yaw Rate (deg/s)",
    "Yaw_Rate",
]


def pick_col(df: pd.DataFrame, names) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def derive_long_accel(df: pd.DataFrame, long_col: str | None):
    """
    Prefer a real longitudinal accel channel if present.
    Fallback: derive accel from speed_mph__canon and time_s.
    """
    if long_col is not None:
        a = pd.to_numeric(df[long_col], errors="coerce")
        return a, f"raw:{long_col}"

    if "speed_mph__canon" not in df.columns or "time_s" not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index), "none"

    v_mph = pd.to_numeric(df["speed_mph__canon"], errors="coerce")
    t = pd.to_numeric(df["time_s"], errors="coerce")

    v = v_mph * 0.44704  # mph -> m/s
    dt = t.diff()
    dv = v.diff()
    with np.errstate(divide="ignore", invalid="ignore"):
        a = dv / dt
    return a, "derived_mps2"


def extract_window(series: pd.Series, t: pd.Series, t0: float, win: tuple[float, float]):
    t_start = t0 + win[0]
    t_end = t0 + win[1]
    m = (t >= t_start) & (t <= t_end)
    return series[m]


def main():
    os.makedirs(OUT_DEBUG, exist_ok=True)

    pattern = os.path.join(CLEAN_DIR, "__trans_focus__clean_FULL__*.csv")
    files = sorted(glob.glob(pattern))
    print(f"[INFO] HARSH scan: found {len(files)} CLEAN_FULL file(s) in {CLEAN_DIR}")
    if not files:
        raise SystemExit("[ERROR] No CLEAN_FULL files found for HARSH scan.")

    events = []

    for path in files:
        base = os.path.basename(path)
        print(f"\n[FILE] {base}")
        try:
            df = pd.read_csv(path)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] Failed to read: {e}")
            continue

        req = ["time_s", "speed_mph__canon", "throttle_pct__canon", "gear_actual__canon", "brake"]
        missing = [c for c in req if c not in df.columns]
        if missing:
            print(f"  [ERROR] Missing required columns: {missing}")
            continue

        time_s = pd.to_numeric(df["time_s"], errors="coerce")
        speed = pd.to_numeric(df["speed_mph__canon"], errors="coerce")
        tps = pd.to_numeric(df["throttle_pct__canon"], errors="coerce")
        gear = pd.to_numeric(df["gear_actual__canon"], errors="coerce").fillna(0).astype(int)
        brake = pd.to_numeric(df["brake"], errors="coerce").fillna(0.0)

        # Optional TCC state
        tcc_locked = None
        for cand in ["tcc_locked_built__canon", "tcc_locked_built"]:
            if cand in df.columns:
                tcc_locked = pd.to_numeric(df[cand], errors="coerce").fillna(0).astype(int)
                break

        long_col = pick_col(df, LONG_ALIASES)
        lat_col = pick_col(df, LAT_ALIASES)
        yaw_col = pick_col(df, YAW_ALIASES)

        long_accel, long_src = derive_long_accel(df, long_col)
        lat_accel = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else None
        yaw_rate = pd.to_numeric(df[yaw_col], errors="coerce") if yaw_col else None

        print(f"  [INFO] long_accel source: {long_src}")
        if lat_col:
            print(f"  [INFO] lat_accel column: {lat_col}")
        if yaw_col:
            print(f"  [INFO] yaw_rate column: {yaw_col}")

        # Gear edges
        gear_prev = gear.shift(1)
        is_edge = (gear != gear_prev) & gear_prev.notna()
        is_edge &= gear_prev.between(1, 6) & gear.between(1, 6)

        idx_edges = df.index[is_edge]
        print(f"  [INFO] total gear edges: {len(idx_edges)}")

        for i in idx_edges:
            g_from = int(gear_prev.loc[i])
            g_to = int(gear.loc[i])
            t0 = float(time_s.loc[i])
            spd0 = float(speed.loc[i]) if pd.notna(speed.loc[i]) else np.nan
            tps0 = float(tps.loc[i]) if pd.notna(tps.loc[i]) else np.nan
            br0 = float(brake.loc[i]) if pd.notna(brake.loc[i]) else np.nan

            # Basic gating
            if not (MIN_SPEED <= spd0 <= MAX_SPEED):
                continue
            if not (MIN_TPS <= tps0 <= MAX_TPS):
                continue
            if br0 > 0.5:
                continue

            # Gate out BIG corners, but allow moderate turns
            if lat_accel is not None:
                lat0 = float(lat_accel.loc[i]) if pd.notna(lat_accel.loc[i]) else 0.0
                if abs(lat0) > CORNER_LAT_G:
                    continue
            if yaw_rate is not None:
                yaw0 = float(yaw_rate.loc[i]) if pd.notna(yaw_rate.loc[i]) else 0.0
                if abs(yaw0) > CORNER_YAW_DEG:
                    continue

            win_full = extract_window(long_accel, time_s, t0, FULL_WIN)
            win_pre = extract_window(long_accel, time_s, t0, PRE_WIN)

            # Always try to score this edge, even if the windows are sparse.
            # If PRE window has no valid samples, baseline becomes NaN.
            win_pre_valid = win_pre.dropna()
            if win_pre_valid.empty:
                baseline = np.nan
            else:
                baseline = float(win_pre_valid.median())

            delta = win_full - baseline
            delta_abs = delta.abs()

            if delta_abs.dropna().empty:
                harsh_peak = np.nan
            else:
                harsh_peak = float(delta_abs.max())
            harsh_side = "pos" if delta.max() >= abs(delta.min()) else "neg"

            win_full_sorted = win_full.dropna()
            if win_full_sorted.shape[0] >= 3:
                t_win = time_s[win_full_sorted.index]
                dv = win_full_sorted.diff()
                dt = t_win.diff()
                with np.errstate(divide="ignore", invalid="ignore"):
                    jerk = (dv / dt).abs()
                harsh_jerk = float(jerk.max(skipna=True))
            else:
                harsh_jerk = np.nan

            # Harshness flag (do not gate on this; just record)
            if np.isnan(harsh_peak) or np.isnan(harsh_jerk):
                is_harsh = False
            else:
                is_harsh = (abs(harsh_peak) >= HARSH_PEAK_MIN) and (
                    abs(harsh_jerk) >= HARSH_JERK_MIN
                )

            tcc_lock_state = (
                int(tcc_locked.loc[i])
                if tcc_locked is not None and pd.notna(tcc_locked.loc[i])
                else None
            )

            events.append(
                {
                    "file": base,
                    "time_s": t0,
                    "from_gear": g_from,
                    "to_gear": g_to,
                    "speed_mph": spd0,
                    "throttle_pct": tps0,
                    "brake": br0,
                    "tcc_locked": tcc_lock_state,
                    "long_accel_baseline": baseline,
                    "harsh_peak_delta": harsh_peak,
                    "harsh_side": harsh_side,
                    "harsh_jerk": harsh_jerk,
                    "is_harsh": is_harsh,
                }
            )

    if not events:
        print("\n[WARN] No candidate harsh shifts found with relaxed gating.")
        ev_df = pd.DataFrame(
            columns=[
                "file",
                "time_s",
                "from_gear",
                "to_gear",
                "speed_mph",
                "throttle_pct",
                "brake",
                "tcc_locked",
                "long_accel_baseline",
                "harsh_peak_delta",
                "harsh_side",
                "harsh_jerk",
                "is_harsh",
            ]
        )
    else:
        ev_df = pd.DataFrame(events)
        ev_df = ev_df.sort_values(
            ["harsh_peak_delta", "harsh_jerk"], ascending=[False, False]
        ).reset_index(drop=True)

    # Always write full HARSH_SHIFT_EVENTS.tsv (may be empty)
    out_path = os.path.join(OUT_DEBUG, "HARSH_SHIFT_EVENTS.tsv")
    ev_df.to_csv(out_path, sep="\t", index=False)
    print(f"\n[OK] Wrote HARSH_SHIFT_EVENTS.tsv with {len(ev_df)} events -> {out_path}")

    # Always write TOPN debug file
    out_topn = os.path.join(OUT_DEBUG, "HARSH_SHIFT_EVENTS__TOPN.tsv")
    if ev_df.empty:
        ev_df.to_csv(out_topn, sep="\t", index=False)
        print(f"[OK] Wrote empty HARSH_SHIFT_EVENTS__TOPN.tsv -> {out_topn}")
    else:
        topn = 25
        top_df = ev_df.head(topn)
        top_df.to_csv(out_topn, sep="\t", index=False)
        print(f"[OK] Wrote HARSH_SHIFT_EVENTS__TOPN.tsv (top {topn}) -> {out_topn}")

        print("\n[TOP 10 HARSH EVENTS]")
        with pd.option_context("display.max_columns", 20, "display.width", 160):
            print(ev_df.head(10))


if __name__ == "__main__":
    main()
