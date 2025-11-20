import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TPS_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]


def nearest_tps_bin(tps_value: float) -> int:
    """Return the TPS axis value closest to tps_value."""
    if pd.isna(tps_value):
        return TPS_AXIS[0]
    arr = np.asarray(TPS_AXIS, dtype=float)
    idx = np.abs(arr - float(tps_value)).argmin()
    return int(arr[idx])


def pick_first_existing(df: pd.DataFrame, candidates):
    """Return a Series corresponding to the first existing column with any non-null values, else None."""
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return s
    return None


def compute_event_metrics(df: pd.DataFrame, idx: int, gear_prev, gear_curr) -> dict:
    """
    Compute per-shift metrics for a single upshift event at row idx.

    Uses:
        time_s, speed_mph, engine_rpm__canon, torque series, tcc_slip_fused,
        Lateral Acceleration, Yaw Rate (if present).
    """
    time = pd.to_numeric(df["time_s"], errors="coerce").to_numpy()
    speed = pd.to_numeric(df["speed_mph"], errors="coerce").to_numpy()

    t0 = time[idx]

    # approximate duration: time difference between this sample and previous
    if idx > 0 and not np.isnan(time[idx - 1]):
        duration = max(time[idx] - time[idx - 1], 0.0)
    else:
        duration = np.nan

    # time windows
    pre_window = 0.40
    post_window = 0.60
    pre_mask = (time >= t0 - pre_window) & (time <= t0 - 0.05)
    win_mask = (time >= t0 - pre_window) & (time <= t0 + post_window)

    # jerk from speed
    max_abs_jerk = np.nan
    idx_speed = np.where(win_mask & ~np.isnan(speed))[0]
    if idx_speed.size >= 3:
        t_win = time[idx_speed]
        s_win = speed[idx_speed]
        dt = np.diff(t_win)
        dv = np.diff(s_win)
        valid = dt > 1e-3
        if valid.any():
            a = dv[valid] / dt[valid]
            t_a = t_win[:-1][valid]
            if a.size >= 2:
                da = np.diff(a)
                dt2 = np.diff(t_a)
                valid2 = dt2 > 1e-3
                if valid2.any():
                    jerk = da[valid2] / dt2[valid2]
                    if jerk.size > 0:
                        max_abs_jerk = float(np.nanmax(np.abs(jerk)))

    # engine rpm flare
    rpm_flare = np.nan
    if "engine_rpm__canon" in df.columns:
        rpm = pd.to_numeric(df["engine_rpm__canon"], errors="coerce").to_numpy()
        idx_pre = np.where(pre_mask & ~np.isnan(rpm))[0]
        idx_win = np.where(win_mask & ~np.isnan(rpm))[0]
        if idx_pre.size > 0 and idx_win.size > 0:
            pre_rpm = float(np.nanmedian(rpm[idx_pre]))
            max_rpm = float(np.nanmax(rpm[idx_win]))
            rpm_flare = max(0.0, max_rpm - pre_rpm)

    # torque hole (prefer axle torque, then delivered engine torque, then engine torque)
    torque_series = pick_first_existing(
        df,
        [
            "Actual Axle Torque",
            "Delivered Engine Torque",
            "engine_torque",
            "engine_torque_trans",
            "engine_torque_ecm",
            "Trans Engine Torque",
        ],
    )
    torque_hole = np.nan
    if torque_series is not None:
        torque = torque_series.to_numpy()
        idx_pre_t = np.where(pre_mask & ~np.isnan(torque))[0]
        idx_win_t = np.where(win_mask & ~np.isnan(torque))[0]
        if idx_pre_t.size > 0 and idx_win_t.size > 0:
            pre_t = float(np.nanmedian(torque[idx_pre_t]))
            min_t = float(np.nanmin(torque[idx_win_t]))
            torque_hole = max(0.0, pre_t - min_t)

    # TCC slip spike (from tcc_slip_fused)
    slip_spike = np.nan
    if "tcc_slip_fused" in df.columns:
        slip = pd.to_numeric(df["tcc_slip_fused"], errors="coerce").to_numpy()
        idx_pre_s = np.where(pre_mask & ~np.isnan(slip))[0]
        idx_win_s = np.where(win_mask & ~np.isnan(slip))[0]
        if idx_pre_s.size > 0 and idx_win_s.size > 0:
            pre_s = float(np.nanmedian(np.abs(slip[idx_pre_s])))
            max_s = float(np.nanmax(np.abs(slip[idx_win_s])))
            slip_spike = max(0.0, max_s - pre_s)

    # Chassis: max abs lat-g and yaw (for debug, not used in score yet)
    latg_max = np.nan
    yaw_max = np.nan
    if "Lateral Acceleration" in df.columns:
        latg = pd.to_numeric(df["Lateral Acceleration"], errors="coerce").to_numpy()
        idx_lat = np.where(win_mask & ~np.isnan(latg))[0]
        if idx_lat.size > 0:
            latg_max = float(np.nanmax(np.abs(latg[idx_lat])))
    if "Yaw Rate" in df.columns:
        yaw = pd.to_numeric(df["Yaw Rate"], errors="coerce").to_numpy()
        idx_yaw = np.where(win_mask & ~np.isnan(yaw))[0]
        if idx_yaw.size > 0:
            yaw_max = float(np.nanmax(np.abs(yaw[idx_yaw])))

    return {
        "time_at_shift": float(t0),
        "gear_from": int(gear_prev),
        "gear_to": int(gear_curr),
        "duration_s": float(duration) if not np.isnan(duration) else np.nan,
        "max_abs_jerk": float(max_abs_jerk) if not np.isnan(max_abs_jerk) else np.nan,
        "rpm_flare": float(rpm_flare) if not np.isnan(rpm_flare) else np.nan,
        "torque_hole": float(torque_hole) if not np.isnan(torque_hole) else np.nan,
        "slip_spike": float(slip_spike) if not np.isnan(slip_spike) else np.nan,
        "latg_max": float(latg_max) if not np.isnan(latg_max) else np.nan,
        "yaw_max": float(yaw_max) if not np.isnan(yaw_max) else np.nan,
    }


def build_events_from_full(path: Path) -> pd.DataFrame:
    """Extract comfort, non-WOT upshift events with metrics from a FULL file."""
    print(f"[INFO] Processing FULL file: {path}")
    df = pd.read_csv(path)

    required = [
        "time_s",
        "gear_actual__canon",
        "speed_mph",
        "throttle_pct",
        "mode_is_pattern_a",
        "mode_is_wot",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] Missing required columns in {path.name}: {missing}; skipping file.")
        return pd.DataFrame()

    # sort by time
    df = df.sort_values("time_s").reset_index(drop=True)

    # numeric gear
    gear = pd.to_numeric(df["gear_actual__canon"], errors="coerce")
    gear_prev = gear.shift(1)

    # upshift detection
    mask = (
        gear.notna()
        & gear_prev.notna()
        & (gear > gear_prev)
        & (gear_prev >= 1)
        & (gear <= 6)
    )
    idxs = np.flatnonzero(mask.to_numpy())
    if idxs.size == 0:
        print(f"[INFO] No upshifts found in {path.name}")
        return pd.DataFrame()

    # common arrays
    speed = pd.to_numeric(df["speed_mph"], errors="coerce").to_numpy()
    throttle = pd.to_numeric(df["throttle_pct"], errors="coerce").to_numpy()
    mode_is_pattern = pd.to_numeric(df["mode_is_pattern_a"], errors="coerce").to_numpy()
    mode_is_wot = pd.to_numeric(df["mode_is_wot"], errors="coerce").to_numpy()

    brake_pedal = df["Brake Pedal"] if "Brake Pedal" in df.columns else None
    tcs_request = df["TCS Request"] if "TCS Request" in df.columns else None

    events = []

    for idx in idxs:
        if idx <= 0 or idx >= len(df):
            continue

        g_prev = gear_prev.iloc[idx]
        g_curr = gear.iloc[idx]
        if pd.isna(g_prev) or pd.isna(g_curr):
            continue

        g_prev_i = int(g_prev)
        g_curr_i = int(g_curr)

        # Only simple 1-step upshifts (1->2..5->6)
        if g_curr_i != g_prev_i + 1:
            continue

        # Comfort only: non-pattern A, non-WOT
        if np.isnan(mode_is_pattern[idx]) or np.isnan(mode_is_wot[idx]):
            continue
        if mode_is_pattern[idx] != 0:
            continue
        if mode_is_wot[idx] != 0:
            continue

        # skip braking events if we can detect them
        if brake_pedal is not None:
            val = str(brake_pedal.iloc[idx]).strip()
            if val.lower() == "yes":
                continue

        # skip active TCS requests if present
        if tcs_request is not None:
            val = str(tcs_request.iloc[idx]).strip()
            if val.lower() == "yes":
                continue

        # speed & TPS at shift
        v = speed[idx]
        tps = throttle[idx]
        if np.isnan(v) or np.isnan(tps):
            # need both speed + TPS at event
            continue

        metrics = compute_event_metrics(df, idx, g_prev_i, g_curr_i)
        metrics.update(
            {
                "source_file": path.name,
                "speed_at_shift": float(v),
                "tps_at_shift": float(tps),
                "tps_bin": nearest_tps_bin(float(tps)),
                "mode_is_wot": int(mode_is_wot[idx]),
            }
        )
        events.append(metrics)

    if not events:
        print(f"[INFO] No usable comfort upshifts in {path.name}")
        return pd.DataFrame()

    ev_df = pd.DataFrame(events)
    print(
        f"[INFO] Extracted {len(ev_df)} comfort non-WOT upshift events from {path.name}"
    )
    return ev_df


def compute_scores(events: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized metrics and a composite comfort score per event."""
    if events.empty:
        return events

    df = events.copy()

    metrics = ["duration_s", "max_abs_jerk", "rpm_flare", "torque_hole", "slip_spike"]
    scales = {}
    for m in metrics:
        vals = pd.to_numeric(df[m], errors="coerce")
        v = vals[vals.notna()]
        if v.empty:
            scales[m] = 1.0
        else:
            scales[m] = float(np.nanpercentile(v.to_numpy(), 90)) or 1.0

    scores = []
    for _, row in df.iterrows():
        score = 0.0
        missing = 0

        # Duration
        d = row["duration_s"]
        if pd.notna(d):
            nd = min(d / (scales["duration_s"] + 1e-9), 10.0)
            score += 0.5 * nd
        else:
            missing += 1

        # Jerk
        j = row["max_abs_jerk"]
        if pd.notna(j):
            nj = min(j / (scales["max_abs_jerk"] + 1e-9), 10.0)
            score += 1.0 * nj
        else:
            missing += 1

        # RPM flare
        rf = row["rpm_flare"]
        if pd.notna(rf):
            nrf = min(rf / (scales["rpm_flare"] + 1e-9), 10.0)
            score += 0.7 * nrf
        else:
            missing += 1

        # Torque hole
        th = row["torque_hole"]
        if pd.notna(th):
            nth = min(th / (scales["torque_hole"] + 1e-9), 10.0)
            score += 0.7 * nth
        else:
            missing += 1

        # Slip spike
        ss = row["slip_spike"]
        if pd.notna(ss):
            nss = min(ss / (scales["slip_spike"] + 1e-9), 10.0)
            score += 0.5 * nss
        else:
            missing += 1

        # small penalty for missing metrics so incomplete events aren't "perfect"
        score += 0.1 * missing

        scores.append(score)

    df["comfort_score"] = scores
    return df


def build_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics per (gear_from, tps_bin)."""
    if events.empty:
        return events

    group = events.groupby(["gear_from", "tps_bin"], dropna=True)

    summary = group.agg(
        count=("comfort_score", "size"),
        speed_mean=("speed_at_shift", "mean"),
        speed_std=("speed_at_shift", "std"),
        speed_min=("speed_at_shift", "min"),
        speed_max=("speed_at_shift", "max"),
        score_mean=("comfort_score", "mean"),
        duration_mean=("duration_s", "mean"),
        jerk_mean=("max_abs_jerk", "mean"),
        rpm_flare_mean=("rpm_flare", "mean"),
        torque_hole_mean=("torque_hole", "mean"),
        slip_spike_mean=("slip_spike", "mean"),
        latg_max_mean=("latg_max", "mean"),
        yaw_max_mean=("yaw_max", "mean"),
    )

    # recommended speed from lowest-score events
    def best_speed(g: pd.DataFrame) -> float:
        g_sorted = g.sort_values("comfort_score")
        n = len(g_sorted)
        if n == 0:
            return np.nan
        top_n = min(20, n)
        subset = g_sorted.head(top_n)
        return float(subset["speed_at_shift"].median())

    speed_reco = group.apply(best_speed).rename("speed_reco")
    summary = summary.join(speed_reco)

    return summary.reset_index().sort_values(["gear_from", "tps_bin"])


def build_up_table_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Build a comfort UP table (mph vs TPS) from summary speed_reco."""
    if summary.empty:
        raise ValueError("Empty summary; cannot build comfort UP table.")

    # MultiIndex for speed_reco lookup
    speed_map = summary.set_index(["gear_from", "tps_bin"])["speed_reco"]

    columns = ["mph"] + [str(x) for x in TPS_AXIS]
    rows = []

    for gear_from in range(1, 6):  # 1->2 .. 5->6
        label = f"{gear_from} -> {gear_from + 1} Shift"
        row_vals = []
        for tps in TPS_AXIS:
            val = speed_map.get((gear_from, tps), np.nan)
            row_vals.append(val)
        rows.append([label] + row_vals)

    up = pd.DataFrame(rows, columns=columns)

    # Interpolate along TPS axis for each row; then ffill/bfill.
    for idx in up.index:
        row = pd.to_numeric(up.loc[idx, columns[1:]], errors="coerce")
        if row.notna().sum() >= 2:
            row_interp = row.interpolate(limit_direction="both")
        else:
            row_interp = row.fillna(method="ffill").fillna(method="bfill")
        up.loc[idx, columns[1:]] = row_interp.values

    # Enforce monotonic with TPS (non-decreasing across each row)
    for idx in up.index:
        vals = (
            pd.to_numeric(up.loc[idx, columns[1:]], errors="coerce")
            .to_numpy(dtype=float)
        )
        last = None
        for i, v in enumerate(vals):
            if np.isnan(v):
                continue
            if last is None:
                last = v
            else:
                if v < last:
                    v = last
                last = v
            vals[i] = v
        up.loc[idx, columns[1:]] = vals

    # Round to 0.1 mph
    up[columns[1:]] = up[columns[1:]].astype(float).round(1)
    return up


def build_down_table(up: pd.DataFrame, gap_mph: float) -> pd.DataFrame:
    """Build a DOWN table by subtracting a fixed hysteresis gap from the UP table."""
    down = up.copy()
    cols = list(down.columns)
    for idx in down.index:
        row = (
            pd.to_numeric(down.loc[idx, cols[1:]], errors="coerce")
            .to_numpy(dtype=float)
        )
        new_vals = []
        for v in row:
            if np.isnan(v):
                new_vals.append(np.nan)
            else:
                new_vals.append(max(v - gap_mph, 1.0))
        down.loc[idx, cols[1:]] = new_vals
    down[cols[1:]] = down[cols[1:]].astype(float).round(1)
    return down


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, float_format="%.1f")


def main():
    parser = argparse.ArgumentParser(
        description="Comfort shift-quality metrics + metric-optimized comfort shift tables (log-first)."
    )
    parser.add_argument(
        "--full-glob",
        default="newlogs/cleaned/__trans_focus__clean_FULL__*.csv",
        help="Glob for FULL cleaned CSVs.",
    )
    parser.add_argument(
        "--passes-out-dir",
        default="newlogs/output/02_passes/COMFORT",
        help="Output directory for comfort metric pass outputs.",
    )
    parser.add_argument(
        "--tables-out-dir",
        default="newlogs/output/01_tables/shift",
        help="Output directory for shift tables.",
    )
    parser.add_argument(
        "--comfort-gap",
        type=float,
        default=4.0,
        help="Hysteresis gap (mph) for Comfort DOWN table (DOWN = UP - gap).",
    )

    args = parser.parse_args()
    full_paths = sorted(Path().glob(args.full_glob))
    if not full_paths:
        raise SystemExit(f"No FULL files matched glob: {args.full_glob}")

    print("[INFO] Using FULL files:")
    for p in full_paths:
        print(f"  - {p}")

    # Collect events from all FULL files
    all_events = []
    for p in full_paths:
        ev = build_events_from_full(p)
        if not ev.empty:
            all_events.append(ev)

    if not all_events:
        raise SystemExit("No comfort upshift events found in any FULL file.")

    events = pd.concat(all_events, ignore_index=True)
    print(f"[INFO] Total comfort non-WOT upshift events: {len(events)}")

    # Compute scores
    events_scored = compute_scores(events)

    passes_dir = Path(args.passes_out_dir)
    passes_dir.mkdir(parents=True, exist_ok=True)

    events_out = passes_dir / "SHIFT_QUALITY__COMFORT__EVENTS.csv"
    events_scored.to_csv(events_out, index=False)
    print(f"[OK] Wrote event-level metrics to {events_out}")

    # Summary
    summary = build_summary(events_scored)
    summary_out = passes_dir / "SHIFT_QUALITY__COMFORT__SUMMARY.tsv"
    summary.to_csv(summary_out, sep="\t", index=False)
    print(f"[OK] Wrote comfort summary to {summary_out}")

    # Build comfort UP/DOWN tables from summary
    up = build_up_table_from_summary(summary)
    down = build_down_table(up, gap_mph=args.comfort_gap)

    tables_dir = Path(args.tables_out_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    up_path = tables_dir / "SHIFT_TABLES__UP__Throttle17__COMFORT_METRIC_OPT.tsv"
    down_path = tables_dir / "SHIFT_TABLES__DOWN__Throttle17__COMFORT_METRIC_OPT.tsv"

    write_table(up, up_path)
    write_table(down, down_path)

    print(f"[OK] Wrote comfort metric-optimized UP table to {up_path}")
    print(f"[OK] Wrote comfort metric-optimized DOWN table to {down_path}")


if __name__ == "__main__":
    main()

