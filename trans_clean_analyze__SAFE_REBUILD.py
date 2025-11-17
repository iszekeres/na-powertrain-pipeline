# trans_clean_analyze__SAFE_REBUILD.py  (v3.1, duplicate-header safe + __canon in FULL)
# Clean + map + fused-slip + soft-lock + shift events + mapping + summary
# Usage:
#   python .\trans_clean_analyze__SAFE_REBUILD.py --in-glob ".\newlogs\*.csv" --out-dir ".\06_Logs\Trans_Review"

import argparse, glob, os
from datetime import datetime
import numpy as np, pandas as pd


# === PROGRESS HELPERS (injected) ===
import time, os, sys
from contextlib import contextmanager
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def iter_progress(it, total=None, desc=""):
    if tqdm and total is not None:
        return tqdm(it, total=total, desc=desc, unit="step")
    return it

@contextmanager
def phase(name, enabled=True):
    t0 = time.time()
    if enabled:
        print(f"[>] {name} ...", flush=True)
    try:
        yield
    finally:
        if enabled:
            dt = time.time() - t0
            print(f"[✓] {name} ({dt:.1f}s)", flush=True)

def _read_csv_chunked(path, chunksize, show):
    import pandas as pd
    try:
        filesize = os.path.getsize(path)
        est_rows = max(1, int(filesize / 120))
        est_chunks = max(1, est_rows // max(1, chunksize))
    except Exception:
        est_chunks = None
    chunks = pd.read_csv(path, chunksize=chunksize, low_memory=False)
    if tqdm and show and est_chunks:
        from tqdm import tqdm as _tqdm
        chunks = _tqdm(chunks, total=est_chunks, desc="read_csv", unit="chunk")
    return pd.concat(list(chunks), ignore_index=True)

def install_pd_read_csv_progress(args):
    """Monkey-patch pandas.read_csv to show progress when reading large files.
    Active when --progress and --chunksize>0; safe no-op otherwise.
    """
    import pandas as _pd
    if not getattr(args, "progress", False) or not getattr(args, "chunksize", 0):
        return
    _orig = _pd.read_csv
    def _patched(*a, **k):
        # if first arg is a file path and caller didn't pass chunksize, use chunked
        path = a[0] if a and isinstance(a[0], str) else None
        if path and os.path.exists(path) and "chunksize" not in k:
            try:
                return _read_csv_chunked(path, chunksize=int(getattr(args,"chunksize",0)), show=True)
            except Exception:
                pass
        return _orig(*a, **k)
    _pd.read_csv = _patched
    print(f"[progress] enabled (chunksize={getattr(args,'chunksize',0)})", flush=True)
# === END PROGRESS HELPERS ===

FINAL_DRIVE = 3.08
GEAR_RATIOS = {1:4.03, 2:2.36, 3:1.53, 4:1.15, 5:0.85, 6:0.67}
_SUSTAIN_SEC = 0.4
_REFRACTORY_SEC = 2.0

BRAKE_PRESSURE_ALIASES = ["Brake Pressure (kPa)", "Brake Pressure"]
BRAKE_PRESSURE_THRESHOLD = 15.0  # kPa; above this = brake ON


def derive_brake_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a binary brake flag with strict/no-fallback semantics.

    Priority:
      1) Brake Pedal channel (binary or %):
           - Exact 'Brake Pedal' (case-insensitive), or
           - Any column whose name contains both 'brake' and 'pedal'.
         If values look like a percentage (range > ~1.5), treat as 0–100 and
         use >5% as "pressed". Otherwise, treat non-zero as pressed.
      2) Fallback: analog brake pressure in BRAKE_PRESSURE_ALIASES
         using BRAKE_PRESSURE_THRESHOLD. Emits a warning.
      3) If neither exists, raise a clear error (no silent fallback).
    Output column 'brake' is strict 0/1 (0 = off, 1 = on).
    """
    # 1) Prefer a Brake Pedal-style channel
    pedal_col = None
    lower_cols = {str(c).lower(): c for c in df.columns}
    # exact match first
    for name_lower, orig in lower_cols.items():
        if name_lower == "brake pedal":
            pedal_col = orig
            break
    if pedal_col is None:
        # any column containing both "brake" and "pedal"
        for name_lower, orig in lower_cols.items():
            if "brake" in name_lower and "pedal" in name_lower:
                pedal_col = orig
                break

    if pedal_col is not None:
        raw = df[pedal_col]
        num = pd.to_numeric(raw, errors="coerce")
        if num.notna().any():
            max_val = float(num.max())
            if max_val > 1.5:
                # Treat as percentage or similar; threshold at >5%
                df["brake"] = (num > 5.0).astype(int)
            else:
                # Treat as binary-ish; any non-zero is pressed
                df["brake"] = (num != 0).astype(int)
        else:
            # Non-numeric but present; treat common textual booleans
            s = raw.astype(str).str.strip().str.lower()
            df["brake"] = s.isin(["on", "true", "pressed", "1"]).astype(int)
        return df

    # 2) Fallback: analog brake pressure (older logs)
    src = None
    for c in BRAKE_PRESSURE_ALIASES:
        if c in df.columns:
            src = c
            break

    if src is not None:
        print(
            "[WARN] Using analog brake pressure fallback for brake "
            "(no 'Brake Pedal' channel found)."
        )
        bp = pd.to_numeric(df[src], errors="coerce")
        df["brake"] = (bp > BRAKE_PRESSURE_THRESHOLD).astype(int)
        return df

    # 3) Hard failure: no usable brake signal
    raise SystemExit(
        "[ERROR] No usable brake channel found. "
        "Expected a 'Brake Pedal' style channel or one of: "
        f"{BRAKE_PRESSURE_ALIASES}."
    )


def ffill_with_gap(time_s, series, max_gap=1.0):
    """
    Forward-fill with a maximum allowed temporal gap.

    - time_s: 1D array-like of seconds (monotonic increasing preferred)
    - series: pandas Series to forward-fill
    - max_gap: maximum allowed age (in seconds) for a carried value;
               beyond this, values are set back to NaN.
    """
    t = pd.to_numeric(time_s, errors="coerce")
    s = pd.to_numeric(series, errors="coerce")
    last_valid_time = t.where(s.notna()).ffill()
    s_ff = s.ffill()
    age = t - last_valid_time
    s_ff[age > max_gap] = pd.NA
    return s_ff


def _lock_thr_temp_F(tf):
    if pd.isna(tf): return 25.0
    if tf < 100.0:  return 34.0   # cold
    if tf <= 180.0: return 22.0   # normal
    return 26.0                   # hot

def _estimate_dt(offset):
    s = pd.to_numeric(offset, errors="coerce")
    d = s.diff(); d = d[(d>0) & np.isfinite(d)]
    return float(d.median()) if len(d) else 0.01

def _estimate_gear_series(df):
    if "gear_actual" in df and df["gear_actual"].notna().any():
        return pd.to_numeric(df["gear_actual"], errors="coerce").round().clip(1,6).astype("Int64")
    if {"turbine_rpm","output_rpm"} <= set(df.columns):
        tr = pd.to_numeric(df["turbine_rpm"], errors="coerce").to_numpy(float)
        orpm = pd.to_numeric(df["output_rpm"], errors="coerce").to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = tr / np.where(orpm==0.0, np.nan, orpm)
        ratios = np.array(list(GEAR_RATIOS.values())); keys = np.array(list(GEAR_RATIOS.keys()))
        gear = np.full(len(r), np.nan); valid = np.isfinite(r)
        if valid.any():
            idx = np.argmin(np.abs(r[valid,None]-ratios[None,:]), axis=1)
            gear[valid] = keys[idx]
        return pd.Series(gear, index=df.index)
    return pd.Series(np.nan, index=df.index)

def _fused_slip(df, gear, win_s, dt):
    s1 = df["tcc_slip"].abs() if "tcc_slip" in df else None
    s2 = (df["engine_rpm"] - df["turbine_rpm"]).abs() if {"engine_rpm","turbine_rpm"} <= set(df.columns) else None
    s3 = None
    if {"engine_rpm","output_rpm"} <= set(df.columns):
        ratio_est = gear.map(GEAR_RATIOS).astype(float)
        s3 = (df["engine_rpm"] - (df["output_rpm"] * ratio_est)).abs()
    slip = None
    for cand in (s1, s2, s3):
        if cand is None: continue
        cand = pd.to_numeric(cand, errors="coerce")
        slip = cand.copy() if slip is None else slip.fillna(cand)
    if slip is None: slip = pd.Series([np.nan]*len(df), index=df.index)
    win = max(1, int(win_s/max(dt,1e-3)))
    return slip.rolling(win, min_periods=1).median()

def _build_locked_soft(slip, temp_f, desired, dt):
    thr_temp = temp_f.apply(_lock_thr_temp_F) if "apply" in dir(temp_f) else pd.Series(25.0, index=slip.index)
    des = desired.abs().fillna(np.nan) if desired is not None else pd.Series(np.nan, index=slip.index)
    lock_thr = pd.Series(np.nan, index=slip.index)
    unlock_thr = pd.Series(np.nan, index=slip.index)
    for i in slip.index:
        t = thr_temp.iat[i] if i < len(thr_temp) else 25.0
        d = des.iat[i] if i < len(des) else np.nan
        if np.isfinite(d) and d <= 30.0:
            lock_thr.iat[i]   = min(t, d + 12.0)
            unlock_thr.iat[i] = max(t + 8.0, d + 22.0)
        else:
            lock_thr.iat[i]   = t
            unlock_thr.iat[i] = t + 8.0
    win = max(1, int(_SUSTAIN_SEC/max(dt,1e-3)))
    below_s = (slip <= lock_thr).rolling(win, min_periods=win).apply(lambda x: 1.0 if np.all(x==1.0) else 0.0, raw=True).fillna(0).astype(bool)
    above_s = (slip >= unlock_thr).rolling(win, min_periods=win).apply(lambda x: 1.0 if np.all(x==1.0) else 0.0, raw=True).fillna(0).astype(bool)
    locked = np.zeros(len(slip), dtype=np.int8); state = 0; last_edge = -10**9
    refr = int(_REFRACTORY_SEC/max(dt,1e-6))
    for i in range(len(slip)):
        if (i - last_edge) < refr:
            locked[i] = state; continue
        if above_s.iat[i]:
            if state != 0: state = 0; last_edge = i
        elif below_s.iat[i]:
            if state != 1: state = 1; last_edge = i
        locked[i] = state
    return pd.Series(locked, index=slip.index, dtype="int8")

ALIASES = {
    "speed_mph":   ["Vehicle Speed (SAE)","VSS mph","speed_mph"],
    "throttle_pct":["Throttle Position","Throttle Position (SAE)","throttle_pct"],
    "pedal_pct":   ["Throttle Desired Position","Accelerator Pedal Position","pedal_pct"],
    "engine_rpm":  ["Engine RPM (SAE)","Engine Speed","engine_rpm"],
    "turbine_rpm": ["Trans Turbine RPM","Trans Input Shaft RPM","ISS","turbine_rpm"],
    "output_rpm":  ["Trans Output Shaft RPM","OSS","output_rpm"],
    "tcc_slip":    ["TCC Slip","TCC Slip RPM","tcc_slip"],
    "tcc_desired": ["TCC Desired Slip","tcc_desired"],
    "trans_temp_f":["Trans Fluid Temp","Trans Temp F","Transmission Fluid Temp","trans_temp_f"],
    "gear_actual": ["gear_actual","Trans Current Gear (Actual)","Trans Current Gear"],
    "gear_cmd":    ["gear_cmd","Gear Commanded","Trans Gear Commanded"],
    # Brake is always derived from pressure into 'brake' before mapping.
    "brake":       ["brake"],
    "offset":      ["offset","Offset","Time"],
}

KEEP_ORDER = ["offset","speed_mph","throttle_pct","pedal_pct","gear_actual","gear_cmd",
              "engine_rpm","turbine_rpm","output_rpm","tcc_slip","tcc_desired",
              "trans_temp_f","trans_temp_c","brake"]

# --- duplicate-header-safe getters
def _cols_matching(df, name):
    return [i for i,c in enumerate(df.columns) if c == name]

def _coalesce_block_to_series(block, numeric=True):
    if isinstance(block, pd.Series):
        s = block
        return pd.to_numeric(s, errors="coerce") if numeric else s
    if numeric:
        block = block.apply(pd.to_numeric, errors="coerce")
    if block.shape[1] == 1:
        s = block.iloc[:,0]
    else:
        s = block.ffill(axis=1).iloc[:,-1]
    return s

def _get_series_dupesafe(df, name, numeric=True):
    idxs = _cols_matching(df, name)
    if not idxs:
        return None
    block = df.iloc[:, idxs]
    return _coalesce_block_to_series(block, numeric=numeric)

def _map_df(df):
    # promote row-0 labels if headers are numeric + first row textual
    try:
        if all(str(c).isdigit() for c in df.columns) and df.shape[0] > 0:
            first = df.iloc[0].astype(str).tolist()
            if sum(any(ch.isalpha() for ch in s) for s in first) >= 5:
                df = df.iloc[1:].reset_index(drop=True); df.columns = first
    except Exception:
        pass

    out = {}; mapping_rows = []

    for canon, choices in ALIASES.items():
        src_names = [nm for nm in choices if (nm in df.columns)]
        if not src_names: 
            continue
        s = None; picked = None
        for nm in src_names:
            ser = _get_series_dupesafe(df, nm, numeric=(canon!="brake"))
            if ser is not None:
                s = ser; picked = nm
                break
        if s is None: 
            continue
        out[canon] = s
        mapping_rows.append({"canonical": canon, "source": picked})

    out = pd.DataFrame(out)

    if "trans_temp_f" in out.columns:
        out["trans_temp_c"] = (pd.to_numeric(out["trans_temp_f"], errors="coerce") - 32.0) * (5.0/9.0)

    if "brake" in out.columns:
        b = pd.to_numeric(out["brake"], errors="coerce")
        if b.notna().any():
            out["brake"] = (b > 0.5).astype(int)
        else:
            out["brake"] = out["brake"].astype(str).str.lower().isin(["on","true","pressed","1"]).astype(int)

    if "offset" not in out.columns:
        out["offset"] = np.arange(len(out)) * 0.01

    for c in [c for c in out.columns if c!="brake"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    for k in KEEP_ORDER:
        if k not in out.columns: out[k] = np.nan
    out = out[KEEP_ORDER]
    map_df = pd.DataFrame(mapping_rows)
    return out, map_df

def _add_tcc_softlock(clean_df):
    dt = _estimate_dt(clean_df["offset"])
    gear = _estimate_gear_series(clean_df)
    slip = _fused_slip(clean_df, gear, win_s=0.25, dt=dt)
    desired = clean_df["tcc_desired"] if "tcc_desired" in clean_df.columns else None
    tempf   = clean_df["trans_temp_f"] if "trans_temp_f" in clean_df.columns else pd.Series(np.nan, index=clean_df.index)
    built = _build_locked_soft(slip, tempf, desired, dt)
    clean_df["tcc_slip_fused"] = slip
    clean_df["tcc_locked_built"] = built
    return clean_df


def _shift_events(full_df: pd.DataFrame, tag: str = "<unknown>") -> pd.DataFrame:
    """
    Build SHIFT_EVENTS v2 from a CLEAN_FULL-like dataframe.

    Uses canonical columns:
      - time_s
      - gear_actual, gear_cmd   (0–6 after canon/step-hold)
      - speed_mph, throttle_pct, pedal_pct (optional)
      - engine_rpm, turbine_rpm (optional)
      - tcc_slip_fused, tcc_locked_built
      - brake
      - mode_trans (optional, any trans mode channel)

    For each commanded gear change (UP/DOWN between 1–6):
      - t_cmd: time of gear_cmd edge
      - t_act: time when gear_actual settles to new gear
      - t_event: midpoint 0.5 * (t_cmd + t_act)
      - Samples *_event fields via merge_asof on time_s.
      - Computes shift_duration_s, tcc_slip_max, harshness_metric (placeholder),
        mode_trans, brake_event, tcc_locked_start.
    """
    required = [
        "time_s",
        "gear_actual",
        "gear_cmd",
        "speed_mph",
        "throttle_pct",
        "brake",
        "tcc_slip_fused",
        "tcc_locked_built",
    ]
    missing = [c for c in required if c not in full_df.columns]
    if missing:
        raise SystemExit(
            f"[ERROR] SHIFT_EVENTS v2 missing required columns in FULL: {missing}"
        )

    df = full_df.copy()
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df = df.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)

    for col in [
        "gear_actual",
        "gear_cmd",
        "speed_mph",
        "throttle_pct",
        "pedal_pct",
        "engine_rpm",
        "turbine_rpm",
        "tcc_slip_fused",
        "brake",
        "tcc_locked_built",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Step-hold for discrete gear series.
    for col in ["gear_actual", "gear_cmd"]:
        df[col] = df[col].ffill()

    ts = df["time_s"].to_numpy()
    g_cmd = df["gear_cmd"].to_numpy()
    g_act = df["gear_actual"].to_numpy()
    slip = df["tcc_slip_fused"].to_numpy()
    locked = df["tcc_locked_built"].to_numpy()

    SHIFT_UP_PAIRS = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    SHIFT_DN_PAIRS = [(2, 1), (3, 2), (4, 3), (5, 4), (6, 5)]
    VALID_PAIRS = set(SHIFT_UP_PAIRS + SHIFT_DN_PAIRS)

    events_meta = []
    n = len(df)
    min_stable_samples = 3

    for i in range(1, n):
        prev_cmd = g_cmd[i - 1]
        curr_cmd = g_cmd[i]
        if np.isnan(prev_cmd) or np.isnan(curr_cmd):
            continue
        if prev_cmd == curr_cmd:
            continue
        pair = (int(prev_cmd), int(curr_cmd))
        if pair not in VALID_PAIRS:
            continue

        # Commanded shift from prev_cmd -> curr_cmd at index i
        t_cmd = ts[i]

        # Find when actual gear settles to the new gear.
        target = curr_cmd
        j_act = None
        j = i
        while j <= n - min_stable_samples:
            window = g_act[j : j + min_stable_samples]
            if np.all(window == target):
                j_act = j
                break
            j += 1

        if j_act is None:
            # No clear settle point; skip this event.
            continue

        t_act = ts[j_act]
        t_event = 0.5 * (t_cmd + t_act)

        # TCC slip metrics over [i, j_act]
        lo = i
        hi = j_act
        if hi < lo:
            hi = lo
        window_slip = slip[lo : hi + 1]
        if window_slip.size and np.isfinite(window_slip).any():
            tcc_slip_max = float(np.nanmax(window_slip))
        else:
            tcc_slip_max = np.nan

        tcc_locked_start = (
            float(locked[i]) if i < len(locked) and np.isfinite(locked[i]) else np.nan
        )
        shift_duration_s = float(t_act - t_cmd)

        events_meta.append(
            {
                "time_s": t_event,
                "from_gear": int(prev_cmd),
                "to_gear": int(curr_cmd),
                "t_cmd": t_cmd,
                "t_act": t_act,
                "shift_duration_s": shift_duration_s,
                "tcc_slip_max": tcc_slip_max,
                # Placeholder for now; can be refined later.
                "harshness_metric": 0.0,
                "tcc_locked_start": tcc_locked_start,
            }
        )

    cols_final = [
        "time_s",
        "from_gear",
        "to_gear",
        "speed_mph_event",
        "throttle_pct_event",
        "pedal_pct_event",
        "engine_rpm_event",
        "turbine_rpm_event",
        "shift_duration_s",
        "tcc_slip_max",
        "harshness_metric",
        "mode_trans",
        "brake_event",
        "tcc_locked_start",
    ]

    if not events_meta:
        print(f"[WARN] No shift events detected in CLEAN_FULL for {tag}.")
        return pd.DataFrame(columns=cols_final)

    events_df = pd.DataFrame(events_meta)
    events_df = events_df.sort_values("time_s").reset_index(drop=True)

    # Sample canonical channels at t_event via merge_asof.
    sample_cols = [
        "speed_mph",
        "throttle_pct",
        "pedal_pct",
        "engine_rpm",
        "turbine_rpm",
        "brake",
    ]
    if "mode_trans" in df.columns:
        sample_cols.append("mode_trans")

    present = [c for c in sample_cols if c in df.columns]
    slim = df[["time_s"] + present].copy()
    slim = slim.sort_values("time_s").reset_index(drop=True)

    merged = pd.merge_asof(
        events_df.sort_values("time_s").reset_index(drop=True),
        slim,
        on="time_s",
        direction="nearest",
    )

    # Rename sampled columns to *_event, except for mode_trans.
    rename_map = {}
    for c in present:
        if c == "mode_trans":
            continue
        if c == "brake":
            rename_map[c] = "brake_event"
        else:
            rename_map[c] = f"{c}_event"
    merged = merged.rename(columns=rename_map)

    # Ensure all expected columns exist, even if optional ones are missing.
    for col in cols_final:
        if col not in merged.columns:
            merged[col] = np.nan

    merged = merged[cols_final]
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_single", help="Single input raw CSV")
    ap.add_argument("--in-glob", help="Glob for input raw CSVs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars and phase timings",
    )
    ap.add_argument(
        "--chunksize",
        type=int,
        default=0,
        help=(
            "If >0, read CSV in chunks of this many rows "
            "(enables a read progress bar)"
        ),
    )
    args = ap.parse_args()

    # progress: install pandas.read_csv monkey-patch
    try:
        install_pd_read_csv_progress(args)
    except Exception:
        pass

    if not args.in_single and not args.in_glob:
        raise SystemExit("[ERROR] One of --in or --in-glob is required.")
    if args.in_single and args.in_glob:
        raise SystemExit("[ERROR] Use only one of --in or --in-glob, not both.")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.in_single:
        paths = [args.in_single]
    else:
        paths = sorted(glob.glob(args.in_glob))

    if not paths:
        which = args.in_single if args.in_single else args.in_glob
        print(f"[INFO] No raw CSVs in {which} - nothing to do."); return

    for raw in paths:
        try:
            base = os.path.splitext(os.path.basename(raw))[0]
            stamp = datetime.now().strftime("%Y%m%d__%H%M%S"); tag = base

            df = pd.read_csv(raw, encoding="utf-8-sig", low_memory=False)

            # Ensure a time_s column exists for downstream builders.
            if "time_s" not in df.columns:
                cols_lower = {c.lower(): c for c in df.columns}
                for cand in ["time_s", "offset", "Time (s)", "time", "time_sec"]:
                    key = cand.lower()
                    if key in cols_lower:
                        df["time_s"] = df[cols_lower[key]]
                        break

            if "time_s" not in df.columns:
                raise SystemExit(
                    "[ERROR] Could not derive time_s; expected one of "
                    "['time_s','offset','Time (s)','time','time_sec']."
                )

            df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
            if df["time_s"].isna().all():
                raise SystemExit("[ERROR] time_s column is all NaN.")

            # Use time_s as the canonical time axis and offset.
            df = df.sort_values("time_s").reset_index(drop=True)
            df["offset"] = df["time_s"]

            df = derive_brake_flag(df)
            clean_df, map_df = _map_df(df)

            # Add canonical time axis into clean_df.
            clean_df["time_s"] = df["time_s"].values

            # Apply neutral ffill-with-gap for key continuous signals.
            t = clean_df["time_s"]
            for col in ["speed_mph", "throttle_pct", "pedal_pct"]:
                if col in clean_df.columns:
                    clean_df[col] = ffill_with_gap(t, clean_df[col])

            # Step-hold for discrete gear series.
            for col in ["gear_actual", "gear_cmd"]:
                if col in clean_df.columns:
                    clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce").ffill()

            clean_df = _add_tcc_softlock(clean_df)

            # CLEAN_FULL v2 required canonical columns.
            required_v2 = [
                "time_s",
                "speed_mph",
                "throttle_pct",
                "gear_actual",
                "gear_cmd",
                "brake",
                "tcc_slip_fused",
                "tcc_locked_built",
            ]
            missing_v2 = [c for c in required_v2 if c not in clean_df.columns]
            if missing_v2:
                raise SystemExit(
                    f"[ERROR] CLEAN_FULL v2 missing required canonical columns: {missing_v2}"
                )

            clean_path = os.path.join(args.out_dir, f"__trans_focus__clean__{tag}__{stamp}.csv")
            full_path  = os.path.join(args.out_dir, f"__trans_focus__clean_FULL__{tag}__{stamp}.csv")
            shift_path = os.path.join(args.out_dir, f"__trans_focus__shift_events__{tag}__{stamp}.csv")
            map_path   = os.path.join(args.out_dir, f"__trans_focus__mapping__{tag}__{stamp}.csv")
            sum_path   = os.path.join(args.out_dir, f"__trans_focus__summary__{tag}__{stamp}.txt")

            # write clean (focused)
            clean_df.to_csv(clean_path, index=False)

            # write full (original + appended canonical cols with __canon suffix)
            merged = df.copy()
            for c in clean_df.columns:
                merged[f"{c}__canon"] = clean_df[c]

            # Add unsuffixed v2 canonical columns for convenience.
            direct = [
                "time_s",
                "speed_mph",
                "throttle_pct",
                "pedal_pct",
                "gear_actual",
                "gear_cmd",
                "brake",
                "tcc_slip_fused",
                "tcc_locked_built",
            ]
            for c in direct:
                if c in clean_df.columns:
                    merged[c] = clean_df[c]

            merged.to_csv(full_path, index=False)

            # shift events v2 from FULL
            se = _shift_events(merged, tag=tag)
            se.to_csv(shift_path, index=False)

            # mapping
            map_df.to_csv(map_path, index=False)

            # summary
            lines = []
            lines.append("non-null counts -> " + ", ".join([f"{c}:{int(pd.to_numeric(clean_df[c], errors='coerce').notna().sum())}" for c in clean_df.columns]))
            lines.append("shift pair counts:")
            if not se.empty:
                counts = se.groupby(["from_gear","to_gear"]).size().reset_index(name="count").sort_values("count", ascending=False)
                for _,r in counts.iterrows():
                    lines.append(f"  {int(r['from_gear'])} -> {int(r['to_gear'])}: {int(r['count'])}")
            with open(sum_path,"w",encoding="utf-8") as f:
                f.write("\n".join(lines))

            print(f"[OK] {base}.csv → __trans_focus__clean__{tag}__{stamp}.csv (and FULL, shifts, map, summary)")

        except Exception as e:
            print(f"[ERROR] {os.path.basename(raw)}: {e}")

if __name__ == "__main__":
    main()
