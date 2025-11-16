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
    Always derive a binary brake flag from brake pressure.

    Raw logs are expected to have 'Brake Pressure (kPa)' or 'Brake Pressure'.
    Output column 'brake' is strict 0/1 (0 = off, 1 = on).
    """
    src = None
    for c in BRAKE_PRESSURE_ALIASES:
        if c in df.columns:
            src = c
            break

    if src is None:
        raise SystemExit(
            "[ERROR] No brake pressure column found. "
            "Expected one of: 'Brake Pressure (kPa)' or 'Brake Pressure'."
        )

    bp = pd.to_numeric(df[src], errors="coerce")
    df["brake"] = (bp > BRAKE_PRESSURE_THRESHOLD).astype(int)
    return df


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

def _shift_events(clean_df):
    se = []
    if "gear_actual" not in clean_df.columns: return pd.DataFrame(se)
    g = pd.to_numeric(clean_df["gear_actual"], errors="coerce").round().astype("Int64").ffill()
    chg = g != g.shift(1)
    idx = np.where(chg.fillna(False))[0]
    for i in idx:
        if i==0: continue
        frm = g.iat[i-1]; to = g.iat[i]
        if pd.isna(frm) or pd.isna(to) or frm==to: continue
        mph = float(pd.to_numeric(clean_df["speed_mph"], errors="coerce").iat[i]) if "speed_mph" in clean_df else np.nan
        tps = float(pd.to_numeric(clean_df["throttle_pct"], errors="coerce").iat[i]) if "throttle_pct" in clean_df else np.nan
        ts  = float(pd.to_numeric(clean_df["offset"], errors="coerce").iat[i]) if "offset" in clean_df else np.nan
        se.append({"time_s":ts,"from":int(frm),"to":int(to),"mph":mph,"tps":tps})
    return pd.DataFrame(se)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-glob", required=True)
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

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(args.in_glob))
    if not paths:
        print(f"[INFO] No raw CSVs in {args.in_glob} - nothing to do."); return

    for raw in paths:
        try:
            base = os.path.splitext(os.path.basename(raw))[0]
            stamp = datetime.now().strftime("%Y%m%d__%H%M%S"); tag = base

            df = pd.read_csv(raw, encoding="utf-8-sig", low_memory=False)

            # Ensure a time_s column exists for downstream RAWEDGES/TCC builders.
            if "time_s" not in df.columns:
                cols_lower = {c.lower(): c for c in df.columns}
                for cand in ["time_s", "offset", "Time (s)", "time", "time_sec"]:
                    key = cand.lower()
                    if key in cols_lower:
                        df["time_s"] = df[cols_lower[key]]
                        break

            df = derive_brake_flag(df)
            clean_df, map_df = _map_df(df)
            clean_df = _add_tcc_softlock(clean_df)

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
            merged.to_csv(full_path, index=False)

            # shift events
            se = _shift_events(clean_df)
            se.to_csv(shift_path, index=False)

            # mapping
            map_df.to_csv(map_path, index=False)

            # summary
            lines = []
            lines.append("non-null counts -> " + ", ".join([f"{c}:{int(pd.to_numeric(clean_df[c], errors='coerce').notna().sum())}" for c in clean_df.columns]))
            lines.append("shift pair counts:")
            if not se.empty:
                counts = se.groupby(["from","to"]).size().reset_index(name="count").sort_values("count", ascending=False)
                for _,r in counts.iterrows():
                    lines.append(f"  {int(r['from'])} -> {int(r['to'])}: {int(r['count'])}")
            with open(sum_path,"w",encoding="utf-8") as f:
                f.write("\n".join(lines))

            print(f"[OK] {base}.csv → __trans_focus__clean__{tag}__{stamp}.csv (and FULL, shifts, map, summary)")

        except Exception as e:
            print(f"[ERROR] {os.path.basename(raw)}: {e}")

if __name__ == "__main__":
    main()
