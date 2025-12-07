import os, glob, json, argparse
import pandas as pd
import numpy as np

# ---------------------- Fast utilities ----------------------

def z(x):
    """Fast z-score for large arrays: NumPy-only, robust to NaN/inf."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return x
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = np.nan
    good = ~np.isnan(x)
    if not good.any():
        return np.array([], dtype=float)
    x = x[good]
    if x.size > 500_000:
        step = int(np.ceil(x.size / 500_000.0))
        x = x[::step]
    m = x.mean()
    s = x.std()
    if not np.isfinite(s) or s == 0.0:
        return np.zeros_like(x)
    return (x - m) / s

def _rehdr_if_pid_firstrow_labels(df: pd.DataFrame) -> pd.DataFrame:
    """If columns are numeric PIDs and row 0 contains human labels, use row 0 as header."""
    if df.empty:
        return df
    cols = [str(c) for c in df.columns]
    all_numeric = all(c.isdigit() for c in cols)
    first_row = df.iloc[0].astype(str)
    has_alpha = first_row.str.contains(r"[A-Za-z]", regex=True).any()
    if all_numeric and has_alpha:
        df2 = df.iloc[1:].reset_index(drop=True).copy()
        df2.columns = first_row.values
        return df2
    return df

def _choose_raw_rpm_col(raw_df: pd.DataFrame) -> str:
    """Pick a viable RAW RPM column by variance from common label names."""
    candidates = [
        "Engine RPM","engine_rpm","RPM","Engine Speed",
        "Turbine Speed","Input Shaft Speed","Turbine RPM"
    ]
    best, best_std = None, -1.0
    for c in candidates:
        if c in raw_df.columns:
            s = pd.to_numeric(raw_df[c], errors="coerce").dropna()
            v = float(s.std()) if not s.empty else -1.0
            if v > best_std:
                best, best_std = c, v
    if best is not None:
        print(f"[INFO] Using RAW RPM column: {best} (std={best_std:.2f})")
        return best

    smells = [c for c in raw_df.columns if "rpm" in c.lower() or "engine" in c.lower()]
    for c in smells:
        s = pd.to_numeric(raw_df[c], errors="coerce").dropna()
        v = float(s.std()) if not s.empty else -1.0
        if v > best_std:
            best, best_std = c, v
    if best is None:
        raise RuntimeError("No suitable RAW RPM column found after re-header.")
    print(f"[INFO] Using RAW RPM column (fallback): {best} (std={best_std:.2f})")
    return best

def _align_time_by_correlation(clean_t, clean_rpm, raw_t, raw_rpm, max_offset_s=10.0, step_s=0.25, min_pts=200):
    """Robust alignment by correlation with guards for empty/low-variance inputs."""
    def _prep(t, y):
        t = np.asarray(t, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if t.size == 0 or y.size == 0:
            return None, None
        good = np.isfinite(t) & np.isfinite(y)
        t = t[good]; y = y[good]
        if t.size == 0 or y.size == 0:
            return None, None
        if np.nanmax(y) - np.nanmin(y) < 50.0:
            return None, None
        return t, y

    ct, cy = _prep(clean_t, clean_rpm)
    rt, ry = _prep(raw_t, raw_rpm)
    if ct is None or rt is None:
        return 0.0, float("-inf")

    fs = 10.0
    t0 = max(ct.min(), rt.min())
    t1 = min(ct.max(), rt.max())
    if not np.isfinite([t0, t1]).all() or t1 - t0 < 5.0:
        return 0.0, float("-inf")
    grid = np.arange(t0, t1, 1.0/fs)

    cyi = np.interp(grid, ct, cy, left=np.nan, right=np.nan)
    ryi = np.interp(grid, rt, ry, left=np.nan, right=np.nan)
    good = np.isfinite(cyi) & np.isfinite(ryi)
    cyi = cyi[good]; ryi = ryi[good]
    if cyi.size < min_pts:
        return 0.0, float("-inf")

    def _z(v):
        v = v - v.mean()
        s = v.std()
        if not np.isfinite(s) or s == 0.0:
            return None
        return v / s

    cyz = _z(cyi); ryz = _z(ryi)
    if cyz is None or ryz is None:
        return 0.0, float("-inf")

    offs = np.arange(-max_offset_s, max_offset_s + 1e-9, step_s)
    best_corr = float("-inf"); best_dt = 0.0
    n = cyz.size
    for dt in offs:
        shift = int(round(dt * fs))
        if shift >= 0:
            a = cyz[shift:]
            b = ryz[:n-shift]
        else:
            a = cyz[:n+shift]
            b = ryz[-shift:]
        m = min(a.size, b.size)
        if m < min_pts:
            continue
        c = float((a[:m] * b[:m]).sum()) / max(1.0, m)
        if c > best_corr:
            best_corr = c
            best_dt = dt
    return best_dt, best_corr

# ---------------------- Main flow ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-glob", default=r".\06_Logs\Trans_Review\__trans_focus__clean__*.csv")
    ap.add_argument("--raw-glob", default=r".\newlogs\*.csv")
    ap.add_argument("--max-offset-s", type=float, default=10.0)
    ap.add_argument("--step-s", type=float, default=0.25)
    ap.add_argument("--out-dir", default=r".\06_Logs\Trans_Review")
    args = ap.parse_args()

    clean_files = sorted(glob.glob(args.clean_glob))
    raw_files = sorted(glob.glob(args.raw_glob))

    if not raw_files:
        raise SystemExit("[ERROR] No RAW CSVs found under .\\newlogs\\")
    raw_path = raw_files[0]

    # Load RAW and re-header if needed
    raw_df = pd.read_csv(raw_path, encoding="utf-8-sig", engine="python", on_bad_lines="skip", low_memory=False)
    raw_df = _rehdr_if_pid_firstrow_labels(raw_df)

    # Choose RAW time column (best-effort)
    raw_t = None
    for tname in ["Time", "Time (s)", "Time(s)", "Time [s]", "Seconds", "offset", "time"]:
        if tname in raw_df.columns:
            raw_t = pd.to_numeric(raw_df[tname], errors="coerce").values
            break
    if raw_t is None:
        raw_t = np.arange(len(raw_df), dtype=float) / 10.0

    # Choose RAW RPM column
    raw_rpm_col = _choose_raw_rpm_col(raw_df)
    raw_rpm = pd.to_numeric(raw_df[raw_rpm_col], errors="coerce").values

    # Iterate clean files
    alignments = []
    for cpath in clean_files:
        try:
            cdf = pd.read_csv(cpath, encoding="utf-8-sig", engine="python", on_bad_lines="skip", low_memory=False)
        except Exception as e:
            print(f"[WARN] Could not read clean file {os.path.basename(cpath)}: {e}")
            continue

        # Clean time source
        clean_t = None
        for tname in ["offset","Time","time","Seconds","Time (s)","Time[s]"]:
            if tname in cdf.columns:
                clean_t = pd.to_numeric(cdf[tname], errors="coerce").values
                break
        if clean_t is None:
            clean_t = np.arange(len(cdf), dtype=float) / 10.0

        if "engine_rpm" not in cdf.columns:
            print(f"[WARN] No engine_rpm in {os.path.basename(cpath)}. Skipping this clean file.")
            continue
        clean_rpm = pd.to_numeric(cdf["engine_rpm"], errors="coerce").values

        dt, corr = _align_time_by_correlation(clean_t, clean_rpm, raw_t, raw_rpm,
                                              max_offset_s=args.max_offset_s, step_s=args.step_s)
        if not np.isfinite(corr) or corr == float("-inf"):
            print(f"[WARN] Could not confidently align RAW for {os.path.basename(cpath)} (best corr={corr}). Skipping this clean file.")
            continue

        alignments.append({
            "clean_file": os.path.basename(cpath),
            "dt_seconds": float(dt),
            "corr_score": float(corr)
        })

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "ROADLOAD__MULTILOG__estimate.json")
    out_txt  = os.path.join(args.out_dir, "ROADLOAD__MULTILOG__summary.txt")

    payload = {
        "raw_file": os.path.basename(raw_path),
        "raw_rpm_col": raw_rpm_col,
        "clean_files_used": [a["clean_file"] for a in alignments],
        "alignments": alignments,
        "params": {
            "max_offset_s": args.max_offset_s,
            "step_s": args.step_s
        }
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        f"RAW file: {os.path.basename(raw_path)}",
        f"RAW rpm col: {raw_rpm_col}",
        f"Clean files used: {len(alignments)}",
    ]
    for a in alignments:
        lines.append(f"- {a['clean_file']}: dt={a['dt_seconds']:.2f}s, corr={a['corr_score']:.4f}")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[OK] Wrote ROADLOAD__MULTILOG__estimate.json and ROADLOAD__MULTILOG__summary.txt to {args.out_dir}")

if __name__ == "__main__":
    main()
