# -*- coding: utf-8 -*-
# passes_common.py — RAW-only helpers for pass scripts (strict/no-fallback)
import os, numpy as np, pandas as pd

# Canonical 17-pt TPS axis & table header
TPS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
HDR = ["mph"] + [str(x) for x in TPS] + ["%"]

# Shift row labels
ROWS_UP = ["1 -> 2 Shift","2 -> 3 Shift","3 -> 4 Shift","4 -> 5 Shift","5 -> 6 Shift"]
ROWS_DN = ["2 -> 1 Shift","3 -> 2 Shift","4 -> 3 Shift","5 -> 4 Shift","6 -> 5 Shift"]

# RAW column names we will accept (no aliases)
RAW = {
  "time":     ["time_s__canon","time_s","Offset","Time (s)","offset"],
  "speed":    ["speed_mph__canon","speed_mph","Vehicle Speed (SAE)","Speed"],
  "thr":      ["throttle_pct__canon","throttle_pct","Throttle Position","Accelerator Pedal Position"],
  "ga":       ["gear_actual","gear_actual__canon","Trans Gear","Gear Actual"],
  "gc":       ["gear_cmd","gear_cmd__canon","Trans Gear Commanded","Gear Commanded"],
  "eng":      ["engine_rpm__canon","engine_rpm","Engine RPM (SAE)","Engine Speed"],
  "turb":     ["turbine_rpm__canon","turbine_rpm","Trans Turbine RPM","Turbine Speed"],
  "brake":    ["brake__canon","brake","Brake Pressure","Brake"],
  "tft":      ["tftF__canon","Trans Fluid Temp","Trans Fluid Temp (SAE)"],
  "ect":      ["ectF__canon","Engine Coolant Temp (SAE)"],
  "latg":     ["Lateral Acceleration","lat_g","LatAcc","lateral acceleration"],
  "yaw":      ["Yaw Rate","yaw_rate_deg_s","Yaw Rate (deg/s)"],
  "steer":    ["Steering Wheel Position","Steering Angle","Steering Wheel Angle"],
}

def _read_header(path):
    return pd.read_csv(path, nrows=0).columns.tolist()

def require_columns(path, req_lists, hdr=None):
    if hdr is None:
        hdr = _read_header(path)
    miss = []
    for candidates in req_lists:
        if not any(c in hdr for c in candidates):
            miss.append(candidates[0])
    if miss:
        raise RuntimeError(f"[MISS] {os.path.basename(path)} missing: {', '.join(miss)}")

def load_clean_list(p):
    if not os.path.exists(p): raise RuntimeError(f"[MISS] clean-list: {p}")
    with open(p, "r", encoding="utf-8") as f:
        files = [ln.strip() for ln in f if ln.strip()]
    if not files: raise RuntimeError("[MISS] clean-list is empty.")
    return files

def _num(s): return pd.to_numeric(s, errors="coerce").to_numpy()

def load_raw_arrays(fp, need):
    hdr = _read_header(fp)
    require_columns(fp, [RAW[k] for k in need], hdr=hdr)
    col_map = {}
    out = {}
    for k in need:
        for cand in RAW[k]:
            if cand in hdr:
                col_map[k] = cand
                break
        else:
            raise RuntimeError(f"[MISS] {os.path.basename(fp)} missing usable data for {k}")
    df = pd.read_csv(fp, usecols=list(col_map.values()), low_memory=False)
    for k, col in col_map.items():
        series = df[col]
        if series.isna().all():
            raise RuntimeError(f"[MISS] {os.path.basename(fp)} column {col} all NaN")
        out[k] = _num(series)
    return out

def tps_bin(v):
    # nearest neighbor on the 17-pt axis
    return int(np.argmin(np.abs(np.array(TPS, float) - float(v))))

def makedelta(rows, deltas):
    # rows: list of row labels; deltas: array shape[rows, len(TPS)]
    df = pd.DataFrame(columns=HDR); df["mph"] = rows
    for i in range(len(rows)):
        vals = deltas[i]
        df.loc[i, df.columns[1:-1]] = [("" if (np.isnan(x) or x==0) else f"{x:.1f}") for x in vals]
        df.loc[i, "%"] = ""
    return df

def write_delta(out_dir, name, rows, deltas):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    makedelta(rows, deltas).to_csv(p, sep="\t", index=False)
    return p

def read_shift_base(which="UP", base_dir=r".\newlogs\output\01_tables\shift\TF_BASE"):
    fname = f"SHIFT_TABLES__{which}__Throttle17.tsv"
    path = os.path.join(base_dir, fname)
    if not os.path.exists(path):
        raise RuntimeError(f"[MISS] shift base not found: {path}")
    df = pd.read_csv(path, sep="\t")
    # returns rows x TPS-numeric (5x17)
    rows = ROWS_UP if which=="UP" else ROWS_DN
    mat = np.vstack([
        pd.to_numeric(df.loc[df["mph"]==r, df.columns[1:-1]].iloc[0], errors="coerce").to_numpy()
        for r in rows
    ])
    return mat

def read_tcc_base(which="APPLY", base_dir=r".\newlogs\output\01_tables\tcc"):
    fname = f"TCC_{which}__Throttle17.tsv"
    path = os.path.join(base_dir, fname)
    if not os.path.exists(path):
        raise RuntimeError(f"[MISS] tcc base not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)
    rows = [f"{g} {which.title()}" for g in ["1st","2nd","3rd","4th","5th","6th"]]
    mat = np.vstack([
        pd.to_numeric(df.loc[df["mph"]==r, df.columns[1:-1]].iloc[0], errors="coerce").to_numpy()
        for r in rows
    ])
    return mat
