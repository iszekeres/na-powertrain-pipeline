#!/usr/bin/env python3
# consist_pass_weighted.py
# Strict/no-fallback CONSIST pass:
# - Reads CLEAN_FULL files listed in --clean-list
# - Finds 1-gear step edges (+1 UP, -1 DOWN) with warm + no-brake gating
# - Aggregates per TPS bin (17-pt axis) using robust medians
# - Writes SUGGESTED absolute tables and (if baseline present) DELTA tables
#   * DELTA clamps: |delta| <= --max-step (default 0.35 mph)
#   * DOWN deltas: positive values clamped to 0 (never reduce hysteresis)
# - 0.1 mph rounding; canonical header; no fallbacks.

import os, sys, argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

TPS_AXIS = np.array([0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100], dtype=float)
ROW_UP   = ["1 -> 2 Shift","2 -> 3 Shift","3 -> 4 Shift","4 -> 5 Shift","5 -> 6 Shift"]
ROW_DN   = ["2 -> 1 Shift","3 -> 2 Shift","4 -> 3 Shift","5 -> 4 Shift","6 -> 5 Shift"]
HDR      = ["mph"] + [str(int(x)) for x in TPS_AXIS] + ["%"]

REQ = [
  "time_s__canon","speed_mph__canon","throttle_pct__canon",
  "gear_actual__canon","brake__canon","gear_cmd__canon"
]

KEY_CANON_COLS = [
  "time_s__canon","speed_mph__canon",
  "gear_actual__canon","throttle_pct__canon"
]

ALIAS_MAP = {
  "time_s__canon": [
      "time_s__canon","time_s","Time_s","offset","Offset","Time (s)","Time_S"
  ],
  "speed_mph__canon": [
      "speed_mph__canon","speed_mph","Vehicle Speed (SAE)","vehicle_speed_mph"
  ],
  "gear_actual__canon": [
      "gear_actual__canon","gear_actual","Trans Current Gear","Transmission Gear",
      "Trans Gear","Gear Actual"
  ],
  "gear_cmd__canon": [
      "gear_cmd__canon","gear_cmd","Trans Commanded Gear","Trans Gear Commanded",
      "Gear Commanded","Transmission Commanded Gear"
  ],
  "throttle_pct__canon": [
      "throttle_pct__canon","throttle_pct","Accelerator Pedal Position",
      "Accelerator Pedal Position (%)","Throttle Position (%)","Pedal Position (%)"
  ],
  "oncoming_clutch__canon": [
      "oncoming_clutch__canon","oncoming_clutch","Oncoming Clutch",
      "C4 Oncoming","Clutch 4 Oncoming"
  ],
  "tftF__canon": [
      "tftF__canon","trans_temp_f__canon","trans fluid temp (f)",
      "Trans Fluid Temp","Trans Fluid Temp (SAE)"
  ],
  "ectF__canon": [
      "ectF__canon","engine_coolant_temp__canon","engine coolant temp (f)",
      "Engine Coolant Temp (SAE)"
  ],
}

def fail(msg): print(msg, file=sys.stderr); sys.exit(2)

def nearest_tps_idx(val):
    # clip [0,100], then nearest axis bin
    v = 0.0 if np.isnan(val) else float(min(100.0, max(0.0, val)))
    return int(np.argmin(np.abs(TPS_AXIS - v)))

def detect_shift_events(df, path):
    """
    Detect shift events from gear_actual__canon transitions.
    Returns a list of dicts describing each event.
    """
    events = []

    ga = df["gear_int"].astype(float).to_numpy()
    t = df["time_s__canon"].to_numpy()
    v = df["speed_mph__canon"].to_numpy()
    tps = df["throttle_pct__canon"].to_numpy()

    have_cmd = "gear_cmd__canon" in df.columns
    have_clutch = "oncoming_clutch__canon" in df.columns

    dgear = np.diff(ga)
    idx = np.where(dgear != 0)[0]

    for i in idx:
        g_from = ga[i]
        g_to = ga[i + 1]

        if np.isnan(g_from) or np.isnan(g_to):
            continue
        if g_from == g_to:
            continue

        try:
            gb = float(g_from)
            ga_ = float(g_to)
        except (TypeError, ValueError):
            continue

        if not (1.0 <= gb <= 6.0 and 1.0 <= ga_ <= 6.0):
            continue

        dgear = ga_ - gb
        if abs(dgear) != 1.0:
            continue

        table = "UP" if dgear > 0 else "DOWN"
        row = f"{int(gb)} -> {int(ga_)} Shift"

        j = i + 1
        time_event = float(t[j])
        speed_event = float(v[j])
        tps_event = float(tps[j])

        idx_label = df.index[j]
        if have_cmd:
            try:
                gear_cmd_val = float(df.loc[idx_label, "gear_cmd__canon"])
            except Exception:
                gear_cmd_val = float("nan")
        else:
            gear_cmd_val = float("nan")
        if have_clutch:
            try:
                clutch_val = float(df.loc[idx_label, "oncoming_clutch__canon"])
            except Exception:
                clutch_val = float("nan")
        else:
            clutch_val = float("nan")

        events.append({
            "file": path.name,
            "table": table,
            "row": row,
            "time_s": time_event,
            "speed_mph": speed_event,
            "tps": tps_event,
            "gear_int_before": gb,
            "gear_int_after": ga_,
            "gear_actual_before": gb,
            "gear_actual_after": ga_,
            "gear_cmd": gear_cmd_val,
            "oncoming_clutch": clutch_val,
        })

    return events

KEY_CANON_COLS = [
  "time_s__canon","speed_mph__canon","gear_actual__canon","throttle_pct__canon"
]

def ensure_canon(df, path):
    for canon, candidates in ALIAS_MAP.items():
        needs_alias = False
        if canon not in df.columns:
            needs_alias = True
        else:
            if df[canon].isna().all():
                needs_alias = True
        if needs_alias:
            for c in candidates:
                if c in df.columns and not df[c].isna().all():
                    df[canon] = df[c]
                    break
    missing = [c for c in KEY_CANON_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path.name}: missing required column(s): {missing}")
    df = df.dropna(subset=KEY_CANON_COLS, how="any").copy()
    if "gear_actual__canon" not in df.columns:
        raise RuntimeError(f"{path.name}: missing gear_actual__canon for shift detection")
    gear = df["gear_actual__canon"].astype(float).round().astype("Int64")
    df = df.assign(gear_int=gear)
    return df

def load_clean_files(clean_list):
    if not os.path.exists(clean_list):
        fail(f"[MISS] clean_list.txt not found: {clean_list}")
    files = [p for p in open(clean_list, "r", encoding="utf-8").read().splitlines() if p.strip()]
    if not files: fail("[MISS] no files in clean_list.txt")
    frames = []
    all_events = []
    for p in files:
        path = Path(p)
        df = pd.read_csv(path)
        df = ensure_canon(df, path)
        # Warm gating disabled for now (leave old logic for future reference)
        # warm_mask = (
        #     (df["tftF__canon"] >= 100.0) &
        #     (df["ectF__canon"] >= 100.0)
        # )
        warm_mask = pd.Series(True, index=df.index)
        df = df[warm_mask].copy()

        events = detect_shift_events(df, path)
        all_events.extend(events)

        missing_req = [c for c in REQ if c not in df.columns]
        if missing_req:
            fail(f"[FAIL] {path.name} missing required columns: {missing_req}")
        frames.append(df[REQ].copy())
    if not frames:
        fail("[MISS] no usable data after canonical filtering")
    df = pd.concat(frames, ignore_index=True)
    return df, all_events

def extract_edges(df):
    # TEMP: disable warm gating; only basic brake/speed filter
    mask = (
        (df["brake__canon"] <= 1.0) &
        (df["speed_mph__canon"] >= 3.0)
    )
    d = df.loc[mask, ["time_s__canon","speed_mph__canon","throttle_pct__canon","gear_actual__canon"]].dropna()
    if d.empty: return [], []

    g = d["gear_actual__canon"].astype(float).round().astype("Int64")
    # shift edges: +1 (UP), -1 (DOWN)
    dg = g.diff()
    up_idx   = d.index[(dg == 1)].tolist()
    down_idx = d.index[(dg == -1)].tolist()

    up_events = []
    for i in up_idx:
        row = d.loc[i]
        g_from = int(d.loc[i-1, "gear_actual__canon"]) if i-1 in d.index else None
        g_to   = int(row["gear_actual__canon"])
        if g_from is None: continue
        if not (1 <= g_from <= 5 and g_to == g_from+1): continue
        up_events.append((g_from, float(row["speed_mph__canon"]), float(row["throttle_pct__canon"])))

    dn_events = []
    for i in down_idx:
        row = d.loc[i]
        g_to = int(row["gear_actual__canon"])
        g_from = int(d.loc[i-1, "gear_actual__canon"]) if i-1 in d.index else None
        if g_from is None: continue
        if not (2 <= g_from <= 6 and g_to == g_from-1): continue
        dn_events.append((g_from, float(row["speed_mph__canon"]), float(row["throttle_pct__canon"])))

    return up_events, dn_events

def aggregate(events, up=True, min_n=6, std_max=4.0):
    # events: list of (gear_from, mph, tps)
    # returns dict: key=(row_name), value=np.array of length 17 with floats or np.nan
    # also returns counts & stats for debug
    rows = ROW_UP if up else ROW_DN
    # map row name to 'from-gear'
    row_to_gfrom = {rows[i]: (i+1 if up else i+2) for i in range(len(rows))}
    values = {name: [[] for _ in range(len(TPS_AXIS))] for name in rows}

    for g_from, mph, tps in events:
        name = (f"{g_from} -> {g_from+1} Shift" if up else f"{g_from} -> {g_from-1} Shift")
        if name not in values: continue
        j = nearest_tps_idx(tps)
        values[name][j].append(float(mph))

    # robust per-bin statistic
    out = {}
    stats = {}  # (count, median, std)
    for name in rows:
        arr = np.full(len(TPS_AXIS), np.nan, dtype=float)
        st  = []
        for j in range(len(TPS_AXIS)):
            v = np.array(values[name][j], dtype=float)
            v = v[np.isfinite(v)]
            if v.size >= min_n:
                med = float(np.median(v))
                sd  = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
                # optional std filter
                if sd <= std_max or std_max <= 0:
                    arr[j] = med
                st.append((v.size, med, sd))
            else:
                st.append((v.size, np.nan, np.nan))
        # enforce monotone vs TPS (prefix max)
        cur = -1e18
        for j in range(len(arr)):
            if not np.isnan(arr[j]):
                if arr[j] < cur: arr[j] = cur
                cur = arr[j]
        out[name] = arr
        stats[name] = st
    return out, stats

def write_table(path, rows, surf):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("mph\t" + "\t".join([str(int(x)) for x in TPS_AXIS]) + "\t%\n")
        for name in rows:
            vals = surf[name]
            cells = []
            for v in vals:
                if np.isnan(v): cells.append("")
                else:           cells.append(f"{round(v,1):.1f}")
            f.write("\t".join([name] + cells + [""]) + "\n")

def load_baseline(dirpath):
    up_p = os.path.join(dirpath, "SHIFT_TABLES__UP__Throttle17.tsv")
    dn_p = os.path.join(dirpath, "SHIFT_TABLES__DOWN__Throttle17.tsv")
    if not (os.path.exists(up_p) and os.path.exists(dn_p)):
        return None, None
    def rd(p):
        df = pd.read_csv(p, sep="\t")
        if list(df.columns) != HDR: return None
        m = {}
        for name in df["mph"]:
            row = pd.to_numeric(df.loc[df["mph"]==name, df.columns[1:-1]].iloc[0], errors="coerce").to_numpy(dtype=float)
            m[name] = row
        return m
    return rd(up_p), rd(dn_p)

def write_delta(path, rows, sug, base, max_step=0.35, clamp_down_positive=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("mph\t" + "\t".join([str(int(x)) for x in TPS_AXIS]) + "\t%\n")
        for name in rows:
            s = sug[name]
            b = None if base is None else base.get(name, None)
            cells = []
            for j in range(len(TPS_AXIS)):
                sv = s[j]
                if np.isnan(sv) or b is None or b is None or (b is not None and (b is None)):
                    # no suggestion or no baseline -> blank (isolated pass)
                    cells.append("")
                    continue
                bv = b[j]
                if np.isnan(bv):
                    cells.append("")  # baseline empty at this bin
                    continue
                d = float(sv - bv)
                # clamp range
                if d >  max_step: d =  max_step
                if d < -max_step: d = -max_step
                # DOWN special rule: never positive (avoid shrinking hysteresis)
                if clamp_down_positive and name in ROW_DN and d > 0:
                    d = 0.0
                cells.append(f"{round(d,1):.1f}" if abs(d) >= 0.05 else "")  # tiny deltas -> blank
            f.write("\t".join([name] + cells + [""]) + "\n")

def write_debug(out_dir, stats_up, stats_dn):
    # Flatten into a CSV for inspection
    recs = []
    for title, stats in [("UP", stats_up), ("DOWN", stats_dn)]:
        for name, st in stats.items():
            for j, (cnt, med, sd) in enumerate(st):
                recs.append({
                    "table": title,
                    "row": name,
                    "tps_bin": int(TPS_AXIS[j]),
                    "count": int(cnt) if not (cnt is np.nan) else 0,
                    "median_mph": med,
                    "std_mph": sd
                })
    df = pd.DataFrame.from_records(recs)
    df.to_csv(os.path.join(out_dir, "CONSIST__DEBUG_SUMMARY.csv"), index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-list", required=True)
    ap.add_argument("--out-dir", default=r".\newlogs\output\02_passes\CONSIST")
    ap.add_argument("--baseline-dir", default=r".\newlogs\output\01_tables\shift\TF_BASE")
    ap.add_argument("--min-n", type=int, default=6)
    ap.add_argument("--std-max", type=float, default=4.0)
    ap.add_argument("--max-step", type=float, default=0.35)
    args = ap.parse_args()

    df, all_events = load_clean_files(args.clean_list)

    up_ev = []
    dn_ev = []
    for ev in all_events:
        mph = ev["speed_mph"]
        tps = ev["tps"]
        if np.isnan(mph) or np.isnan(tps):
            continue
        try:
            g_from = int(ev["row"].split("->")[0].strip())
        except Exception:
            continue
        if ev["table"] == "UP":
            up_ev.append((g_from, mph, tps))
        else:
            dn_ev.append((g_from, mph, tps))

    up_sug, up_stats = aggregate(up_ev, up=True,  min_n=args.min_n, std_max=args.std_max)
    dn_sug, dn_stats = aggregate(dn_ev, up=False, min_n=args.min_n, std_max=args.std_max)

    # Write SUGGESTED absolute tables
    sug_up_path = os.path.join(args.out_dir, "CONSIST__SHIFT_UP__SUGGESTED.tsv")
    sug_dn_path = os.path.join(args.out_dir, "CONSIST__SHIFT_DOWN__SUGGESTED.tsv")
    write_table(sug_up_path, ROW_UP, up_sug)
    write_table(sug_dn_path, ROW_DN, dn_sug)

    # Optional DELTA vs baseline (if present)
    base_up, base_dn = load_baseline(args.baseline_dir)
    delta_up_path = os.path.join(args.out_dir, "CONSIST__SHIFT_UP__DELTA.tsv")
    delta_dn_path = os.path.join(args.out_dir, "CONSIST__SHIFT_DOWN__DELTA.tsv")
    write_delta(delta_up_path, ROW_UP, up_sug, base_up, max_step=args.max_step, clamp_down_positive=False)
    write_delta(delta_dn_path, ROW_DN, dn_sug, base_dn, max_step=args.max_step, clamp_down_positive=True)

    write_debug(args.out_dir, up_stats, dn_stats)
    if all_events:
        ev_df = pd.DataFrame(all_events)
        events_path = os.path.join(args.out_dir, "CONSIST__SHIFT_EVENTS_DEBUG.csv")
        ev_df.to_csv(events_path, index=False)
    else:
        print("[CONSIST] WARNING: no shift events detected in any file.", file=sys.stderr)

    print("[OK] CONSIST written to", os.path.abspath(args.out_dir))
    for p in [sug_up_path, sug_dn_path, delta_up_path, delta_dn_path]:
        print("  -", p)
    print("  -", os.path.join(args.out_dir, "CONSIST__DEBUG_SUMMARY.csv"))

if __name__ == "__main__":
    main()
