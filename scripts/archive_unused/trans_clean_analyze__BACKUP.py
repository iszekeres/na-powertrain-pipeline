#!/usr/bin/env python3
# NA Powertrain Tuning — trans_clean_analyze.py (STRICT v6)
# - No PID-based gear rename; assumes headers already standardized to: gear_actual, gear_cmd
# - Strict preflight: fails with explicit list if required headers are missing
# - time_s derived from offset; brake gate from switch/pressure; robust TCC slip; soft-lock inject
# - Writes the 4 standard outputs under .\06_Logs\Trans_Review

import os, re, glob, argparse, datetime as _dt
import pandas as pd, numpy as np
from tcc_softlock_injector import inject_tcc_softlock

GEAR_RATIO = {1:4.03, 2:2.36, 3:1.53, 4:1.15, 5:0.85, 6:0.67}

FOCUSED_KEEP = [
    "time_s","offset","speed_mph","throttle_pct","pedal_pct",
    "gear_actual","gear_cmd",
    "engine_rpm","turbine_rpm","output_rpm",
    "tcc_slip","tcc_locked","tcc_desired",
    "trans_temp_f","trans_temp_c",
    "MAP","BARO","brake","brake_raw",
    "yaw_rate_dps","steering_deg","lat_g",
    "PCS1","PCS2","PCS3","PCS4","PCS5","fill_cmd","oncoming",
]

CANONICAL = {
    "Vehicle Speed":"speed_mph","Vehicle Speed (SAE)":"speed_mph","Speed (MPH)":"speed_mph","VSS":"speed_mph",
    "Trans Output Shaft RPM":"output_rpm","Output Shaft RPM":"output_rpm","Output Shaft Speed":"output_rpm","OSS":"output_rpm",
    "Engine RPM":"engine_rpm","Engine RPM (SAE)":"engine_rpm","Engine Speed":"engine_rpm","Engine Speed (SAE)":"engine_rpm","RPM":"engine_rpm","RPM (SAE)":"engine_rpm",
    "Turbine Speed":"turbine_rpm","Turbine RPM":"turbine_rpm","Trans Input Shaft RPM":"turbine_rpm","Input Shaft Speed":"turbine_rpm","ISS":"turbine_rpm","Trans Turbine Speed":"turbine_rpm",
    "Throttle Position":"throttle_pct","Throttle Position (SAE)":"throttle_pct",
    "Accelerator Pedal Position":"pedal_pct","Throttle Desired Position":"pedal_pct",
    # (no gear aliases here — we require gear_actual/gear_cmd already present)
    "TCC Slip":"tcc_slip","TCC Desired Slip":"tcc_desired","TCC Commanded Slip":"tcc_desired",
    "Trans Fluid Temperature":"trans_temp_f","Transmission Fluid Temp":"trans_temp_f","Transmission Fluid Temperature":"trans_temp_f","Trans Fluid Temp":"trans_temp_f","Trans Temp":"trans_temp_f",
    "MAP":"MAP","Manifold Absolute Pressure":"MAP","BARO":"BARO",
    "Brake Switch":"brake_raw","Brake Pressure":"brake_raw","Brake":"brake_raw",
    "Yaw Rate":"yaw_rate_dps","Yaw Rate (SAE)":"yaw_rate_dps",
    "Lateral Acceleration":"lat_g","Lateral Accel":"lat_g","Lat G":"lat_g",
    "Steering Wheel Position":"steering_deg","Steering Wheel Angle":"steering_deg","Steering Angle":"steering_deg",
    "PCS 1":"PCS1","PCS 2":"PCS2","PCS 3":"PCS3","PCS 4":"PCS4","PCS 5":"PCS5",
    "Oncoming Clutch":"oncoming","Fill Command":"fill_cmd","Fill Pressure Cmd":"fill_cmd",
}

ALT = {  # lowercase search helpers (do not include any gear keys)
    "speed":"speed_mph","mph":"speed_mph",
    "oss":"output_rpm","output shaft":"output_rpm",
    "engine rpm":"engine_rpm","engine speed":"engine_rpm","rpm":"engine_rpm",
    "turbine":"turbine_rpm","input shaft":"turbine_rpm","iss":"turbine_rpm",
    "throttle position":"throttle_pct",
    "accelerator pedal position":"pedal_pct","throttle desired position":"pedal_pct",
    "tcc slip":"tcc_slip","tcc desired":"tcc_desired",
    "trans fluid temperature":"trans_temp_f","transmission fluid temp":"trans_temp_f","trans temp":"trans_temp_f",
    "manifold absolute pressure":"MAP","baro":"BARO",
    "brake":"brake_raw","brake pressure":"brake_raw","brake switch":"brake_raw",
    "yaw":"yaw_rate_dps","yaw rate":"yaw_rate_dps",
    "lateral accel":"lat_g","lateral acceleration":"lat_g","lat g":"lat_g",
    "steering":"steering_deg","steering wheel":"steering_deg",
}

def _ts_from_filename(path):
    b = os.path.basename(path)
    m = re.search(r"__(\d{8})__(\d{6})", b)
    if m:
        d, t = m.group(1), m.group(2)
        try: return _dt.datetime.strptime(d + t, "%Y%m%d%H%M%S")
        except Exception: pass
    return _dt.datetime.now()

def _reheader_if_pid_firstrow(df):
    if df is None or df.empty: return df, {}
    cols = [str(c) for c in df.columns]
    pid_by_index = {}
    if all(c.isdigit() for c in cols):
        pid_by_index = {i: cols[i] for i in range(len(cols))}
        first = df.iloc[0].astype(str)
        if bool(first.str.contains(r"[A-Za-z]", regex=True).any()):
            out = df.iloc[1:].reset_index(drop=True).copy()
            out.columns = first.values
            return out, pid_by_index
    return df, pid_by_index

def _map_columns(df):
    mapping = []; df2 = df.copy()
    for src, canon in CANONICAL.items():
        if src in df2.columns:
            df2 = df2.rename(columns={src: canon}); mapping.append((src, canon))
    lower = {c.lower(): c for c in df2.columns}
    for alt, canon in ALT.items():
        if canon not in df2.columns:
            for k, actual in lower.items():
                if alt in k:
                    df2 = df2.rename(columns={actual: canon}); mapping.append((actual, canon)); break
    return df2, mapping

def _first_series(obj, df=None):
    if df is not None and isinstance(obj, (str, int)):
        try: obj = df[obj]
        except Exception: return pd.Series(np.nan, index=df.index, dtype="float64")
    if isinstance(obj, pd.Series): return pd.to_numeric(obj, errors="coerce")
    if hasattr(obj, "apply"):
        num = obj.apply(pd.to_numeric, errors="coerce")
        return num.bfill(axis=1).iloc[:, 0]
    idx = df.index if df is not None else None
    return pd.Series(np.nan, index=idx, dtype="float64")

def _ensure_offset(df):
    if "offset" in df.columns: return df
    for tname in ["Time","Time (s)","Time(s)","Time [s]","Seconds","time","Offset"]:
        if tname in df.columns: return df.rename(columns={tname:"offset"})
    out = df.copy(); out["offset"] = np.arange(len(out)) / 10.0
    return out

def _convert_temp(df):
    if "trans_temp_f" in df.columns:
        sf = _first_series("trans_temp_f", df=df)
        df["trans_temp_f"] = sf; df["trans_temp_c"] = (sf - 32.0) * (5.0/9.0)
    elif "trans_temp_c" in df.columns:
        sc = _first_series("trans_temp_c", df=df)
        df["trans_temp_c"] = sc; df["trans_temp_f"] = sc * 9.0/5.0 + 32.0
    return df

def _derive_brake(df):
    if "brake_raw" in df.columns:
        s = _first_series("brake_raw", df=df)
        unique_nonnull = pd.unique(s.dropna())
        try: digital_set = set(map(float, unique_nonnull))
        except Exception: digital_set = set()
        if len(unique_nonnull) <= 4 and digital_set.issubset({0.0, 1.0}):
            df["brake"] = (s > 0.5).astype(int)
        else:
            sn = s.clip(lower=0)
            if sn.notna().any():
                baseline = sn[sn <= sn.quantile(0.2)]
                mu = float(baseline.median()) if not baseline.empty else float(sn.median())
                sigma = float(baseline.std()) if not baseline.empty else float(sn.std())
            else:
                mu = 0.0; sigma = 0.0
            thr = max(mu + 3*sigma, 50.0)
            if sn.max() <= 10: thr = max(mu + 3*sigma, 2.0)
            df["brake"] = (sn >= thr).astype(int)
    else:
        df["brake"] = 0
    return df

def _thr_from_temp_f(tf):
    if pd.isna(tf): return 25.0
    if tf < 100.0:  return 34.0
    if tf <= 180.0: return 22.0
    return 26.0

def _robust_tcc_slip(df):
    direct = _first_series("tcc_slip", df=df) if "tcc_slip" in df.columns else None
    iss = None
    if "engine_rpm" in df.columns and "turbine_rpm" in df.columns:
        iss = _first_series("engine_rpm", df=df) - _first_series("turbine_rpm", df=df)
    oss = None
    if "engine_rpm" in df.columns and "output_rpm" in df.columns and "gear_actual" in df.columns:
        out = _first_series("output_rpm", df=df)
        ga  = _first_series("gear_actual", df=df).round().clip(1,6).astype("Int64")
        ratio = ga.map(pd.Series(GEAR_RATIO)).astype(float)
        turb_from_oss = out * ratio
        oss = _first_series("engine_rpm", df=df) - turb_from_oss

    def quality_mask(base):
        if base is None: return None
        ok = pd.Series(True, index=base.index)
        if "speed_mph" in df.columns: ok &= (_first_series("speed_mph", df=df) > 20)
        if "brake" in df.columns:     ok &= (_first_series("brake", df=df) == 0)
        if "throttle_pct" in df.columns:
            thr = _first_series("throttle_pct", df=df).fillna(method="ffill")
            d = thr.diff().abs().fillna(0); ok &= (d <= 3.0)
        if "gear_actual" in df.columns:
            g = _first_series("gear_actual", df=df).ffill()
            ok &= (g.diff().fillna(0) == 0)
        return ok

    masks = {"direct": quality_mask(direct), "iss": quality_mask(iss), "oss": quality_mask(oss)}
    cand  = {"direct": direct, "iss": iss, "oss": oss}

    win = 25
    fused = pd.Series(np.nan, index=df.index, dtype="float64")
    for key in ["direct","iss","oss"]:
        s = cand[key]
        if s is None: continue
        m = masks[key]
        if m is None: continue
        s2 = s.where(m)
        rs = s2.rolling(win, min_periods=5).std()
        take = rs.notna() & (fused.isna())
        fused = fused.where(~take, s2)

    if direct is not None: fused = fused.fillna(direct)
    if iss is not None:    fused = fused.fillna(iss)
    if oss is not None:    fused = fused.fillna(oss)

    fused = fused.clip(-4000, 4000).rolling(win, min_periods=3).median()
    df["tcc_slip"] = fused

    tf = _first_series("trans_temp_f", df=df) if "trans_temp_f" in df.columns else pd.Series([np.nan]*len(df), index=df.index)
    thresholds = tf.apply(_thr_from_temp_f)
    slip_abs = fused.abs()

    lock_cond   = (slip_abs <= thresholds)
    unlock_cond = (slip_abs >= (thresholds + 8.0))

    state = False
    locked = np.zeros(len(df), dtype=np.int32)
    for i in range(len(df)):
        if unlock_cond.iloc[i]: state = False
        elif lock_cond.iloc[i]: state = True
        locked[i] = 1 if state else 0
    df["tcc_locked"] = pd.Series(locked, index=df.index, dtype="int32")
    return df

def _collapse_focused(df, keep_cols):
    out = pd.DataFrame(index=df.index)
    for c in keep_cols:
        if c in df.columns:
            out[c] = _first_series(c, df=df)
    return out

def _detect_shift_events(df):
    if "gear_actual" not in df.columns:
        return pd.DataFrame(columns=["from","to","index","offset","speed_mph","engine_rpm","throttle_pct","pedal_pct"])
    gear = _first_series("gear_actual", df=df).ffill().astype(float)
    changes = gear.diff().fillna(0.0).ne(0.0)
    idxs = np.flatnonzero(changes.values)
    rows = []
    for i in idxs:
        frm = gear.iloc[i-1] if i > 0 else np.nan
        to  = gear.iloc[i]
        row = {"from": int(frm) if pd.notna(frm) else None, "to": int(to) if pd.notna(to) else None, "index": int(i)}
        for k in ["offset","speed_mph","engine_rpm","throttle_pct","pedal_pct"]:
            if k in df.columns:
                try: val = _first_series(k, df=df).iloc[i]
                except Exception: val = np.nan
                row[k] = float(val) if pd.notna(val) else np.nan
        rows.append(row)
    se = pd.DataFrame(rows)
    if not se.empty:
        se = se[(se["from"].fillna(0) >= 0) & (se["to"].fillna(0) >= 0)]
    return se

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--in", dest="in_path", default=None, help="single CSV file or glob")
    ap.add_argument("--in-glob", default=r".\newlogs\*.csv", help="glob pattern if -i not provided")
    ap.add_argument("-o","--out-dir", default=r".\06_Logs\Trans_Review")
    ap.add_argument("--speed-ffill-sec", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve input files
    if args.in_path:
        files = glob.glob(args.in_path) if any(x in args.in_path for x in "*?[]") else [args.in_path]
    else:
        files = sorted(glob.glob(args.in_glob))
    if not files:
        print("[INFO] No input CSVs — nothing to do."); return

    for path in files:
        try:
            raw = pd.read_csv(path, encoding="utf-8-sig", engine="python", on_bad_lines="skip")
        except Exception as e:
            print(f"[WARN] Could not read {os.path.basename(path)}: {e}"); continue

        # Header fix if first row had names, then canonical remap
        after, _ = _reheader_if_pid_firstrow(raw)
        if after is None or after.empty:
            print(f"[WARN] Empty after reheader: {os.path.basename(path)}"); continue
        full_df = after.copy()
        mapped, mapping = _map_columns(after)

        # Strict preflight (no fallbacks): require gear headers you standardized
        required = ["gear_actual","gear_cmd"]
        missing = [c for c in required if c not in mapped.columns]
        if missing:
            raise SystemExit("Missing headers: " + ", ".join(missing))

        # Basic derivations
        mapped = _ensure_offset(mapped)
        if "time_s" not in mapped.columns and "offset" in mapped.columns:
            mapped["time_s"] = mapped["offset"]
        mapped = _convert_temp(mapped)
        mapped = _derive_brake(mapped)
        mapped = _robust_tcc_slip(mapped)

        clean_df = _collapse_focused(mapped, FOCUSED_KEEP)
        if "speed_mph" in clean_df.columns:
            limit = max(1, int(args.speed_ffill_sec * 10))  # assume ~10 Hz logs
            clean_df["speed_mph"] = clean_df["speed_mph"].ffill(limit=limit)

        # Outputs (per file)
        tag = os.path.splitext(os.path.basename(path))[0]
        ts  = _ts_from_filename(path); stamp = ts.strftime("%Y%m%d__%H%M%S")

        out_clean      = os.path.join(args.out_dir, f"__trans_focus__clean__{tag}__{stamp}.csv")
        out_clean_full = os.path.join(args.out_dir, f"__trans_focus__clean_FULL__{tag}__{stamp}.csv")
        out_shift      = os.path.join(args.out_dir, f"__trans_focus__shift_events__{tag}__{stamp}.csv")
        out_map        = os.path.join(args.out_dir, f"__trans_focus__mapping__{tag}__{stamp}.csv")
        out_sum        = os.path.join(args.out_dir, f"__trans_focus__summary__{tag}__{stamp}.txt")

        clean_df = inject_tcc_softlock(clean_df)
        clean_df.to_csv(out_clean, index=False)

        # Append canonical/derived into FULL passthrough where missing
        for cname in FOCUSED_KEEP:
            if cname in clean_df.columns and cname not in full_df.columns:
                full_df[cname] = clean_df[cname]
        full_df.to_csv(out_clean_full, index=False)

        # Shift events / mapping / summary
        se = _detect_shift_events(clean_df); se.to_csv(out_shift, index=False)
        pd.DataFrame([{"source": s, "canonical": c} for (s,c) in mapping]).to_csv(out_map, index=False)

        nn_parts = []
        for c in clean_df.columns:
            try: cnt = int(pd.to_numeric(clean_df[c], errors="coerce").notna().sum())
            except Exception: cnt = int(clean_df[c].notna().sum()) if hasattr(clean_df[c], "notna") else 0
            nn_parts.append(f"{c}:{cnt}")

        lines = []
        lines.append(f"input_file: {os.path.basename(path)}")
        lines.append(f"clean_rows: {len(clean_df)}  columns: {', '.join(clean_df.columns)}")
        lines.append("non-null counts -> " + ", ".join(nn_parts))
        if not se.empty:
            grp = se.groupby(["from","to"]).size().reset_index(name="count").sort_values("count", ascending=False)
            lines.append("shift pair counts:")
            for _, r in grp.iterrows():
                lines.append(f"  {int(r['from']):>4} -> {int(r['to']):<4}  {int(r['count'])}")
        with open(out_sum, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"[OK] {os.path.basename(path)} → {os.path.basename(out_clean)} (and FULL, shifts, map, summary)")

if __name__ == "__main__":
    main()
