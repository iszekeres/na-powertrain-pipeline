#!/usr/bin/env python3
# NA Powertrain Tuning - Cleaner & Analyzer (FIX)
# Same as prior, but with properly escaped backslashes in the "no raw CSVs" message.

import os, re, glob, argparse, datetime as _dt
import pandas as pd
import numpy as np

CANONICAL = {
    "Vehicle Speed": "speed_mph",
    "Vehicle Speed (SAE)": "speed_mph",
    "Speed (MPH)": "speed_mph",
    "VSS": "speed_mph",
    "Output Shaft Speed": "output_rpm",
    "Output Shaft RPM": "output_rpm",
    "Engine RPM": "engine_rpm",
    "Engine Speed": "engine_rpm",
    "Turbine Speed": "turbine_rpm",
    "Turbine RPM": "turbine_rpm",
    "Trans Turbine Speed": "turbine_rpm",
    "Trans Input Shaft RPM": "turbine_rpm",
    "Throttle Position": "throttle_pct",
    "Throttle Position (SAE)": "throttle_pct",
    "Accelerator Pedal Position": "pedal_pct",
    "Accelerator Pedal Position (SAE)": "pedal_pct",
    "Trans Current Gear": "gear_actual",
    "Trans Current Gear.1": "gear_cmd",
    "TCC Slip": "tcc_slip",
    "TCC Desired Slip": "tcc_desired",
    "TCC Commanded Slip": "tcc_desired",
    "Trans Fluid Temperature": "trans_temp_c",
    "Transmission Fluid Temp": "trans_temp_c",
    "MAP": "MAP",
    "Manifold Absolute Pressure": "MAP",
    "BARO": "BARO",
    "Brake Switch": "brake",
    "PCS 1": "PCS1",
    "PCS 2": "PCS2",
    "PCS 3": "PCS3",
    "PCS 4": "PCS4",
    "PCS 5": "PCS5",
    "Oncoming Clutch": "oncoming",
    "Fill Command": "fill_cmd",
}

ALT_MAP = {
    "speed": "speed_mph",
    "mph": "speed_mph",
    "oss": "output_rpm",
    "rpm": "engine_rpm",
    "eng_rpm": "engine_rpm",
    "turbine": "turbine_rpm",
    "turb_rpm": "turbine_rpm",
    "gear": "gear_actual",
    "gear commanded": "gear_cmd",
    "tcc slip": "tcc_slip",
    "tcc desired": "tcc_desired",
    "trans temp": "trans_temp_c",
}

def _ts_from_filename(path):
    b = os.path.basename(path)
    m = re.search(r"__(\d{8})__(\d{6})", b)
    if m:
        d, t = m.group(1), m.group(2)
        try:
            return _dt.datetime.strptime(d + t, "%Y%m%d%H%M%S")
        except Exception:
            pass
    return _dt.datetime.now()

def _reheader_if_pid_firstrow(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols = [str(c) for c in df.columns]
    all_numeric = all(c.isdigit() for c in cols)
    first = df.iloc[0].astype(str) if len(df) else None
    has_alpha = bool(first.str.contains(r"[A-Za-z]", regex=True).any()) if first is not None else False
    if all_numeric and has_alpha:
        out = df.iloc[1:].reset_index(drop=True).copy()
        out.columns = first.values
        return out
    return df

def _map_columns(df: pd.DataFrame):
    mapping = []
    df2 = df.copy()
    for src, canon in CANONICAL.items():
        if src in df2.columns and canon not in df2.columns:
            df2 = df2.rename(columns={src: canon})
            mapping.append((src, canon))
    lower_cols = {c.lower(): c for c in df2.columns}
    for alt, canon in ALT_MAP.items():
        if alt in lower_cols and canon not in df2.columns:
            src_name = lower_cols[alt]
            df2 = df2.rename(columns={src_name: canon})
            mapping.append((src_name, canon))
    return df2, mapping

def _compute_tcc_slip(df: pd.DataFrame) -> pd.DataFrame:
    if "tcc_slip" in df.columns and not df["tcc_slip"].isna().all():
        return df
    if "engine_rpm" in df.columns and "turbine_rpm" in df.columns:
        e = pd.to_numeric(df["engine_rpm"], errors="coerce")
        t = pd.to_numeric(df["turbine_rpm"], errors="coerce")
        df["tcc_slip"] = e - t
    return df

def _clean_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _detect_shift_events(df: pd.DataFrame) -> pd.DataFrame:
    if "gear_actual" not in df.columns:
        return pd.DataFrame(columns=["from","to","index","offset","speed_mph","engine_rpm","throttle_pct","pedal_pct"])
    gear = pd.to_numeric(df["gear_actual"], errors="coerce").fillna(method="ffill").astype(float)
    changes = gear.diff().fillna(0.0).ne(0.0)
    idxs = np.flatnonzero(changes.values)
    rows = []
    for i in idxs:
        frm = gear.iloc[i-1] if i > 0 else np.nan
        to  = gear.iloc[i]
        row = {"from": int(frm) if pd.notna(frm) else None, "to": int(to) if pd.notna(to) else None, "index": int(i)}
        for k in ["offset","speed_mph","engine_rpm","throttle_pct","pedal_pct"]:
            if k in df.columns:
                try:
                    val = pd.to_numeric(df[k], errors="coerce").iloc[i]
                except Exception:
                    val = np.nan
                row[k] = float(val) if pd.notna(val) else np.nan
        rows.append(row)
    se = pd.DataFrame(rows)
    if not se.empty:
        se = se[(se["from"].fillna(0) >= 0) & (se["to"].fillna(0) >= 0)]
    return se

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-glob", default=r".\newlogs\*.csv")
    ap.add_argument("--out-dir", default=r".\06_Logs\Trans_Review")
    ap.add_argument("--final-drive", type=float, default=3.08)
    ap.add_argument("--speed-ffill-sec", type=float, default=0.5, help="short forward-fill for speed_mph (metrics only)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(args.in_glob))
    if not files:
        print("[INFO] No raw CSVs in .\newlogs\ - nothing to do.")
        return

    for path in files:
        try:
            raw = pd.read_csv(path, encoding="utf-8-sig", engine="python", on_bad_lines="skip")
        except Exception as e:
            print(f"[WARN] Could not read {os.path.basename(path)}: {e}")
            continue

        raw = _reheader_if_pid_firstrow(raw)
        mapped, mapping = _map_columns(raw)

        keep_cols = ["offset","speed_mph","throttle_pct","pedal_pct","gear_actual","gear_cmd",
                     "engine_rpm","turbine_rpm","output_rpm","tcc_slip","tcc_desired",
                     "trans_temp_c","MAP","BARO","brake",
                     "PCS1","PCS2","PCS3","PCS4","PCS5","fill_cmd","oncoming"]

        if "offset" not in mapped.columns:
            for tname in ["Time","Time (s)","Time(s)","Time [s]","Seconds","time"]:
                if tname in mapped.columns:
                    mapped = mapped.rename(columns={tname: "offset"})
                    mapping.append((tname,"offset"))
                    break

        mapped = _compute_tcc_slip(mapped)
        _clean_numeric(mapped, [c for c in keep_cols if c in mapped.columns])
        if "speed_mph" in mapped.columns:
            mapped["speed_mph"] = mapped["speed_mph"].ffill(limit=5)

        tag = os.path.splitext(os.path.basename(path))[0]
        ts  = _ts_from_filename(path)
        stamp = ts.strftime("%Y%m%d__%H%M%S")
        clean_name = f"__trans_focus__clean__{tag}__{stamp}.csv"
        shift_name = f"__trans_focus__shift_events__{tag}__{stamp}.csv"
        map_name   = f"__trans_focus__mapping__{tag}__{stamp}.csv"
        sum_name   = f"__trans_focus__summary__{tag}__{stamp}.txt"

        clean_df = mapped[[c for c in keep_cols if c in mapped.columns]].copy()
        clean_df.to_csv(os.path.join(args.out_dir, clean_name), index=False)

        se = _detect_shift_events(clean_df)
        se.to_csv(os.path.join(args.out_dir, shift_name), index=False)

        map_rows = [{"source": s, "canonical": c} for (s,c) in mapping]
        pd.DataFrame(map_rows).to_csv(os.path.join(args.out_dir, map_name), index=False)

        lines = []
        lines.append(f"input_file: {os.path.basename(path)}")
        lines.append(f"clean_rows: {len(clean_df)}  columns: {', '.join(clean_df.columns)}")
        lines.append("non-null counts -> " + ", ".join([f"{c}:{int(clean_df[c].notna().sum())}" for c in clean_df.columns]))
        if not se.empty:
            grp = se.groupby(["from","to"]).size().reset_index(name="count").sort_values("count", ascending=False)
            lines.append("shift pair counts:")
            for _, r in grp.iterrows():
                lines.append(f"  {int(r['from']):>4} -> {int(r['to']):<4}  {int(r['count'])}")
        with open(os.path.join(args.out_dir, sum_name), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"[OK] Wrote clean + shift + mapping + summary for {os.path.basename(path)}")

if __name__ == "__main__":
    main()
