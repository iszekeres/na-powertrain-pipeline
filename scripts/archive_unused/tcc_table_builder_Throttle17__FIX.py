
# tcc_table_builder_Throttle17__FIX.py ? STRICT RAW VERSION
# Uses ONLY these RAW headers from CLEAN_FULL (no aliasing, no canon, no derived slip):
#   Offset, Vehicle Speed (SAE), Throttle Position, gear_actual, Brake Pressure,
#   Engine RPM (SAE), Trans Turbine RPM, TCC Slip
# Optional: Trans Fluid Temp (for warm gate). Skips non-CLEAN_FULL CSVs.
# TPS bin band = ?3.0%. 0.1 mph outputs. RELEASE ? APPLY + 1.1 mph. Progress to stderr.

import os, sys, argparse
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

# TPS sampling window center bias (percent)
TPS_BIAS = 1.0

REQ_RAW = [
    "Offset","Vehicle Speed (SAE)","Throttle Position","gear_actual","Brake Pressure",
    "Engine RPM (SAE)","Trans Turbine RPM","TCC Slip"
]
OPT_RAW = ["Trans Fluid Temp"]

def round_1dp_preserve(x):
    if pd.isna(x): return x
    try:
        xi = int(x)
        if xi in (317,318): return xi
    except Exception:
        pass
    return float(Decimal(str(x)).quantize(Decimal("0.0"), rounding=ROUND_HALF_UP))

def to_fahrenheit(series):
    if series is None or series.dropna().empty:
        return pd.Series(np.nan, index=series.index if series is not None else None)
    s = pd.to_numeric(series, errors="coerce")
    med = s.dropna().median()
    return s*9/5 + 32 if med < 80 else s

def build_softlock(slip, tftF, brake01, speed_mph, gear, time_s):
    warm = (tftF >= 100) if (tftF is not None and not pd.to_numeric(tftF, errors="coerce").dropna().empty) else pd.Series(True, index=slip.index)
    moving = pd.to_numeric(speed_mph, errors="coerce") >= 25
    gearok = pd.to_numeric(gear, errors="coerce") >= 3
    brake0 = (pd.to_numeric(brake01, errors="coerce") == 0)
    ok = warm & moving & gearok & brake0 & pd.to_numeric(slip, errors="coerce").notna()

    ts = pd.to_numeric(time_s, errors="coerce")
    if ts.dropna().shape[0] > 1:
        dt = np.diff(ts.dropna().values)
        med_dt = float(np.median(np.clip(dt, 1e-3, None)))
    else:
        med_dt = 0.01
    win_on  = max(1, int(round(0.6 / med_dt)))
    win_off = max(1, int(round(0.4 / med_dt)))

    slipv = pd.to_numeric(slip, errors="coerce")
    on  = (slipv.abs() <= 30).rolling(win_on,  min_periods=win_on).apply(lambda a: bool(np.all(a)), raw=False).astype(bool)
    off = (slipv.abs() >= 80).rolling(win_off, min_periods=win_off).apply(lambda a: bool(np.all(a)), raw=False).astype(bool)

    locked = pd.Series(False, index=slip.index, name="tcc_locked_built")
    state = False
    for i in range(len(slip)):
        if not ok.iat[i]:
            locked.iat[i] = state
            continue
        if not state and on.iat[i]:  state = True
        if  state and off.iat[i]:    state = False
        locked.iat[i] = bool(state)
    return locked

def build_tables(df_all: pd.DataFrame, out_dir: str, band: float = 3.0):
    tps_axis = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
    df_all["speed_mph"]    = df_all["speed_mph"].map(round_1dp_preserve)
    df_all["throttle_pct"] = pd.to_numeric(df_all["throttle_pct"], errors="coerce")
    df_all["gear_actual"]  = pd.to_numeric(df_all["gear_actual"],  errors="coerce")

    def grid(kind: str):
        rows = []
        for g in (3,4,5,6):
            row = []
            gsel = (df_all["gear_actual"] == g)
            for t in tps_axis:
                sel = gsel & df_all["throttle_pct"].between((t+TPS_BIAS)-band, (t+TPS_BIAS)+band)
                if not sel.any():
                    row.append(np.nan); continue
                mph = df_all.loc[sel, "speed_mph"]
                locked = df_all.loc[sel, "tcc_locked_built"]
                if kind == "apply":
                    val = mph[locked].quantile(0.35) if locked.any() else np.nan
                else:
                    un = ~locked
                    val = mph[un].quantile(0.65) if un.any() else np.nan
                row.append(round_1dp_preserve(val) if pd.notna(val) else np.nan)
            rows.append((f"{g}th {'Apply' if kind=='apply' else 'Release'}", row))
        return rows

    apply_rows   = grid("apply")
    release_rows = grid("release")

    # RELEASE ? APPLY + 1.1 mph
    for i in range(len(apply_rows)):
        for j in range(len(apply_rows[i][1])):
            a = apply_rows[i][1][j]; r = release_rows[i][1][j]
            if pd.notna(a) and pd.notna(r) and r < a + 1.1:
                release_rows[i][1][j] = round_1dp_preserve(a + 1.1)

    ap_n = sum(sum(0 if pd.isna(v) else 1 for v in row) for _, row in apply_rows)
    rl_n = sum(sum(0 if pd.isna(v) else 1 for v in row) for _, row in release_rows)
    sys.stderr.write("[tcc_builder] grid coverage: APPLY=" + str(ap_n) + "/68, RELEASE=" + str(rl_n) + "/68\\n")

    hdr = "mph\\t" + "\\t".join(map(str, tps_axis)) + "\\t%"
    def write(rows, name):
        path = os.path.join(out_dir, f"{name}__Throttle17.tsv")
        with open(path, "w", newline="") as f:
            f.write(hdr + "\\n")
            for label, vals in rows:
                vals = [("" if pd.isna(v) else str(round_1dp_preserve(v))) for v in vals]
                f.write(label + "\\t" + "\\t".join(vals) + "\\t\\n")
        return path

    ap = write(apply_rows, "TCC_APPLY")
    rl = write(release_rows, "TCC_RELEASE")
    return ap, rl

def main(clean_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csvs = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.lower().endswith(".csv")]
    if not csvs: raise SystemExit("No CSV files found")
    # only CLEAN_FULL
    csvs_cf = [c for c in csvs if "clean_full" in os.path.basename(c).lower()]
    # de-dup pairs; prefer __withbrake variant
    bykey = {}
    for c in csvs_cf:
        base = os.path.basename(c)
        key = base.lower().replace("__withbrake","")
        bykey.setdefault(key, []).append(c)
    chosen, skipped = [], []
    for key, files in bykey.items():
        if len(files) == 1:
            chosen.append(files[0])
        else:
            wb = [f for f in files if "__withbrake" in os.path.basename(f).lower()]
            pick = wb[0] if wb else files[0]
            chosen.append(pick)
            for f in files:
                if f != pick: skipped.append(f)
    import sys
    for k in skipped:
        sys.stderr.write("[tcc_builder] de-dup: skipping " + os.path.basename(k) + " (duplicate)\\n")
    csvs_cf = chosen
    for k in (set(csvs) - set(csvs_cf)):
        sys.stderr.write("[tcc_builder] skipping non-CLEAN_FULL: " + os.path.basename(k) + "\\n")
    if not csvs_cf: raise SystemExit("No CLEAN_FULL CSVs found")

    frames = []
    for p in csvs_cf:
        hdr = pd.read_csv(p, nrows=0).columns.tolist()
        missing = [c for c in REQ_RAW if c not in hdr]
        if missing:
            raise SystemExit("Required RAW headers missing in " + os.path.basename(p) + ": " + ", ".join(missing))
        usecols = REQ_RAW + [c for c in OPT_RAW if c in hdr]

        sys.stderr.write("[tcc_builder] processing " + os.path.basename(p) + "\\n")
        df = pd.read_csv(p, low_memory=False, usecols=usecols)

        part = pd.DataFrame({
            "speed_mph":   pd.to_numeric(df["Vehicle Speed (SAE)"], errors="coerce"),
            "throttle_pct":pd.to_numeric(df["Throttle Position"],    errors="coerce"),
            "gear_actual": pd.to_numeric(df["gear_actual"],          errors="coerce"),
            "time_s":      pd.to_numeric(df["Offset"],               errors="coerce"),
            "brake01":     (pd.to_numeric(df["Brake Pressure"],      errors="coerce").fillna(0) >= 15).astype(int),
            "slip":        pd.to_numeric(df["TCC Slip"],             errors="coerce"),
        })

        tftF = None
        if "Trans Fluid Temp" in df.columns:
            tftF = to_fahrenheit(df["Trans Fluid Temp"])
        part["tftF"] = tftF if (tftF is not None) else float("nan")

        part = part.sort_values("time_s", kind="mergesort").reset_index(drop=True)

        locked = build_softlock(
            part["slip"], part["tftF"], part["brake01"],
            part["speed_mph"], part["gear_actual"], part["time_s"]
        )
        part["tcc_locked_built"] = locked

        gcol = pd.to_numeric(part["gear_actual"], errors="coerce")
        gear_counts = ", ".join("g"+str(g)+"="+str(int((gcol==g).sum())) for g in (1,2,3,4,5,6))
        sys.stderr.write("[tcc_builder]   rows=" + format(len(part), ",") + " locked=" + format(int(locked.sum()), ",") + "  " + gear_counts + "\\n")

        frames.append(part[["speed_mph","throttle_pct","gear_actual","time_s","tcc_locked_built"]])

    all_df = pd.concat(frames, ignore_index=True).dropna(subset=["speed_mph","throttle_pct","gear_actual","time_s"])
    ap, rl = build_tables(all_df, out_dir, band=3.0)
    print("Wrote:", ap); print("Wrote:", rl)
    sys.stderr.write("[tcc_builder] done. outputs written\\n")

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--clean-dir", required=True)
    a.add_argument("--out-dir",  required=True)
    args = a.parse_args()
    main(args.clean_dir, args.out_dir)
