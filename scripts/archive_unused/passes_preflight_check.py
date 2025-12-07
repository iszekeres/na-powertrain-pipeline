import os, glob, pandas as pd, sys
clean = r".\newlogs\cleaned"
files = sorted(glob.glob(os.path.join(clean, "__trans_focus__clean_FULL__*.csv")), key=os.path.getmtime)
if not files: 
    print("[FAIL] no CLEAN_FULL csvs found", file=sys.stderr); sys.exit(2)
p = files[-1]
df = pd.read_csv(p, nrows=200000)
need = ["speed_mph__canon","time_s__canon","throttle_pct__canon","gear_actual__canon",
        "engine_rpm__canon","turbine_rpm__canon","brake__canon","tftF__canon","ectF__canon",
        "gear_cmd__canon","tcc_slip_fused__canon"]
missing=[c for c in need if c not in df.columns]
if missing:
    print("[FAIL] missing required __canon columns:", ", ".join(missing), file=sys.stderr); sys.exit(3)
print("[OK] CLEAN_FULL preflight: required columns present")
