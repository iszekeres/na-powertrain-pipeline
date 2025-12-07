# -*- coding: utf-8 -*-
"""
run_strict_pass.py  — python-only strict runner (avoids PowerShell policies)

Examples:
  python run_strict_pass.py --cmd "python corner_exit_pass_weighted__shim.py --logs-glob .\\newlogs\\cleaned\\__trans_focus__clean_FULL__*withbrake*.csv --out .\\newlogs\\output\\02_passes\\CORNER\\CORNER__SHIFT_DOWN__DELTA.tsv --min-speed 8 --max-speed 30 --thr-rate 16 --min-score 50 --delta-mph 0.3" --out-file .\\newlogs\\output\\02_passes\\CORNER\\CORNER__SHIFT_DOWN__DELTA.tsv --require-nonzero

  python run_strict_pass.py --cmd "python corner_exit_pass_weighted__chassis_shim.py --logs-glob ... --out ... --lat-g 0.12 --yaw-rate 18 --steer-abs 65 --steer-rate 30 --steer-column \"Steering Wheel Position\" --latg-column \"Lateral Acceleration\" --yaw-column \"Yaw Rate\" --min-score 55 --delta-mph 0.3" --out-file .\\newlogs\\output\\02_passes\\CORNER\\CORNER__SHIFT_DOWN__DELTA__CHASSIS.tsv --require-nonzero --enforce-down-nonpositive --clamp
"""
import argparse, subprocess, os, sys, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--cmd", required=True, help="Command line to execute (quoted string).")
ap.add_argument("--out-file", required=True, help="Expected output path to validate.")
ap.add_argument("--require-nonzero", action="store_true", help="Fail if the TSV contains all zeros.")
ap.add_argument("--enforce-down-nonpositive", action="store_true", help="Ensure all cells <= 0 (use for DOWN delta files).")
ap.add_argument("--clamp", action="store_true", help="If enforcing nonpositive, clamp positives to zero instead of failing.")
args = ap.parse_args()

print("[strict] running:", args.cmd)
ret = subprocess.run(args.cmd, shell=True)
if ret.returncode != 0:
  print("[strict] command failed with exit code %d" % ret.returncode, file=sys.stderr)
  sys.exit(ret.returncode)

if not os.path.exists(args.out_file):
  print("[strict] expected output missing: %s" % args.out_file, file=sys.stderr)
  sys.exit(4)

if args.require_nonzero:
  df = pd.read_csv(args.out_file, sep="\t")
  cols = df.columns[1:-1] if df.columns[-1] == "%" else df.columns[1:]
  n = (df[cols].apply(pd.to_numeric, errors="coerce") != 0).to_numpy().sum()
  if n <= 0:
    print("[strict] output is all zeros:", args.out_file, file=sys.stderr)
    sys.exit(5)

if args.enforce_down_nonpositive:
  cmd = 'python validate_down_deltas_strict.py "%s"%s' % (args.out_file, " --clamp" if args.clamp else "")
  ret = subprocess.run(cmd, shell=True)
  if ret.returncode != 0:
    print("[strict] positive cells found in DOWN delta file.", file=sys.stderr)
    sys.exit(6)

print("strict checks passed for %s" % os.path.basename(args.out_file))
