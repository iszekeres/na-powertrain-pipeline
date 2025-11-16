#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# driver_intent_pass_weighted__PLUS_v3.py  (Patched shim)
#
# This is a drop-in replacement that fixes the "__NONE__" / no‑gates behavior
# without modifying your existing INTENT logic.
#
# HOW IT WORKS
# - Parses a minimal set of args we need to identify the input files and which
#   gates to disable. Everything else is passed straight through to the original
#   core script via subprocess.
# - If any of the following are disabled (explicitly via --no-... flags or by
#   passing the column name "__NONE__"), this shim creates temporary copies of
#   the CLEAN_FULL CSVs with those columns physically removed:
#     • TCC lock flag column (e.g., "tcc_locked_built__canon")
#     • Steering Wheel Position
#     • Lateral Acceleration
#     • Yaw Rate
# - Then it invokes the original script (renamed by you to
#   "driver_intent_pass_weighted__PLUS_v3__CORE.py") against those temp copies.
#   Because the columns are gone, the core script cannot accidentally re-enable
#   those gates.
#
# ONE-TIME SETUP
# Rename your current file to the CORE name:
#   driver_intent_pass_weighted__PLUS_v3.py  →  driver_intent_pass_weighted__PLUS_v3__CORE.py
# Put THIS patched file in its place as driver_intent_pass_weighted__PLUS_v3.py.
#
# USAGE (same as before, plus the convenience flags)
# Examples (PowerShell):
#
# Disable all chassis + TCC lock (“nogates” sanity)
# python .\driver_intent_pass_weighted__PLUS_v3.py `
#   --logs-glob ".\newlogs\cleaned\__trans_focus__clean_FULL__*withbrake*.csv" `
#   --out-dir   ".\newlogs\output\INTENT_TUNE3_NOGATES" `
#   --no-chassis --no-tcc-lock `
#   --thr-rate-pedal 12 --thr-rate-throttle 9
#
# Disable only TCC lock (keep chassis)
# python .\driver_intent_pass_weighted__PLUS_v3.py `
#   --logs-glob ".\newlogs\cleaned\__trans_focus__clean_FULL__*withbrake*.csv" `
#   --out-dir   ".\newlogs\output\INTENT_TUNE3_TCCONLY" `
#   --no-tcc-lock `
#   --thr-rate-pedal 12 --thr-rate-throttle 9
#
# Or disable by passing "__NONE__" for any column arg
#   --tcc-column "__NONE__" --steer-column "__NONE__" --latg-column "__NONE__" --yaw-column "__NONE__"
#
# Notes
# - Temp copies are written to a sibling "_NG_cache" folder next to your inputs.
# - This shim escapes percent signs in help strings to avoid argparse format errors.

import argparse, glob, os, re, shutil, subprocess, sys
from pathlib import Path

def _is_none(v):
    return (v is None) or (str(v).strip().upper() == "__NONE__")

def _glob_list(pat):
    return sorted(glob.glob(pat))

def _make_ng_copies(files, drop_tcc, drop_chassis):
    """Create 'no-gates' copies of CSVs by physically removing columns.
       Returns (ng_dir_str, ng_glob)"""
    if not files:
        raise RuntimeError("no files to copy")
    first_dir = Path(files[0]).resolve().parent
    ng_dir = first_dir / "_NG_cache"
    ng_dir.mkdir(parents=True, exist_ok=True)

    # Compile drop patterns
    pats = []
    if drop_tcc:
        pats.append(r"tcc.*locked.*built")
    if drop_chassis:
        pats.append(r"(steering\s*wheel\s*position|lateral\s*acceleration|yaw\s*rate)")
    drop_re = re.compile("|".join(pats), re.I) if pats else None

    # Do the work
    try:
        import pandas as pd
    except Exception as e:
        print("[INTENT shim] pandas is required in this phase.", file=sys.stderr); raise

    for src in files:
        dst = ng_dir / (Path(src).name.replace("withbrake", "withbrake__NG"))
        if drop_re is None:
            shutil.copy2(src, dst)
            print(f"[INTENT shim] copied {src} -> {dst}")
            continue
        df = pd.read_csv(src)  # full-memory path by user preference
        keep = [c for c in df.columns if not drop_re.search(c)]
        dropped = [c for c in df.columns if drop_re.search(c)]
        df[keep].to_csv(dst, index=False)
        print(f"[INTENT shim] wrote {dst.name}: kept {len(keep)} cols, dropped {len(dropped)}")
    return str(ng_dir), str(ng_dir / "*withbrake__NG*.csv")

def main():
    # Parse *known* args for gating; keep the rest to forward to the CORE script.
    ap = argparse.ArgumentParser(add_help=False)
    # Required pass-through
    ap.add_argument("--logs-glob", required=True)
    ap.add_argument("--out-dir",   required=True)
    # Column args we care about (escape % in help strings)
    ap.add_argument("--tcc-column", default=None, help="Name of TCC lock flag column; '__NONE__' to disable")
    ap.add_argument("--steer-column", default=None, help="Steering column; '__NONE__' to disable")
    ap.add_argument("--latg-column",  default=None, help="Lateral acceleration column; '__NONE__' to disable")
    ap.add_argument("--yaw-column",   default=None, help="Yaw rate column; '__NONE__' to disable")
    # Convenience switches
    ap.add_argument("--no-chassis",  action="store_true", help="Disable chassis gating (steer/lat/yaw)")
    ap.add_argument("--no-tcc-lock", action="store_true", help="Disable TCC lock gating")
    # Normal help
    ap.add_argument("-h", "--help", action="store_true")
    args, unknown = ap.parse_known_args()

    if args.help:
        print("Patched INTENT shim\n\n"
              "Required:\n"
              "  --logs-glob  Glob for CLEAN_FULL files\n"
              "  --out-dir    Output directory (passed to CORE)\n\n"
              "Gating controls:\n"
              "  --no-chassis, --no-tcc-lock\n"
              "  --steer-column, --latg-column, --yaw-column, --tcc-column (use '__NONE__' to disable)\n\n"
              "All other flags are forwarded unchanged to the CORE script.\n"
              "CORE file expected: driver_intent_pass_weighted__PLUS_v3__CORE.py\n")
        sys.exit(0)

    # Decide what to drop
    drop_tcc = args.no_tcc_lock or _is_none(args.tcc_column)
    drop_chassis = args.no_chassis or _is_none(args.steer_column) or _is_none(args.latg_column) or _is_none(args.yaw_column)

    files = _glob_list(args.logs_glob)
    if not files:
        print(f"[INTENT shim] No files matched: {args.logs_glob}", file=sys.stderr)
        sys.exit(2)

    ng_dir, ng_glob = _make_ng_copies(files, drop_tcc=drop_tcc, drop_chassis=drop_chassis)

    # Build the call to the CORE script
    here = Path(__file__).resolve().parent
    core = here / "driver_intent_pass_weighted__PLUS_v3__CORE.py"
    if not core.exists():
        print(f"[INTENT shim] ERROR: CORE script not found at {core}. Please rename your original to this filename.", file=sys.stderr)
        sys.exit(3)

    cmd = [sys.executable, str(core), "--logs-glob", ng_glob, "--out-dir", args.out_dir]
    # Filter out our gating-related column args to avoid confusing CORE
    skip_next = False
    filtered = []
    gated_flags = {"--tcc-column", "--steer-column", "--latg-column", "--yaw-column", "--no-chassis", "--no-tcc-lock"}
    for i, tok in enumerate(unknown):
        if skip_next:
            skip_next = False
            continue
        if tok in gated_flags:
            if tok in ("--tcc-column","--steer-column","--latg-column","--yaw-column"):
                skip_next = True
            continue
        filtered.append(tok)

    cmd.extend(filtered)
    print("[INTENT shim] running CORE:", " ".join(cmd))
    rc = subprocess.call(cmd)
    sys.exit(rc)

if __name__ == "__main__":
    main()
