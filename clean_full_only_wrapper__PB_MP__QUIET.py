#!/usr/bin/env python
# clean_full_only_wrapper__PB_MP__QUIET.py
# Runs trans_clean_analyze__SAFE_REBUILD.py over raw logs, keeps ONLY FULL cleans,
# routes artifacts, deletes small focused cleans. Quiet console output.

import argparse, sys, os, shutil, subprocess, time, glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def newest(paths):
    paths = [Path(p) for p in paths]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None

def move_first(patterns, dest_dir):
    dest = Path(dest_dir); dest.mkdir(parents=True, exist_ok=True)
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(pat))
    if not hits: return []
    moved = []
    for p in hits:
        pth = Path(p)
        target = dest / pth.name
        if target.exists():
            target.unlink()
        shutil.move(str(pth), str(target))
        moved.append(str(target))
    return moved

def rm_all(patterns):
    n = 0
    for pat in patterns:
        for p in glob.glob(pat):
            try:
                Path(p).unlink()
                n += 1
            except Exception:
                pass
    return n

def run_one(raw_csv, cleaner, staging, cleaned_dir, out_root):
    raw = Path(raw_csv)
    staging = Path(staging); staging.mkdir(parents=True, exist_ok=True)
    cleaned_dir = Path(cleaned_dir); cleaned_dir.mkdir(parents=True, exist_ok=True)
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)
    out00 = out_root / "00_cleaner"; out00.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(cleaner), "--in", str(raw), "--out-dir", str(staging)]
    start = time.time()
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        return (raw.name, False, f"cleaner failed ({e.returncode})", e.stderr.strip())

    base = raw.stem  # e.g., outbound1__headerfix
    # move FULL cleans
    full_moved = move_first(
        [str(staging / f"__trans_focus__clean_FULL__*{base}*.csv")],
        cleaned_dir
    )
    # move artifacts
    _ = move_first([str(staging / f"__trans_focus__mapping__*{base}*.csv")], out00)
    _ = move_first([str(staging / f"__trans_focus__shift_events__*{base}*.csv")], out00)
    _ = move_first([str(staging / f"__trans_focus__summary__*{base}*.txt")], out00)

    # delete small focused cleans
    deleted = rm_all([str(staging / f"__trans_focus__clean__*{base}*.csv")])

    dur = time.time() - start
    if not full_moved:
        return (raw.name, False, f"no FULL output found for base '{base}'", f"{dur:.1f}s")
    return (raw.name, True, f"FULL={len(full_moved)} moved, small_clean_deleted={deleted}", f"{dur:.1f}s")

def main():
    ap = argparse.ArgumentParser(description="FULL-only cleaner wrapper (quiet).")
    ap.add_argument("--raw-glob", required=True, help=r"Glob for input raw CSVs (e.g., .\newlogs\*headerfix*.csv)")
    ap.add_argument("--cleaner",   required=True, help="Path to trans_clean_analyze__SAFE_REBUILD.py")
    ap.add_argument("--staging",   required=True, help="Staging output dir for the cleaner")
    ap.add_argument("--cleaned-dir", required=True, help="Destination for __trans_focus__clean_FULL__*.csv")
    ap.add_argument("--out-root",    required=True, help="Root for artifacts (00_cleaner)")
    ap.add_argument("--workers", type=int, default=max(2, os.cpu_count() or 2), help="Parallel workers")
    args = ap.parse_args()

    raws = sorted(glob.glob(args.raw_glob))
    if not raws:
        print(f"[wrapper] No raw files matched: {args.raw_glob}")
        sys.exit(2)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, r, args.cleaner, args.staging, args.cleaned_dir, args.out_root): r for r in raws}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append((Path(futs[fut]).name, False, f"exception: {e}", ""))

    ok = sum(1 for _, ok, _, _ in results if ok)
    fail = len(results) - ok
    # quiet summary
    print(f"[wrapper] DONE. {ok}/{len(results)} succeeded, {fail} failed.")
    for name, okf, msg, tail in results:
        status = "OK " if okf else "ERR"
        print(f"  - {status} {name}: {msg} {('('+tail+')') if tail else ''}")

    sys.exit(0 if ok>0 and fail==0 else (1 if ok>0 else 3))

if __name__ == "__main__":
    main()
