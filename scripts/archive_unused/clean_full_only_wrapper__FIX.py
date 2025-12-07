    #!/usr/bin/env python3
    """clean_full_only_wrapper.py

    Runs your existing cleaner on raw CSVs and keeps ONLY the CLEAN_FULL files.
    - CLEAN_FULL files are moved to: --cleaned-dir (default: .\newlogs\cleaned)
    - Other cleaner artifacts (mapping/shift_events/summary) are moved to: --out-root\00_cleaner
    - The small focused CLEAN files (__trans_focus__clean__*.csv) are DELETED by default
      (pass --keep-focused to retain them and they will be moved into 00_cleaner).

    Usage (PowerShell, from repo root containing trans_clean_analyze__SAFE_REBUILD.py):

      python .\clean_full_only_wrapper.py `
        --raw-glob ".\newlogs\*headerfix*.csv" `
        --cleaner  ".\trans_clean_analyze__SAFE_REBUILD.py" `
        --staging  ".\newlogs\_staging_Review" `
        --cleaned-dir ".\newlogs\cleaned" `
        --out-root ".\newlogs\output"
    """
    import argparse, glob, os, shutil, subprocess, sys
    from pathlib import Path

    def ensure_dir(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

    def move_many(files, dest_dir: Path):
        ensure_dir(dest_dir)
        moved = []
        for f in files:
            f = Path(f)
            to = dest_dir / f.name
            if f.resolve() == to.resolve():
                moved.append(to)
                continue
            shutil.move(str(f), str(to))
            moved.append(to)
        return moved

    def run_cleaner_on(raw: Path, cleaner: Path, staging: Path):
        # Try --in first (newer cleaners), then fall back to --in-glob
        cmd1 = [sys.executable, str(cleaner), "--in", str(raw), "--out-dir", str(staging)]
        cmd2 = [sys.executable, str(cleaner), "--in-glob", str(raw), "--out-dir", str(staging)]
        try:
            subprocess.check_call(cmd1)
            return True
        except subprocess.CalledProcessError:
            subprocess.check_call(cmd2)
            return True
        except FileNotFoundError:
            print(f"[wrapper] Cleaner not found: {cleaner}")
            return False

    def main():
        ap = argparse.ArgumentParser(description="Keep only CLEAN_FULL files and route outputs to ./newlogs/cleaned + ./newlogs/output/00_cleaner")
        ap.add_argument("--raw-glob", required=True, help="Glob for input raw CSVs (e.g., .\newlogs\*headerfix*.csv)")
        ap.add_argument("--cleaner", required=True, help="Path to trans_clean_analyze__SAFE_REBUILD.py")
        ap.add_argument("--staging", default=r".
ewlogs\_staging_Review", help="Temporary out-dir for cleaner")
        ap.add_argument("--cleaned-dir", default=r".
ewlogs\cleaned", help="Where to place CLEAN_FULL outputs")
        ap.add_argument("--out-root", default=r".
ewlogs\output", help="Where to place other outputs (00_cleaner subfolder)")
        ap.add_argument("--keep-focused", action="store_true", help="Keep focused CLEAN files instead of deleting them")
        args = ap.parse_args()

        raw_files = [Path(p) for p in glob.glob(args.raw_glob)]
        if not raw_files:
            print(f"[wrapper] No raw files matched {args.raw_glob}")
            sys.exit(0)

        cleaner = Path(args.cleaner)
        if not cleaner.exists():
            print(f"[wrapper] Cleaner not found: {cleaner}")
            sys.exit(1)

        staging = ensure_dir(Path(args.staging))
        cleaned_dir = ensure_dir(Path(args.cleaned-dir if hasattr(args, 'cleaned-dir') else args.cleaned_dir))
        out_root = ensure_dir(Path(args.out_root))
        out_cleaner = ensure_dir(out_root / "00_cleaner")

        # Run cleaner per raw
        for raw in raw_files:
            print(f"[wrapper] Cleaning: {raw}")
            ok = run_cleaner_on(raw, cleaner, staging)
            if not ok:
                sys.exit(1)

        # Route outputs
        # Move FULL cleaned files
        full_clean = list(Path(staging).glob("__trans_focus__clean_FULL__*.csv"))
        moved_full = move_many(full_clean, cleaned_dir)
        print(f"[wrapper] Moved CLEAN_FULL -> {cleaned_dir} ({len(moved_full)} files)")

        # Handle focused small CLEANs
        small_clean = list(Path(staging).glob("__trans_focus__clean__*.csv"))
        if small_clean and not args.keep_focused and not getattr(args, "keep-focused", False):
            for f in small_clean:
                try:
                    Path(f).unlink(missing_ok=True)
                except TypeError:
                    # Python <3.8: no missing_ok
                    try:
                        Path(f).unlink()
                    except FileNotFoundError:
                        pass
            print(f"[wrapper] Deleted focused CLEAN files: {len(small_clean)}")
        elif small_clean:
            move_many(small_clean, out_cleaner)

        # Move other cleaner outputs
        patterns = [
            "__trans_focus__shift_events__*.csv",
            "__trans_focus__mapping__*.csv",
            "__trans_focus__summary__*.txt",
        ]
        others = []
        for pat in patterns:
            others += list(Path(staging).glob(pat))
        moved = move_many(others, out_cleaner)
        print(f"[wrapper] Moved other cleaner artifacts -> {out_cleaner} ({len(moved)} files)")
        print("[wrapper] DONE. Next: run passes against CLEAN_FULL in .\newlogs\cleaned")

    if __name__ == "__main__":
        main()
