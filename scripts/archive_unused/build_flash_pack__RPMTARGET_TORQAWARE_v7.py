#!/usr/bin/env python
"""
build_flash_pack__RPMTARGET_TORQAWARE_v7.py

Bundle the torque-aware SHIFT/TCC tables (Comfort + Performance),
EC3 slip tables, and their __NOHDR variants into a v7 flash pack zip.

Outputs:
  bundles/Tahoe_6L80_FlashPack__RPMTARGET_TORQAWARE__v7.zip
"""

from pathlib import Path
import textwrap
import zipfile
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    shift_dir = repo_root / "newlogs" / "output" / "01_tables" / "shift"
    tcc_dir = repo_root / "newlogs" / "output" / "01_tables" / "tcc"
    slip_dir = repo_root / "newlogs" / "output" / "01_tables" / "tcc_slip"
    bundles_dir = repo_root / "bundles"
    bundles_dir.mkdir(parents=True, exist_ok=True)

    files = {
        # SHIFT torque-aware
        "shift/COMFORT/SHIFT_TABLES__UP__Throttle17__COMFORT_TORQAWARE.tsv":
            shift_dir / "SHIFT_TABLES__UP__Throttle17__COMFORT_TORQAWARE.tsv",
        "shift/COMFORT/SHIFT_TABLES__DOWN__Throttle17__COMFORT_TORQAWARE.tsv":
            shift_dir / "SHIFT_TABLES__DOWN__Throttle17__COMFORT_TORQAWARE.tsv",
        "shift/PERF/SHIFT_TABLES__UP__Throttle17__PERF_TORQAWARE.tsv":
            shift_dir / "SHIFT_TABLES__UP__Throttle17__PERF_TORQAWARE.tsv",
        "shift/PERF/SHIFT_TABLES__DOWN__Throttle17__PERF_TORQAWARE.tsv":
            shift_dir / "SHIFT_TABLES__DOWN__Throttle17__PERF_TORQAWARE.tsv",
        # SHIFT NOHDR (optional)
        "shift/COMFORT/SHIFT_TABLES__UP__Throttle17__COMFORT_TORQAWARE__NOHDR.tsv":
            shift_dir / "SHIFT_TABLES__UP__Throttle17__COMFORT_TORQAWARE__NOHDR.tsv",
        "shift/COMFORT/SHIFT_TABLES__DOWN__Throttle17__COMFORT_TORQAWARE__NOHDR.tsv":
            shift_dir / "SHIFT_TABLES__DOWN__Throttle17__COMFORT_TORQAWARE__NOHDR.tsv",
        "shift/PERF/SHIFT_TABLES__UP__Throttle17__PERF_TORQAWARE__NOHDR.tsv":
            shift_dir / "SHIFT_TABLES__UP__Throttle17__PERF_TORQAWARE__NOHDR.tsv",
        "shift/PERF/SHIFT_TABLES__DOWN__Throttle17__PERF_TORQAWARE__NOHDR.tsv":
            shift_dir / "SHIFT_TABLES__DOWN__Throttle17__PERF_TORQAWARE__NOHDR.tsv",
        # TCC torque-aware
        "tcc/COMFORT/TCC_APPLY__Throttle17__COMFORT_TORQAWARE.tsv":
            tcc_dir / "TCC_APPLY__Throttle17__COMFORT_TORQAWARE.tsv",
        "tcc/COMFORT/TCC_RELEASE__Throttle17__COMFORT_TORQAWARE.tsv":
            tcc_dir / "TCC_RELEASE__Throttle17__COMFORT_TORQAWARE.tsv",
        "tcc/PERF/TCC_APPLY__Throttle17__PERF_TORQAWARE.tsv":
            tcc_dir / "TCC_APPLY__Throttle17__PERF_TORQAWARE.tsv",
        "tcc/PERF/TCC_RELEASE__Throttle17__PERF_TORQAWARE.tsv":
            tcc_dir / "TCC_RELEASE__Throttle17__PERF_TORQAWARE.tsv",
        # TCC NOHDR
        "tcc/COMFORT/TCC_APPLY__Throttle17__COMFORT_TORQAWARE__NOHDR.tsv":
            tcc_dir / "TCC_APPLY__Throttle17__COMFORT_TORQAWARE__NOHDR.tsv",
        "tcc/COMFORT/TCC_RELEASE__Throttle17__COMFORT_TORQAWARE__NOHDR.tsv":
            tcc_dir / "TCC_RELEASE__Throttle17__COMFORT_TORQAWARE__NOHDR.tsv",
        "tcc/PERF/TCC_APPLY__Throttle17__PERF_TORQAWARE__NOHDR.tsv":
            tcc_dir / "TCC_APPLY__Throttle17__PERF_TORQAWARE__NOHDR.tsv",
        "tcc/PERF/TCC_RELEASE__Throttle17__PERF_TORQAWARE__NOHDR.tsv":
            tcc_dir / "TCC_RELEASE__Throttle17__PERF_TORQAWARE__NOHDR.tsv",
        # Slip tables gears 1-6
        **{
            f"tcc_slip/TCC_SLIP_TABLE__GEAR{g}__EC3_FROM_LOGS.tsv":
            slip_dir / f"TCC_SLIP_TABLE__GEAR{g}__EC3_FROM_LOGS.tsv"
            for g in range(1, 7)
        },
        # Slip NOHDR
        **{
            f"tcc_slip/TCC_SLIP_TABLE__GEAR{g}__EC3_FROM_LOGS__NOHDR.tsv":
            slip_dir / f"TCC_SLIP_TABLE__GEAR{g}__EC3_FROM_LOGS__NOHDR.tsv"
            for g in range(1, 7)
        },
    }

    missing = [(arc, p) for arc, p in files.items() if not p.exists()]
    if missing:
        print("[ERROR] Missing expected files for v7 bundle:")
        for arc, p in missing:
            print(f"  {arc} -> {p}")
        sys.exit(1)

    readme_text = textwrap.dedent(
        """
        Tahoe 6L80 Flash Pack — RPMTARGET TORQAWARE v7
        ==============================================

        Contents:
          - Torque-aware SHIFT tables (Comfort & Performance)
          - Torque-aware TCC APPLY/RELEASE (Comfort & Performance)
          - EC3 slip tables for gears 1–6
          - __NOHDR headerless grids for all SHIFT/TCC/SLIP tables

        Highlights:
          - Up/down shifts place the post-shift RPM in the plateau (1900–2500)
            and hump (2500–2900), explicitly avoiding the 3000–3300 dip as a
            steady state.
          - Comfort: ultra-plush grand touring; conservative TCC with strict
            high-TPS lockout.
          - Performance: as rowdy as safely possible; more eager downshifts and
            TCC lock in torque/hp-favorable zones, with WOT lockout protection.
          - TCC built after the torque-aware SHIFT tables and constrained by
            EC3 slip maps.
        """
    ).strip() + "\n"

    zip_path = bundles_dir / "Tahoe_6L80_FlashPack__RPMTARGET_TORQAWARE__v7.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, path in files.items():
            zf.write(path, arcname)
        zf.writestr(
            "README__Tahoe_6L80_FlashPack__RPMTARGET_TORQAWARE__v7.txt",
            readme_text,
        )

    print(f"[INFO] Flash pack written to: {zip_path}")


if __name__ == "__main__":
    main()
