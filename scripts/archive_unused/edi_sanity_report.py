#!/usr/bin/env python3
# edi_sanity_report.py
# Purpose: Print a compact sanity summary from EDI and (optional) TPS WARP.
# Inputs:
#   --edi-file  path/to/EDI_PROFILE__pergear.tsv
#   --warp-file path/to/TPS_WARP__pergear.tsv (optional)
# Output:
#   Text summary to file (via --out) or stdout

import argparse, sys
import numpy as np
import pandas as pd
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edi-file", required=True)
    ap.add_argument("--warp-file", default=None)
    ap.add_argument("--out", default=None)
    return ap.parse_args()

def main():
    a = parse_args()
    edi = pd.read_csv(a.edi_file, sep="\t")
    buf = []

    buf.append("=== EDI PROFILE SUMMARY ===")
    gears = sorted(edi["gear"].unique())
    buf.append(f"Gears in profile: {gears}")
    for g in gears:
        eg = edi[edi["gear"]==g].sort_values("mph")
        mph_lo, mph_hi = eg["mph"].min(), eg["mph"].max()
        base_lo, base_hi = eg["base_tps"].min(), eg["base_tps"].max()
        scale_med = eg["scale"].median()
        buf.append(f"  G{g}: mph {mph_lo:.1f}-{mph_hi:.1f} | base TPS {base_lo:.1f}→{base_hi:.1f} | median scale {scale_med:.2f}")

    # Approx "TPS to roll" via low-mph bins in 1st gear
    g1 = edi[edi["gear"]==1].sort_values("mph")
    if not g1.empty:
        low = g1[g1["mph"]<=3.0]
        est_move = low["base_tps"].median() if not low.empty else g1["base_tps"].iloc[:3].median()
        buf.append(f"\nApprox TPS to roll (G1 @ ~0-3 mph): ~{est_move:.1f}% (baseline before bias_idle)")
    else:
        buf.append("\nNo G1 rows; cannot estimate TPS-to-roll.")

    # Optional warp peek
    if a.warp_file and Path(a.warp_file).exists():
        warp = pd.read_csv(a.warp_file, sep="\t")
        buf.append("\n=== WARP SNAPSHOT (q -> TPS) ===")
        for g in gears[:3]:
            wg = warp[warp["gear"]==g]
            if wg.empty: 
                buf.append(f"  G{g}: (no warp coverage)")
                continue
            # take a representative mph band near median speed
            mphs = sorted(wg["mph"].unique())
            mid_mph = mphs[len(mphs)//2]
            wgm = wg[wg["mph"]==mid_mph].sort_values("q")
            # show a few quantiles
            qs = [0.1,0.25,0.5,0.75,0.9]
            mini = []
            for q in qs:
                # nearest q
                idx = (wgm["q"]-q).abs().idxmin()
                mini.append(f"q{q:.2f}={float(wgm.loc[idx,'tps_at_q']):.1f}%")
            buf.append(f"  G{g} @ ~{mid_mph:.1f} mph: " + ", ".join(mini))

    out = "\n".join(buf) + "\n"
    if a.out:
        Path(a.out).write_text(out, encoding="utf-8")
        print(f"[OK] Wrote {a.out}")
    else:
        sys.stdout.write(out)

if __name__ == "__main__":
    main()
