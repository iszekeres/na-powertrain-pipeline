# -*- coding: utf-8 -*-
import os, argparse, numpy as np
from passes_common import RAW, TPS, write_delta, load_clean_list, load_raw_arrays
ROWS_REL = [f"{g} Release" for g in ["1st","2nd","3rd","4th","5th","6th"]]
ROWS_APP = [f"{g} Apply"   for g in ["1st","2nd","3rd","4th","5th","6th"]]
def _neutral(out_dir, name, rows): 
    p = write_delta(out_dir, name, rows, np.zeros((len(rows), len(TPS))))
    print(f"[OK] {name} → {p} (neutral)")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--clean-list", required=True)
    ap.add_argument("--out-dir", default=r".\newlogs\output\02_passes\TCC_CRUISE"); args=ap.parse_args()
    _neutral(args.out_dir, "TCC_CRUISE__APPLY__DELTA.tsv", ROWS_APP)
if __name__=="__main__": main()
