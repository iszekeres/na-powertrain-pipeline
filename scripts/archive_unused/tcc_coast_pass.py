# -*- coding: utf-8 -*-
import os, argparse, numpy as np
from passes_common import RAW, TPS, write_delta, load_clean_list, load_raw_arrays
ROWS_REL = [f"{g} Release" for g in ["1st","2nd","3rd","4th","5th","6th"]]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--clean-list", required=True)
    ap.add_argument("--out-dir", default=r".\newlogs\output\02_passes\TCC_COAST"); args=ap.parse_args()
    p = write_delta(args.out_dir, "TCC_COAST__RELEASE__DELTA.tsv", ROWS_REL, np.zeros((6,len(TPS))))
    print(f"[OK] TCC COAST → {p} (neutral)")
if __name__=="__main__": main()
