# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd
from passes_common import (read_clean_list, require_columns, REQ_CANON, parse_gear,
                           ROW_DN, TPS, write_delta, write_dbg)

def main(clean_dir=r".\newlogs\cleaned", out_dir=r".\newlogs\output\02_passes\KICKDOWN",
         dthr_min=10.0, min_events=30, bump=0.3):
    files=read_clean_list(clean_dir)
    used=0; events=0
    hit_bins=set()
    for p in files:
        require_columns(p, [REQ_CANON["time"],REQ_CANON["mph"],REQ_CANON["thr"],REQ_CANON["gear"]])
        df = pd.read_csv(p, usecols=[REQ_CANON["time"],REQ_CANON["mph"],REQ_CANON["thr"],REQ_CANON["gear"]])
        thr = pd.to_numeric(df[REQ_CANON["thr"]], errors="coerce").to_numpy()
        ga  = parse_gear(df[REQ_CANON["gear"]]).to_numpy()
        dthr = np.diff(np.nan_to_num(thr, nan=0.0), prepend=thr[0])
        idx  = np.where(dthr >= dthr_min)[0]
        for i in idx:
            if i+50 < len(ga) and np.nanmax(ga[i:i+50]) < ga[i]:  # look for downshift after spike
                continue
            tbin = TPS[-1] if np.isnan(thr[i]) else min(TPS, key=lambda x: abs(x-thr[i]))
            if tbin>=25: hit_bins.add(tbin); events += 1
        used += 1
    # build output
    mat=[]
    for row in ROW_DN:
        row_d=[]
        for t in TPS:
            v = bump if (events>=min_events and t>=25) else 0.0
            row_d.append(v)
        mat.append(row_d)
    os.makedirs(out_dir, exist_ok=True)
    write_delta(os.path.join(out_dir,"KICKDOWN__SHIFT_DOWN__DELTA.tsv"), ROW_DN, np.array(mat))
    write_dbg(os.path.join(out_dir,"KICKDOWN__DEBUG_SUMMARY.csv"),
              files_used=used, events=events, min_events=min_events, bins_hit=len(hit_bins))
if __name__=="__main__": main()
