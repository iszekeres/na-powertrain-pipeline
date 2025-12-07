#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trans_overlay_ultra_plush_from_RAW__v2_1_TABSAFE.py — baked-in normalization (requires renaming your current overlay to: trans_overlay_ultra_plush_from_RAW__v2_1_TABSAFE__BASE.py)
Workflow:
  1) This file launches trans_overlay_ultra_plush_from_RAW__v2_1_TABSAFE__BASE.py with all the same CLI args.
  2) Then it NORMALIZES the produced shift tables in-place (TABSAFE):
     • Monotonic across TPS for each row.
     • DOWNSHIFT = UPSHIFT − --gap (constant gap across TPS).
  3) TCC, WOT blocks, and slip 9×9 are untouched.

Why this design?
  - You asked to "bake into the original scripts." Replacing your original file
    with this wrapper lets you keep the SAME filename you run, while your previous
    overlay logic lives in a sibling file named trans_overlay_ultra_plush_from_RAW__v2_1_TABSAFE__BASE.py.

One-time step:
  Rename your current script to trans_overlay_ultra_plush_from_RAW__v2_1_TABSAFE__BASE.py in the same folder as this file.

Run exactly like before (same flags).

Example:
  python .\trans_overlay_ultra_plush_from_RAW__v2_1_TABSAFE.py --tag MULTILOG_RAW_WEIGHTED --review-dir ".\06_Logs\Trans_Review" --gap 6 [...]
"""
import argparse, io, sys, subprocess
from pathlib import Path
import pandas as pd
import numpy as np

LABEL_SUFFIX = "ULTRA_PLUSH"
INNER = Path("trans_overlay_ultra_plush_from_RAW__v2_1_TABSAFE__BASE.py")

def read_tsv_fix(p: Path) -> pd.DataFrame:
    raw = p.read_text(encoding="utf-8", errors="replace")
    raw = raw.replace("\r\n","\n").replace("\r","\n").replace("\\t","\t").replace("/t","\t")
    df = pd.read_csv(io.StringIO(raw), sep="\t", dtype=str, engine="python").fillna("")
    if "%" in df.columns: df = df.drop(columns=["%"])
    df.columns = [c.strip() for c in df.columns]
    if "mph" in df.columns:
        df = df[["mph"] + [c for c in df.columns if c != "mph"]]
    return df

def ensure_monotone_row(values):
    out=[]; last=None
    for v in values:
        try: fv = float(v)
        except: fv = float("nan")
        if np.isnan(fv):
            nv = last if last is not None else None
        elif fv >= 300:    # keep 318 sentinel as-is
            nv = 318
        else:
            nv = fv if (last is None or fv >= last) else last
            last = nv if nv < 300 else last
        out.append(nv)
    first = next((x for x in out if isinstance(x,(int,float))), None)
    out = [first if x is None else x for x in out]
    return [int(round(x)) if isinstance(x,(int,float)) else x for x in out]

def normalize_up_down(review_dir: Path, tag: str, gap: float):
    up_p = review_dir / f"HPT_paste__16col__UPSHIFT__{tag}__ULTRA_PLUSH.tsv"
    dn_p = review_dir / f"HPT_paste__16col__DOWNSHIFT__{tag}__ULTRA_PLUSH.tsv"
    up = read_tsv_fix(up_p)
    dn = read_tsv_fix(dn_p)
    tps_cols = [c for c in up.columns if c != "mph"]

    # 1) Monotone UPSHIFT
    for i, r in up.iterrows():
        row = pd.to_numeric(r[tps_cols], errors="coerce").astype(float).tolist()
        row = ensure_monotone_row(row)
        for c, v in zip(tps_cols, row):
            up.at[i, c] = v

    # 2) DOWNSHIFT derived from UPSHIFT with constant gap
    pairs = [("1->2","2->1"),("2->3","3->2"),("3->4","4->3"),("4->5","5->4"),("5->6","6->5")]
    for up_name, dn_name in pairs:
        if up_name in up["mph"].tolist() and dn_name in dn["mph"].tolist():
            iu = up["mph"].tolist().index(up_name)
            idn = dn["mph"].tolist().index(dn_name)
            u = pd.to_numeric(up.loc[iu, tps_cols], errors="coerce").astype(float).tolist()
            new_dn = []
            for uv in u:
                if np.isnan(uv) or uv >= 300:
                    new_dn.append(uv)
                else:
                    new_dn.append(max(1.0, uv - gap))
            new_dn = ensure_monotone_row(new_dn)
            for c, v in zip(tps_cols, new_dn):
                dn.at[idn, c] = v

    # TABSAFE write-back (overwrite the originals)
    up.to_csv(up_p, sep="\t", index=False, float_format="%.0f", lineterminator="\n")
    dn.to_csv(dn_p, sep="\t", index=False, float_format="%.0f", lineterminator="\n")

def main():
    # parse to learn tag/review-dir/gap, but forward all args to INNER
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--review-dir", required=True)
    ap.add_argument("--gap", type=float, default=6.0)
    args, unknown = ap.parse_known_args()

    if not INNER.exists():
        raise SystemExit(f"ERROR: expected your previous overlay renamed to {INNER.name} next to this file.")

    # 1) run the inner overlay with all original args
    rc = subprocess.call([sys.executable, str(INNER)] + sys.argv[1:])
    if rc != 0:
        raise SystemExit(rc)

    # 2) normalize the produced UP/DOWN tables in-place
    rd = Path(args.review_dir)
    normalize_up_down(rd, args.tag, args.gap)
    print("[OK] Shift tables normalized in-place (monotone rows + constant gap).")

if __name__ == "__main__":
    raise SystemExit(main())
