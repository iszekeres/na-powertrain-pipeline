#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST PATTERN Overlay (TABSAFE) — v2 (hardened headers)
- Ignores junk/non-numeric TPS headers (e.g., "Unnamed: 0", blanks).
- Sorts TPS columns numerically (0..100).
- Falls back to default 16-col axis if no valid numeric TPS headers found.
- Everything else identical to v1 (monotonic UP, gap-based DOWN, TCC unlocks, WOT, slip 9×9).
"""
import argparse, sys, io
from pathlib import Path
import pandas as pd
import numpy as np

DEF_TPS = ["0","6","12","19","25","31","37","44","50","56","62","69","75","81","87","94","100"]
UP_ROWS = ["1->2","2->3","3->4","4->5","5->6"]

def read_table_guess_headers(p: Path):
    if not p.exists():
        return None, None, None
    raw = p.read_text(encoding="utf-8", errors="replace")
    raw = raw.replace("\r\n","\n").replace("\r","\n").replace("\\t","\t").replace("/t","\t")
    df = pd.read_csv(io.StringIO(raw), sep="\t", dtype=str, engine="python").fillna("")
    # drop stray percent header, trim headers
    df.columns = [c.strip() for c in df.columns]
    if "%" in df.columns:
        df = df.drop(columns=["%"])
    # drop junk columns
    junk = []
    for c in df.columns:
        if c.lower().startswith("unnamed") or c.strip()=="" or c=="index":
            junk.append(c)
    if junk:
        df = df.drop(columns=junk)
    # ensure mph first
    cols = df.columns.tolist()
    if "mph" in cols:
        tps_cols = [c for c in cols if c != "mph"]
    else:
        tps_cols = cols[:]  # no mph; treat all as candidates
    # keep only numeric tps headers
    good = []
    for c in tps_cols:
        try:
            float(c)
            good.append(c)
        except:
            pass
    if not good:
        return df, None, None
    # sort numerically
    good_sorted = sorted(good, key=lambda x: float(x))
    # rows (prefer known UP order if present)
    rows = df["mph"].tolist() if "mph" in df.columns else None
    return df, good_sorted, rows

def default_tps_headers():
    return DEF_TPS

def write_tsv(df: pd.DataFrame, path: Path):
    path.write_text(df.to_csv(sep="\t", index=False, float_format="%.0f", lineterminator="\n"), encoding="utf-8")

def build_upshift(rows, tps_cols):
    bases = {"1->2":10, "2->3":20, "3->4":30, "4->5":40, "5->6":50}
    tps_vals = [int(float(c)) for c in tps_cols]
    idx_69 = max(i for i,t in enumerate(tps_vals) if t<=69)
    out_rows = []
    for name in (rows or UP_ROWS):
        base = bases.get(name, 10)
        vals = []
        for idx, t in enumerate(tps_vals):
            if t <= 69:
                vals.append(base + idx)
            else:
                vals.append(base + idx_69)
        # monotonic sweep
        last=None
        for i,v in enumerate(vals):
            if last is None or v>=last: last=v
            else: vals[i]=last
        out_rows.append((name, vals))
    df = pd.DataFrame({"mph": [r[0] for r in out_rows]})
    for j,c in enumerate(tps_cols):
        df[c] = [r[1][j] for r in out_rows]
    return df

def build_downshift(up_df, tps_cols, gap_lo=6.0, gap_hi=8.0):
    tps_vals = [int(float(c)) for c in tps_cols]
    pairs = [("1->2","2->1"), ("2->3","3->2"), ("3->4","4->3"), ("4->5","5->4"), ("5->6","6->5")]
    out_rows = []
    for up_name, dn_name in pairs:
        if up_name not in up_df["mph"].tolist(): continue
        u = up_df.loc[up_df["mph"]==up_name, tps_cols].iloc[0].astype(float).values.tolist()
        d = []
        for tps, uv in zip(tps_vals, u):
            g = gap_lo if tps <= 69 else gap_hi
            d.append(max(1.0, uv - g))
        # monotonic
        last=None
        for i,v in enumerate(d):
            if last is None or v>=last: last=v
            else: d[i]=last
        out_rows.append((dn_name, d))
    df = pd.DataFrame({"mph": [r[0] for r in out_rows]})
    for j,c in enumerate(tps_cols):
        df[c] = [r[1][j] for r in out_rows]
    return df

def build_tcc_apply_release(tps_cols, unlock_last_n=6, release_delta=5):
    apply_rows = ["1st Apply","2nd Apply","3rd Apply","4th Apply","5th Apply","6th Apply"]
    tps_vals = [int(float(c)) for c in tps_cols]
    idx_69 = max(i for i,t in enumerate(tps_vals) if t<=69)
    out_apply=[]
    for name in apply_rows:
        row=[]
        for i,t in enumerate(tps_vals):
            if name in ["1st Apply","2nd Apply"]:
                row.append(318)
            else:
                base = {"3rd Apply":30,"4th Apply":42,"5th Apply":52,"6th Apply":60}[name]
                if i >= len(tps_vals)-unlock_last_n:
                    row.append(318)
                else:
                    row.append(base + (i if t<=69 else idx_69))
        # monotonic ignoring 318
        last=None
        for j,v in enumerate(row):
            if v>=300: continue
            if last is None or v>=last: last=v
            else: row[j]=last
        out_apply.append((name,row))
    dfA = pd.DataFrame({"mph":[r[0] for r in out_apply]})
    for j,c in enumerate(tps_cols):
        dfA[c] = [r[1][j] for r in out_apply]
    dfR = dfA.copy()
    dfR["mph"] = dfR["mph"].str.replace("Apply","Release", regex=False)
    for i,_ in dfR.iterrows():
        for c in tps_cols:
            try: a = float(dfA.loc[i,c])
            except: a = float("nan")
            if not (a>=300 or np.isnan(a)):
                dfR.loc[i,c] = max(1.0, a - release_delta)
            else:
                dfR.loc[i,c] = a
    return dfA, dfR

def build_slip_9x9_data_only():
    row = "\t".join("20" for _ in range(9))
    return "\n".join(row for _ in range(9)) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--review-dir", required=True)
    ap.add_argument("--gap-lo", type=float, default=6.0)
    ap.add_argument("--gap-hi", type=float, default=8.0)
    ap.add_argument("--unlock-last-n", type=int, default=6)
    ap.add_argument("--release-delta", type=float, default=5.0)
    args = ap.parse_args()

    rd = Path(args.review_dir); rd.mkdir(parents=True, exist_ok=True)
    raw_up = rd / f"HPT_paste__16col__UPSHIFT__{args.tag}.tsv"

    df_raw, tps_cols, rows = read_table_guess_headers(raw_up)
    if not tps_cols:
        tps_cols = DEF_TPS
    if not rows:
        rows = UP_ROWS

    up = build_upshift(rows, tps_cols)
    dn = build_downshift(up, tps_cols, args.gap_lo, args.gap_hi)
    tccA, tccR = build_tcc_apply_release(tps_cols, args.unlock_last_n, args.release_delta)

    # Write (TABSAFE)
    up_p = rd / f"HPT_paste__16col__UPSHIFT__{args.tag}__TEST_PATTERN.tsv"
    dn_p = rd / f"HPT_paste__16col__DOWNSHIFT__{args.tag}__TEST_PATTERN.tsv"
    ta_p = rd / f"tcc_apply__mph__{args.tag}__TEST_PATTERN.tsv"
    tr_p = rd / f"tcc_release__mph__{args.tag}__TEST_PATTERN.tsv"
    up.to_csv(up_p, sep="\t", index=False, float_format="%.0f", lineterminator="\n")
    dn.to_csv(dn_p, sep="\t", index=False, float_format="%.0f", lineterminator="\n")
    tccA.to_csv(ta_p, sep="\t", index=False, float_format="%.0f", lineterminator="\n")
    tccR.to_csv(tr_p, sep="\t", index=False, float_format="%.0f", lineterminator="\n")

    # WOT blocks (up + down with same gap_lo)
    wot_up = {"1→2":45, "2→3":77, "3→4":115, "4→5":147, "5→6":200}
    lines = []
    lines.append("WOT Shift Speed vs. Shift – Normal")
    for k in ["1→2","2→3","3→4"]:
        lines.append(k); lines.append(str(int(round(wot_up[k]))))
    lines.append("\nWOT Shift Speed – Normal 5th")
    lines.append("4→5"); lines.append(str(int(round(wot_up["4→5"]))))
    lines.append("5→4"); lines.append(str(int(round(wot_up["4→5"] - args.gap_lo))))
    lines.append("\nWOT Shift Speed – Normal 6th")
    lines.append("5→6"); lines.append(str(int(round(wot_up["5→6"]))))
    lines.append("6→5"); lines.append(str(int(round(wot_up["5→6"] - args.gap_lo))))
    (rd / f"WOT_Shift_Speed__{args.tag}__TEST_PATTERN__BLOCKS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Slip 9×9 (data-only, 20 rpm)
    slip_txt = build_slip_9x9_data_only()
    for g in range(1,7):
        (rd / f"TCC_Desired_Slip__G{g}__RPMxTorque__9x9__DATA_ONLY__{args.tag}__TEST_PATTERN.tsv").write_text(slip_txt, encoding="utf-8")

    print("[OK] TEST_PATTERN v2 written (UP/DOWN/TCC/WOT + slip 9×9).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
