#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra Street Racer — MAXATTACK v2 TABSAFE (BASE) — with baked-in TCC shaping + UP/DOWN normalization
- Keeps your data-driven UP/DOWN values (from RAW), then enforces invariants:
  • UPSHIFT rows monotonic across TPS
  • DOWNSHIFT = UPSHIFT − --gap (constant across TPS)
- Builds TCC Apply/Release to MAXATTACK rules:
  • 1st & 2nd unlocked (318)
  • 3rd–6th Apply shaped by anchors at TPS 0/25/50/69, monotone, last N TPS unlocked (default N=6)
  • Release = Apply − 5 mph where Apply<300, else 318
- Writes slip 9×9 per gear (data-only), clamped 5–30 (constant 20 baseline; integers)
- TABSAFE outputs (real tabs, normalized newlines)
- Accepts --smart-sixth / energy flags but ignores them here (no error)
"""
import argparse, io, sys
from pathlib import Path
import pandas as pd
import numpy as np

DEF_TPS = ["0","6","12","19","25","31","37","44","50","56","62","69","75","81","87","94","100"]

def read_tsv_fix(p: Path) -> pd.DataFrame:
    raw = p.read_text(encoding="utf-8", errors="replace")
    raw = raw.replace("\r\n","\n").replace("\r","\n").replace("\\t","\t").replace("/t","\t")
    df = pd.read_csv(io.StringIO(raw), sep="\t", dtype=str, engine="python").fillna("")
    if "%" in df.columns: df = df.drop(columns=["%"])
    df.columns = [c.strip() for c in df.columns]
    if "mph" in df.columns:
        df = df[["mph"] + [c for c in df.columns if c != "mph"]]
    return df

def sanitize_headers(df: pd.DataFrame):
    if "mph" not in df.columns:
        raise SystemExit("Missing 'mph' column in input TSV.")
    tps_cols = [c for c in df.columns if c != "mph"]
    good = []
    for c in tps_cols:
        try:
            float(c); good.append(c)
        except:
            pass
    if not good: good = DEF_TPS
    good_sorted = sorted(good, key=lambda x: float(x))
    return ["mph"] + good_sorted, good_sorted

def ensure_monotone_row(values):
    out=[]; last=None
    for v in values:
        try: fv=float(v)
        except: fv=np.nan
        if np.isnan(fv):
            nv = last if last is not None else None
        elif fv>=300:
            nv = 318
        else:
            nv = fv if (last is None or fv>=last) else last
            last = nv if nv < 300 else last
        out.append(nv)
    first = next((x for x in out if isinstance(x,(int,float))), None)
    out = [first if x is None else x for x in out]
    return [int(round(x)) if isinstance(x,(int,float)) else x for x in out]

def normalize_up_down(up_df: pd.DataFrame, dn_df: pd.DataFrame, tps_cols, gap: float):
    tps_vals = [int(float(c)) for c in tps_cols]
    # Monotone UPSHIFT
    for i, r in up_df.iterrows():
        row = pd.to_numeric(r[tps_cols], errors="coerce").astype(float).tolist()
        row = ensure_monotone_row(row)
        for c, v in zip(tps_cols, row):
            up_df.at[i, c] = v
    # Rebuild DOWN from UP with constant gap
    pairs = [("1->2","2->1"),("2->3","3->2"),("3->4","4->3"),("4->5","5->4"),("5->6","6->5")]
    for up_name, dn_name in pairs:
        if up_name in up_df["mph"].tolist() and dn_name in dn_df["mph"].tolist():
            iu = up_df["mph"].tolist().index(up_name)
            idn = dn_df["mph"].tolist().index(dn_name)
            u = pd.to_numeric(up_df.loc[iu, tps_cols], errors="coerce").astype(float).tolist()
            new_dn = []
            for uv in u:
                if np.isnan(uv) or uv>=300: new_dn.append(uv)
                else: new_dn.append(max(1.0, uv - gap))
            new_dn = ensure_monotone_row(new_dn)
            for c, v in zip(tps_cols, new_dn):
                dn_df.at[idn, c] = v
    return up_df, dn_df

# TCC shaping (MAXATTACK anchors)
ANCHORS = {
    "3rd Apply": (34,38,44,50),  # at TPS 0,25,50,69
    "4th Apply": (42,46,52,58),
    "5th Apply": (44,50,56,62),
    "6th Apply": (58,60,64,70),
}

def interp_piecewise(tps_values, a0,a25,a50,a69):
    pts = [(0,a0),(25,a25),(50,a50),(69,a69)]
    def seg(x):
        if x <= 0: return a0
        if x >= 69: return a69
        for (x1,y1),(x2,y2) in zip(pts[:-1], pts[1:]):
            if x1 <= x <= x2:
                t = (x - x1)/(x2 - x1)
                return y1 + t*(y2 - y1)
        return a69
    return [seg(x) for x in tps_values]

def build_tcc_apply_release(tps_cols, unlock_last_n=6, release_delta=5):
    tps = [float(c) for c in tps_cols]
    idx_last_unlock = len(tps_cols) - unlock_last_n
    # Apply
    rows = []
    for name in ["1st Apply","2nd Apply"]:
        rows.append([name] + [318 for _ in tps_cols])
    for name in ["3rd Apply","4th Apply","5th Apply","6th Apply"]:
        a0,a25,a50,a69 = ANCHORS[name]
        vals = interp_piecewise(tps, a0,a25,a50,a69)
        out=[]; last_val=None
        for j, c in enumerate(tps_cols):
            if j >= idx_last_unlock:
                out.append(318)
            else:
                v2 = int(round(max(1, min(220, vals[j]))))
                if last_val is None or v2 >= last_val: last_val = v2
                else: v2 = last_val
                out.append(v2)
        rows.append([name] + out)
    dfA = pd.DataFrame(rows, columns=["mph"] + tps_cols)
    # Release
    rowsR = []
    for name in ["1st Release","2nd Release"]:
        rowsR.append([name] + [318 for _ in tps_cols])
    for gear in ["3rd","4th","5th","6th"]:
        ar = dfA[dfA["mph"]==f"{gear} Apply"].iloc[0]
        r = []
        for c in tps_cols:
            av = pd.to_numeric(ar[c], errors="coerce")
            if not pd.isna(av) and av < 300:
                r.append(int(max(1, av - release_delta)))
            else:
                r.append(318)
        rowsR.append([f"{gear} Release"] + r)
    dfR = pd.DataFrame(rowsR, columns=["mph"] + tps_cols)
    return dfA, dfR

def write_tabsafe(df: pd.DataFrame, p: Path):
    p.write_text(df.to_csv(sep="\t", index=False, float_format="%.0f", lineterminator="\n"), encoding="utf-8")

def write_slip_9x9_data_only(out_dir: Path, tag: str, label="MAXATTACK"):
    # 9x9 constant 20 rpm; clamp [5,30]; integers
    row = "\t".join("20" for _ in range(9))
    txt = "\n".join(row for _ in range(9)) + "\n"
    for g in range(1,7):
        (out_dir / f"TCC_Desired_Slip__G{g}__RPMxTorque__9x9__DATA_ONLY__{tag}__{label}.tsv").write_text(txt, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--review-dir", required=True)
    ap.add_argument("--gap", type=float, default=6.0)
    ap.add_argument("--land", type=float, default=4350.0)
    ap.add_argument("--unlock-last-n", type=int, default=6)
    ap.add_argument("--sixth-disable-tps", type=float, default=75.0)

    # Accept extra args (ignored but tolerated)
    ap.add_argument("--smart-sixth", action="store_true", default=False)
    ap.add_argument("--energy-csv", type=str, default=None)
    ap.add_argument("--energy-p90-thresh", type=float, default=None)
    ap.add_argument("--energy-nudge", type=float, default=None)

    args = ap.parse_args()
    rd = Path(args.review_dir); rd.mkdir(parents=True, exist_ok=True)

    # 1) Read RAW up/down as baseline
    up_raw_p = rd / f"HPT_paste__16col__UPSHIFT__{args.tag}.tsv"
    dn_raw_p = rd / f"HPT_paste__16col__DOWNSHIFT__{args.tag}.tsv"
    if not up_raw_p.exists() or not dn_raw_p.exists():
        raise SystemExit("RAW up/down TSVs not found. Expected baseline files without suffix (RAW result).")
    up = read_tsv_fix(up_raw_p)
    dn = read_tsv_fix(dn_raw_p)

    # 2) Header sanitation / order
    cols, tps_cols = sanitize_headers(up)
    up = up[cols]
    dn = dn[cols]

    # 3) Normalize UP/DOWN (monotone + constant gap)
    upN, dnN = normalize_up_down(up.copy(), dn.copy(), tps_cols, gap=args.gap)

    # 4) Write UP/DOWN (MAXATTACK)
    up_out_p = rd / f"HPT_paste__16col__UPSHIFT__{args.tag}__MAXATTACK.tsv"
    dn_out_p = rd / f"HPT_paste__16col__DOWNSHIFT__{args.tag}__MAXATTACK.tsv"
    write_tabsafe(upN, up_out_p)
    write_tabsafe(dnN, dn_out_p)

    # 5) Build TCC Apply/Release per MAXATTACK anchors
    tccA, tccR = build_tcc_apply_release(tps_cols, unlock_last_n=args.unlock_last_n, release_delta=5)
    tccA_p = rd / f"tcc_apply__mph__{args.tag}__MAXATTACK.tsv"
    tccR_p = rd / f"tcc_release__mph__{args.tag}__MAXATTACK.tsv"
    write_tabsafe(tccA, tccA_p)
    write_tabsafe(tccR, tccR_p)

    # 6) WOT blocks: if RAW WOT exists, pass-through up and add down with gap; else write defaults
    wot_raw_p = rd / f"WOT_Shift_Speed__{args.tag}__BLOCKS.txt"
    if wot_raw_p.exists():
        txt = wot_raw_p.read_text(encoding="utf-8", errors="replace")
        # very simple parse for up entries; add downs with same gap
        # Expected sections: Normal (1→2,2→3,3→4), 5th (4→5), 6th (5→6)
        import re
        ups = {}
        for m in re.finditer(r"([1-6]→[1-6])\s*\n\s*(\d+)", txt):
            ups[m.group(1)] = int(m.group(2))
        def block():
            lines = []
            lines.append("WOT Shift Speed vs. Shift – Normal")
            for k in ["1→2","2→3","3→4"]:
                if k in ups:
                    lines += [k, str(ups[k])]
            lines.append("\nWOT Shift Speed – Normal 5th")
            if "4→5" in ups:
                lines += ["4→5", str(ups["4→5"])]
                lines += ["5→4", str(max(1, ups["4→5"] - int(round(args.gap))))]
            lines.append("\nWOT Shift Speed – Normal 6th")
            if "5→6" in ups:
                lines += ["5→6", str(ups["5→6"])]
                lines += ["6→5", str(max(1, ups["5→6"] - int(round(args.gap))))]
            return "\n".join(lines) + "\n"
        wtxt = block()
    else:
        # defaults (from earlier pass) with downs added
        wtxt = "\n".join([
            "WOT Shift Speed vs. Shift – Normal",
            "1→2","45",
            "2→3","77",
            "3→4","115",
            "\nWOT Shift Speed – Normal 5th",
            "4→5","147",
            "5→4", str(147 - int(round(args.gap))),
            "\nWOT Shift Speed – Normal 6th",
            "5→6","200",
            "6→5", str(200 - int(round(args.gap))),
            ""
        ])
    wot_out_p = rd / f"WOT_Shift_Speed__{args.tag}__MAXATTACK__BLOCKS.txt"
    wot_out_p.write_text(wtxt, encoding="utf-8")

    # 7) Slip 9×9 (data-only) with clamp [5,30]
    write_slip_9x9_data_only(rd, args.tag, label="MAXATTACK")

    print("[OK] MAXATTACK v2 BASE: wrote UP/DOWN (normalized), TCC Apply/Release (shaped), WOT (up+down), and slip 9×9.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
