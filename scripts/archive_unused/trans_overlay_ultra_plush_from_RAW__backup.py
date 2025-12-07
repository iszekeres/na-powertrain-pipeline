#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra Plush Overlay (v2, hardened) — shaped TCC + 5–30 slip clamp
- 16-col TPS, monotonic rows, ~gap mph hysteresis (default 6)
- TCC applies shaped across TPS; last 4 TPS bins unlocked (81/87/94/100)
- Release = Apply − 3 mph (comfort)
- WOT blocks include up + down with --gap hysteresis
- Slip 9×9 per gear with hard clamp [5,30], non-increasing down torque rows, ints
- Robust I/O: auto row labels, NaN-safe writer (no sentinels), safe monotonic
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

TPS_AXIS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
COLS = ["mph"] + [str(x) for x in TPS_AXIS]
RPM_AXIS = [1000,1200,1400,1600,1800,2000,2200,2400,4400]
TORQUE_AXIS = [0, 0.0922, 36.8781, 88.5075, 103.2587, 121.6978, 132.7612, 143.8246, 184.3905]

def read_tsv(p):
    df = pd.read_csv(p, sep="\t", dtype=str).fillna("")
    keep = [c for c in df.columns if c in COLS]
    keep = (["mph"] if "mph" in keep else []) + [c for c in keep if c != "mph"]
    if not keep:
        raise ValueError(f"{p} missing expected columns")
    df = df[keep].copy()
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def write_tsv(df, p):
    out = df.copy()
    cols = [c for c in out.columns if c != "mph"]
    for i in range(len(out)):
        last = None
        for c in cols:
            try:
                v = float(out.at[i, c])
            except Exception:
                v = float('nan')
            if v != v:  # NaN
                out.at[i, c] = last if last is not None else 1
            else:
                last = v
    out.to_csv(p, sep="\t", index=False, float_format="%.0f")

def ensure_monotonic_row(vals):
    """Non-decreasing across TPS for numeric cells; preserve 318 unlock; no sentinels."""
    out = []
    last = None
    for v in vals:
        try:
            fv = float(v)
        except Exception:
            fv = float('nan')
        if fv != fv:  # NaN
            nv = last if last is not None else None
        elif fv >= 300:
            nv = 318
        else:
            nv = fv if (last is None or fv >= last) else last
            last = nv if nv < 300 else last
        out.append(nv)
    first_num = next((x for x in out if isinstance(x, (int,float))), None)
    out = [first_num if (x is None) else x for x in out]
    out = [int(round(x)) if isinstance(x, (int,float)) else x for x in out]
    return out

def force_labels(df, defaults):
    if df is None or len(df) == 0:
        return pd.DataFrame({"mph": defaults[:0]})
    if "mph" not in df.columns:
        df.insert(0, "mph", defaults[:len(df)])
    elif df["mph"].astype(str).str.strip().eq("").all():
        df["mph"] = defaults[:len(df)]
    return df

# --- Shift overlays (light touch) ---
def overlay_upshift_plush(df_up):
    mat = df_up[[c for c in df_up.columns if c != "mph"]].apply(pd.to_numeric, errors="coerce")
    for r in range(mat.shape[0]):
        mat.iloc[r] = ensure_monotonic_row(mat.iloc[r].tolist())
    out = df_up.copy(); out[[c for c in out.columns if c != "mph"]] = mat
    return out

def overlay_downshift_from_up(df_up, df_dn, gap):
    up = df_up.copy(); dn = df_dn.copy()
    up_mat = up[[c for c in up.columns if c != "mph"]].apply(pd.to_numeric, errors="coerce")
    dn_mat = dn[[c for c in dn.columns if c != "mph"]].apply(pd.to_numeric, errors="coerce")
    m = min(up_mat.shape[0], dn_mat.shape[0])
    for i in range(m):
        base = up_mat.iloc[i].astype(float).to_numpy()
        new_dn = np.clip(base - gap, 1, None)
        new_dn = ensure_monotonic_row(new_dn.tolist())
        dn_mat.iloc[i] = new_dn
    out = dn.copy(); out[[c for c in out.columns if c != "mph"]] = dn_mat
    return out

# --- TCC applies with TPS-shaped anchors (no flat rows) ---
def build_tcc_plush(tcc_apply_raw, tcc_release_raw, unlock_last_n=4, temp_guard=None):
    """
    Plush: shaped applies across TPS with last 4 bins unlocked; Release = Apply − 3.
    Anchors (mph) at TPS [0,25,50,69]:
      3rd: 28, 32, 36, 42
      4th: 36, 40, 46, 52
      5th: 44, 48, 54, 60
      6th: 50, 54, 58, 64
    """
    unlock_cols = [str(x) for x in TPS_AXIS[-unlock_last_n:]]  # default last 4
    anchors = {
        "3rd Apply": [28,32,36,42],
        "4th Apply": [36,40,46,52],
        "5th Apply": [44,48,54,60],
        "6th Apply": [50,54,58,64],
    }

    def interp(tps, a):
        if tps <= 25: return a[0] + (a[1]-a[0])*(tps-0)/25.0
        if tps <= 50: return a[1] + (a[2]-a[1])*(tps-25)/25.0
        return a[2] + (a[3]-a[2])*(tps-50)/19.0  # up to 69

    A_df = tcc_apply_raw.copy()
    if "mph" in A_df.columns:
        A_df["mph"] = A_df["mph"].replace({
            "G1":"1st Apply","G2":"2nd Apply","G3":"3rd Apply","G4":"4th Apply","G5":"5th Apply","G6":"6th Apply"
        })
    cols = [c for c in A_df.columns if c != "mph"]
    A = A_df[cols].apply(pd.to_numeric, errors="coerce")

    rows = A_df["mph"].tolist() if "mph" in A_df.columns else []
    for i, name in enumerate(rows):
        if name in ["1st Apply","2nd Apply"]:
            A.iloc[i] = 318
            continue
        if name in anchors:
            row = []
            for c in cols:
                tps = int(float(c))
                if c in unlock_cols or tps >= 81:  # plush unlock last 4 bins
                    row.append(318)
                else:
                    row.append(max(1, interp(tps, anchors[name])))
            row = ensure_monotonic_row(row)
            A.iloc[i] = row

    if temp_guard:
        for i, name in enumerate(rows):
            if name in anchors:
                row = []
                for val in list(A.iloc[i]):
                    if pd.isna(val): row.append(val)
                    elif val >= 300: row.append(318)
                    else: row.append(val + 2)
                row = ensure_monotonic_row(row)
                A.iloc[i] = row

    apply_out = A_df.copy(); apply_out[cols] = A

    # Release = Apply − 3
    R = A.copy()
    for r in range(R.shape[0]):
        row = []
        for val in list(R.iloc[r]):
            if pd.isna(val): row.append(val)
            elif val >= 300: row.append(318)
            else: row.append(max(1, val - 3))
        row = ensure_monotonic_row(row)
        R.iloc[r] = row
    release_out = tcc_release_raw.copy(); release_out[cols] = R
    return apply_out, release_out

# --- Slip 9x9 with hard clamp [5,30] ---
def write_slip_9x9(tag, out_dir, vmin, vmax, base_min, base_max, g_offsets):
    cols = len(RPM_AXIS); rows = len(TORQUE_AXIS)
    base = np.linspace(base_min, base_max, cols)[None, :]
    base = np.repeat(base, rows, axis=0)
    for gear, off in g_offsets.items():
        grid = base + off
        grid = np.clip(grid, vmin, vmax)
        for c in range(cols):
            prev = None
            for r in range(rows):
                val = grid[r, c]
                if prev is None:
                    prev = val
                else:
                    if val > prev:
                        val = prev
                    prev = val
                grid[r, c] = val
        grid = np.rint(grid).astype(int)
        out_name = f"TCC_Desired_Slip__G{gear}__RPMxTorque__9x9__DATA_ONLY__{tag}__ULTRA_PLUSH.tsv"
        out_path = Path(out_dir) / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            for r in range(rows):
                f.write("\t".join(str(int(x)) for x in grid[r, :]) + "\n")

# --- WOT blocks (include downshifts with ~gap) ---
def write_wot_blocks(rd, tag, gap, ratios, caps):
    def rint(x):
        try: return int(round(float(x)))
        except: return 0
    def rpm_to_mph(rpm, ratio, tire_dia=32.5, fd=3.08, K=336.0):
        return (rpm * tire_dia) / (ratio * fd * K)

    up12 = rpm_to_mph(caps[1], ratios[0]); up23 = rpm_to_mph(caps[2], ratios[1]); up34 = rpm_to_mph(caps[3], ratios[2])
    up45 = rpm_to_mph(caps[4], ratios[3]); up56 = rpm_to_mph(caps[5], ratios[4])
    d21 = max(1.0, up12 - gap); d32 = max(1.0, up23 - gap); d43 = max(1.0, up34 - gap)
    d54 = max(1.0, up45 - gap); d65 = max(1.0, up56 - gap)

    lines = []
    lines.append("WOT Shift Speed vs. Shift – Normal")
    lines.append(f"1→2\n{rint(up12)}"); lines.append(f"2→3\n{rint(up23)}"); lines.append(f"3→4\n{rint(up34)}")
    lines.append(f"2→1\n{rint(d21)}"); lines.append(f"3→2\n{rint(d32)}"); lines.append(f"4→3\n{rint(d43)}")
    lines.append("WOT Shift Speed – Normal 5th")
    lines.append(f"4→5\n{rint(up45)}"); lines.append(f"5→4\n{rint(d54)}")
    lines.append("WOT Shift Speed – Normal 6th")
    lines.append(f"5→6\n{rint(up56)}"); lines.append(f"6→5\n{rint(d65)}")
    with open(Path(rd) / f"WOT_Shift_Speed__{tag}__ULTRA_PLUSH__BLOCKS.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="MULTILOG_RAW_WEIGHTED")
    ap.add_argument("--review-dir", default=r".\06_Logs\Trans_Review")
    ap.add_argument("--gap", type=float, default=6.0)
    ap.add_argument("--unlock-last-n", type=int, default=4)
    ap.add_argument("--temp-guard", type=float, default=None, help="Optional +2 mph on applies when active")
    # Slip
    ap.add_argument("--min", dest="vmin", type=float, default=5.0)
    ap.add_argument("--max", dest="vmax", type=float, default=30.0)
    ap.add_argument("--base-min", type=float, default=5.0)
    ap.add_argument("--base-max", type=float, default=20.0)
    ap.add_argument("--g1", type=float, default=20.0); ap.add_argument("--g2", type=float, default=12.0)
    ap.add_argument("--g3", type=float, default=0.0);  ap.add_argument("--g4", type=float, default=-2.0)
    ap.add_argument("--g5", type=float, default=-4.0); ap.add_argument("--g6", type=float, default=-6.0)
    args = ap.parse_args()

    rd = Path(args.review_dir); tag = args.tag
    up_p = rd / f"HPT_paste__16col__UPSHIFT__{tag}.tsv"
    dn_p = rd / f"HPT_paste__16col__DOWNSHIFT__{tag}.tsv"
    tcc_a_p = rd / f"tcc_apply__mph__{tag}.tsv"
    tcc_r_p = rd / f"tcc_release__mph__{tag}.tsv"
    for p in [up_p, dn_p, tcc_a_p, tcc_r_p]:
        if not p.exists():
            print(f"[ERROR] Missing {p.name} in {rd}")
            return 2

    # Read
    up_raw = read_tsv(up_p); dn_raw = read_tsv(dn_p)
    tcc_a_raw = read_tsv(tcc_a_p); tcc_r_raw = read_tsv(tcc_r_p)

    # Force labels (auto-fill if blank/missing)
    up_raw = force_labels(up_raw, ["1->2","2->3","3->4","4->5","5->6"])
    dn_raw = force_labels(dn_raw, ["2->1","3->2","4->3","5->4","6->5"])
    tcc_a_raw = force_labels(tcc_a_raw, ["1st Apply","2nd Apply","3rd Apply","4th Apply","5th Apply","6th Apply"])
    tcc_r_raw = force_labels(tcc_r_raw, ["1st Apply","2nd Apply","3rd Apply","4th Apply","5th Apply","6th Apply"])

    # Shift overlays
    up_out = overlay_upshift_plush(up_raw)

    ## SMART SIXTH (plush)
    # Allow 6th at low/mid TPS; effectively disable at high TPS by raising 5->6 speed.
    # Floors tuned for comfort/highway: gradually rising through 69% TPS; very high at 81%+.
    try:
        labels = up_out["mph"].tolist() if "mph" in up_out.columns else []
        if "5->6" in labels:
            r = labels.index("5->6")
            cols = [c for c in up_out.columns if c != "mph"]
            tps_values = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
            mph_floor = [58,58,60,62,64,66,68,70,72,75,78,84,94,200,200,200,200]
            floor_map = {str(t): m for t,m in zip(tps_values, mph_floor)}
            for c in cols:
                tps = int(float(c))
                cur = float(up_out.at[r, c])
                target = floor_map.get(str(tps), cur)
                # Take the safer, higher mph
                up_out.at[r, c] = max(cur, target)
    except Exception:
        pass
    
    dn_out = overlay_downshift_from_up(up_out, dn_raw, gap=args.gap)

    # TCC (TPS-shaped) with optional temp guard
    tcc_apply_out, tcc_release_out = build_tcc_plush(tcc_a_raw, tcc_r_raw, unlock_last_n=args.unlock_last_n, temp_guard=args.temp_guard)

    # Write overlay outputs
    write_tsv(up_out, rd / f"HPT_paste__16col__UPSHIFT__{tag}__ULTRA_PLUSH.tsv")
    write_tsv(dn_out, rd / f"HPT_paste__16col__DOWNSHIFT__{tag}__ULTRA_PLUSH.tsv")
    write_tsv(tcc_apply_out, rd / f"tcc_apply__mph__{tag}__ULTRA_PLUSH.tsv")
    write_tsv(tcc_release_out, rd / f"tcc_release__mph__{tag}__ULTRA_PLUSH.tsv")

    # WOT (caps from project) + downshifts with gap
    RATIOS = [4.03, 2.36, 1.53, 1.15, 0.85, 0.67]
    CAPS = {1:5800,2:5800,3:5600,4:5400,5:5400,6:5400}
    write_wot_blocks(rd, tag, args.gap, RATIOS, CAPS)

    # Slip 9×9 per gear (hard clamp 5–30; integers; monotone down torque rows)
    g_offsets = {1:args.g1,2:args.g2,3:args.g3,4:args.g4,5:args.g5,6:args.g6}
    write_slip_9x9(tag, rd, args.vmin, args.vmax, args.base_min, args.base_max, g_offsets)

    print(f"[OK] ULTRA_PLUSH v2 written to {rd} (UP/DOWN/TCC/WOT + slip 9x9, clamp [5,30])")

if __name__ == "__main__":
    raise SystemExit(main())
