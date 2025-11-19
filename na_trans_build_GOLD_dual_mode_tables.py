# na_trans_build_GOLD_dual_mode_tables.py
# Build synthetic "Gold" SHIFT & TCC tables for 6L80 Tahoe (3.08 FD, 32.5" tire).
# Physics + TPS-aware feel targets (Comfort vs Performance). No reuse of old tune/logs.

import os, math, argparse
from collections import OrderedDict

# ===== Vehicle / Driveline =====
FD = 3.08
TIRE_DIAM_IN = 32.5
GEAR = {1:4.03, 2:2.36, 3:1.53, 4:1.15, 5:0.85, 6:0.67}
TPS_AXIS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]

# WOT caps by FROM-gear (caps the 100% column safety-wise)
WOT_CAP_RPM = {1:5800, 2:5800, 3:5600, 4:5400, 5:5400}

# ===== Feel "knobs" (Tahoe-aware TPS anchoring) =====
# Post-shift RPM floors vs TPS (Comfort = plush; Perf = rowdier)
COMFORT_POST_RPM_FLOOR = [
    (0,1200),(6,1250),(12,1300),(19,1350),(25,1400),(31,1450),
    (37,1500),(44,1600),(50,1700),(56,1800),(62,1900),(69,2050),
    (75,2200),(81,2350),(87,2500),(94,2700),(100,3000)
]
PERF_POST_RPM_FLOOR = [
    (0,1400),(6,1500),(12,1600),(19,1700),(25,1800),(31,1900),
    (37,2100),(44,2300),(50,2500),(56,2700),(62,2900),(69,3200),
    (75,3500),(81,3800),(87,4000),(94,4200),(100,4400)
]

# Downshift gaps grow with TPS (calm at light TPS, decisive under load)
COMFORT_DN_GAP = [(0,6),(31,6),(62,8),(81,9),(100,10)]
PERF_DN_GAP    = [(0,5),(31,5),(62,7),(81,8),(100,8)]

# EC³ TCC floors (Convert to mph per gear); high-TPS lockout & hysteresis
COMFORT_TCC_RPM = [
    (0,1350),(12,1400),(19,1450),(25,1500),(31,1600),
    (37,1700),(44,1800),(50,1850),(56,1900),(62,2000),
    (69,2100),(75,2200),(81,2300),(87,2400),(94,2500),(100,2500)
]
PERF_TCC_RPM = [
    (0,1500),(12,1600),(19,1700),(25,1800),(31,1900),
    (37,2100),(44,2300),(50,2500),(56,2700),(62,2900),
    (69,3200),(75,3500),(81,3800),(87,4000),(94,4200),(100,4200)
]
COMFORT_TCC_LOCKOUT_TPS_MIN = 81   # Apply=318, Release=317 for TPS ≥ 81
PERF_TCC_LOCKOUT_TPS_MIN    = 94   # Apply=318, Release=317 for TPS ≥ 94
COMFORT_TCC_REL_DELTA = 4.0        # Release = Apply + Δ (mph)
PERF_TCC_REL_DELTA    = 3.0

# ===== Helpers =====
def circ_miles(diam_in): return (math.pi * diam_in / 12.0) / 5280.0
CIRC_MI = circ_miles(TIRE_DIAM_IN)

def mph_at_rpm(rpm, gear):
    wheel_rpm = rpm / (FD * GEAR[gear])
    return wheel_rpm * CIRC_MI * 60.0

def wot_mph_cap(from_gear):
    rpm = WOT_CAP_RPM.get(from_gear)
    return mph_at_rpm(rpm, from_gear) if rpm else 9e9

def lerp_map(x, pts):
    for i in range(len(pts)-1):
        x0,y0 = pts[i]; x1,y1 = pts[i+1]
        if x <= x0: return y0
        if x >= x1: continue
        t = (x - x0) / (x1 - x0)
        return y0 + t*(y1 - y0)
    return pts[-1][1]

def ensure_cross_gear_monotonic(up_map):
    # enforce 1->2 < 2->3 < ... < 5->6 at every TPS
    for col in range(len(TPS_AXIS)):
        prev = -1e9
        for pair in [(1,2),(2,3),(3,4),(4,5),(5,6)]:
            v = up_map[pair][col]
            if v <= prev:
                v = round(prev + 0.1, 1)
                up_map[pair][col] = v
            prev = v

def gaps_for_tps(gap_pts): return [round(lerp_map(t, gap_pts),1) for t in TPS_AXIS]
def make_down(up_vals, gaps): return [ round(min(u - g, u - 1.1),1) for u,g in zip(up_vals,gaps) ]

def write_shift(path, rows_map, up=True):
    hdr = "mph\t" + "\t".join(str(t) for t in TPS_AXIS) + "\t%"
    with open(path,"w",newline="") as f:
        f.write(hdr+"\n")
        order = [(1,2),(2,3),(3,4),(4,5),(5,6)] if up else [(2,1),(3,2),(4,3),(5,4),(6,5)]
        for (fg,tg) in order:
            lbl = f"{fg} -> {tg} Shift"
            vals = rows_map[(fg,tg)] if up else rows_map[(tg,fg)]
            f.write(lbl + "\t" + "\t".join(f"{v:.1f}" for v in vals) + "\t\n")

def ordinal(n): return {1:"1st",2:"2nd",3:"3rd",4:"4th",5:"5th",6:"6th"}.get(n, f"{n}th")
def write_tcc(path, label, rows):
    hdr = "mph\t" + "\t".join(str(t) for t in TPS_AXIS) + "\t%"
    with open(path,"w",newline="") as f:
        f.write(hdr+"\n")
        for gear in [3,4,5,6]:
            name = f"{ordinal(gear)} {label}"
            f.write(name + "\t" + "\t".join(f"{v:.1f}" for v in rows[gear]) + "\t\n")

# ===== SHIFT builders =====
def build_shift_up(floor_pts):
    up = OrderedDict()
    for (fg,tg) in [(1,2),(2,3),(3,4),(4,5),(5,6)]:
        cap = round(wot_mph_cap(fg), 1)
        vals=[]
        for tps in TPS_AXIS:
            post = lerp_map(tps, floor_pts)          # desired rpm AFTER shift
            mph_to = round(mph_at_rpm(post, tg), 1)  # mph that yields that rpm in TO gear
            vals.append(min(mph_to, cap))
        up[(fg,tg)] = vals
    ensure_cross_gear_monotonic(up)
    return up

def build_shift_mode(mode_name, floor_pts, dn_gap_pts, out_root):
    shift_dir = os.path.join(out_root, mode_name, "shift"); os.makedirs(shift_dir, exist_ok=True)
    up = build_shift_up(floor_pts)
    gaps = gaps_for_tps(dn_gap_pts)
    down = OrderedDict()
    for pair,vals in up.items():
        down[pair] = make_down(vals, gaps)
    write_shift(os.path.join(shift_dir,"SHIFT_TABLES__UP__Throttle17.tsv"), up, up=True)
    write_shift(os.path.join(shift_dir,"SHIFT_TABLES__DOWN__Throttle17.tsv"), down, up=False)
    return up,down

# ===== TCC builders =====
def build_tcc_apply(gear, rpm_pts, lockout_min_tps):
    vals=[]
    for tps in TPS_AXIS:
        if tps >= lockout_min_tps:
            vals.append(318.0)  # lockout sentinel
        else:
            rpm = lerp_map(tps, rpm_pts)
            vals.append(round(mph_at_rpm(rpm, gear),1))
    return vals

def build_tcc_mode(mode_name, rpm_pts, lockout_min_tps, rel_delta, out_root):
    tcc_dir = os.path.join(out_root, mode_name, "tcc"); os.makedirs(tcc_dir, exist_ok=True)
    apply_rows = {1:[318.0]*len(TPS_AXIS), 2:[318.0]*len(TPS_AXIS)}
    release_rows= {1:[317.0]*len(TPS_AXIS), 2:[317.0]*len(TPS_AXIS)}
    for g in [3,4,5,6]:
        a = build_tcc_apply(g, rpm_pts, lockout_min_tps)
        r = [ (317.0 if v>=318.0 else round(v + rel_delta,1)) for v in a ]
        apply_rows[g]  = a
        release_rows[g]= r
    write_tcc(os.path.join(tcc_dir,"TCC_APPLY__Throttle17.tsv"),  "Apply",   apply_rows)
    write_tcc(os.path.join(tcc_dir,"TCC_RELEASE__Throttle17.tsv"),"Release", release_rows)
    return apply_rows, release_rows

# ===== MAIN =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True,
        help=r"Root of 01_tables (e.g., C:\tuning\na-trans-data\newlogs\output\01_tables)")
    args = ap.parse_args()

    # COMFORT_GOLD
    c_up, c_dn = build_shift_mode("COMFORT_GOLD", COMFORT_POST_RPM_FLOOR, COMFORT_DN_GAP, args.out_root)
    c_app, c_rel = build_tcc_mode("COMFORT_GOLD", COMFORT_TCC_RPM, COMFORT_TCC_LOCKOUT_TPS_MIN, COMFORT_TCC_REL_DELTA, args.out_root)

    # PERF_GOLD
    p_up, p_dn = build_shift_mode("PERF_GOLD",    PERF_POST_RPM_FLOOR,    PERF_DN_GAP,    args.out_root)
    p_app, p_rel = build_tcc_mode("PERF_GOLD",    PERF_TCC_RPM,           PERF_TCC_LOCKOUT_TPS_MIN,    PERF_TCC_REL_DELTA, args.out_root)

    # Quick previews (TPS 0,50,75,100)
    keep_cols = [0,8,12,16]
    def pick(row): return [f"{row[i]:.1f}" for i in keep_cols]
    print("\\n[COMFORT_GOLD] SHIFT UP preview (TPS 0,50,75,100)")
    for pair in [(1,2),(2,3),(3,4),(4,5),(5,6)]: print(f"{pair[0]}->{pair[1]}:", pick(c_up[pair]))
    print("\\n[PERF_GOLD]    SHIFT UP preview (TPS 0,50,75,100)")
    for pair in [(1,2),(2,3),(3,4),(4,5),(5,6)]: print(f"{pair[0]}->{pair[1]}:", pick(p_up[pair]))
    print("\\n[COMFORT_GOLD] TCC APPLY preview (TPS 0,50,75,100)")
    for g in [3,4,5,6]: print(f"{g}:", pick(c_app[g]))
    print("\\n[PERF_GOLD]    TCC APPLY preview (TPS 0,50,75,100)")
    for g in [3,4,5,6]: print(f"{g}:", pick(p_app[g]))

if __name__ == "__main__":
    main()
