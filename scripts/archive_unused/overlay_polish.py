# overlay_polish.py
# Applies post-multilog overlays to SHIFT & TCC tables (Throttle17; mph…%)
# Policy: gentle torque-first bias, monotonic TPS, and minimum hysteresis gaps.
# Inputs (in --dir): SHIFT_TABLES__UP__Throttle17.tsv, SHIFT_TABLES__DOWN__Throttle17.tsv,
#                    TCC_APPLY__Throttle17.tsv, TCC_RELEASE__Throttle17.tsv
# Outputs: *_Throttle17__OVERLAY.tsv (originals untouched)

import argparse, os, math

TPS_AXIS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
HEADER = ["mph"] + [str(x) for x in TPS_AXIS] + ["%"]

def read_tsv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f.read().splitlines():
            parts = ln.split("\t")
            rows.append(parts)
    if rows and rows[0] != HEADER:
        raise ValueError(f"{os.path.basename(path)}: header must match Throttle17 'mph … %'")
    data = []
    for r in rows[1:]:
        label = r[0]
        vals = []
        for s in r[1:-1]:
            if s.strip()=="":
                vals.append(float("nan"))
            else:
                try:
                    vals.append(float(s))
                except:
                    vals.append(float("nan"))
        data.append((label, vals))
    return data

def write_tsv(path, data):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(HEADER) + "\n")
        for label, vals in data:
            out = [label] + [("" if (v is None or math.isnan(v)) else f"{v:.1f}") for v in vals] + ["%"]
            f.write("\t".join(out) + "\n")

def fill_and_clip(vals, lo=0.0, hi=140.0):
    xs = vals[:]
    last = None
    for i,v in enumerate(xs):
        if v is None or math.isnan(v):
            if last is not None:
                xs[i] = last
        else:
            last = v
    last = None
    for i in range(len(xs)-1, -1, -1):
        v = xs[i]
        if v is None or math.isnan(v):
            if last is not None:
                xs[i] = last
        else:
            last = v
    for i,v in enumerate(xs):
        if v is None or math.isnan(v):
            continue
        xs[i] = max(lo, min(hi, v))
    return xs

def ensure_monotonic_nondec(vals):
    out = vals[:]
    for i in range(1,len(out)):
        if out[i] is None or math.isnan(out[i]):
            continue
        if out[i-1] is None or math.isnan(out[i-1]):
            out[i-1] = out[i]
        out[i] = max(out[i], out[i-1])
    return out

def apply_bias_curve(vals, curve):
    out = vals[:]
    for i,tps in enumerate(TPS_AXIS):
        if out[i] is None or math.isnan(out[i]):
            continue
        if tps in curve:
            delta = curve[tps]
        else:
            left = max([x for x in curve.keys() if x < tps], default=min(curve.keys()))
            right = min([x for x in curve.keys() if x > tps], default=max(curve.keys()))
            if right==left:
                delta = curve[left]
            else:
                frac = (tps-left)/(right-left)
                delta = curve[left]*(1-frac) + curve[right]*frac
        out[i] += delta
    return out

def enforce_hysteresis(down_vals, up_vals, gap_curve):
    out = down_vals[:]
    for i,tps in enumerate(TPS_AXIS):
        if (up_vals[i] is None or math.isnan(up_vals[i])):
            continue
        keys = sorted(gap_curve.keys())
        gap = gap_curve.get(tps, None)
        if gap is None:
            lower_keys = [k for k in keys if k <= tps]
            gap = gap_curve[lower_keys[-1]] if lower_keys else gap_curve[keys[0]]
        target_max = up_vals[i] - gap
        if out[i] is None or math.isnan(out[i]):
            out[i] = target_max
        else:
            out[i] = min(out[i], target_max)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Folder with *Throttle17.tsv from multilog builders")
    ap.add_argument("--profile", default="NA_torque_first", choices=["NA_torque_first"])
    args = ap.parse_args()

    up_p   = os.path.join(args.dir, "SHIFT_TABLES__UP__Throttle17.tsv")
    dn_p   = os.path.join(args.dir, "SHIFT_TABLES__DOWN__Throttle17.tsv")
    tcca_p = os.path.join(args.dir, "TCC_APPLY__Throttle17.tsv")
    tccr_p = os.path.join(args.dir, "TCC_RELEASE__Throttle17.tsv")

    for p in (up_p, dn_p, tcca_p, tccr_p):
        if not os.path.exists(p):
            raise SystemExit(f"Missing input TSV: {p}")

    up_rows   = read_tsv(up_p)
    dn_rows   = read_tsv(dn_p)
    tcca_rows = read_tsv(tcca_p)
    tccr_rows = read_tsv(tccr_p)

    bias_curve = {0:0.0, 12:0.5, 19:1.0, 25:1.8, 31:2.2, 37:2.5, 44:2.5, 50:2.5, 56:2.5, 62:2.2, 69:1.8, 75:1.2, 81:0.7, 87:0.4, 94:0.2, 100:0.0}
    shift_gap = {0:2.0, 12:2.0, 19:2.5, 25:3.0, 31:3.0, 37:3.5, 44:3.5, 50:4.0, 56:4.0, 62:4.0, 69:4.5, 75:4.5, 81:5.0, 87:5.0, 94:5.0, 100:5.0}
    tcc_gap   = {0:1.0, 12:1.0, 19:1.5, 25:2.0, 31:2.0, 37:2.0, 44:2.0, 50:2.5, 56:2.5, 62:2.5, 69:3.0, 75:3.0, 81:3.0, 87:3.0, 94:3.0, 100:3.0}

    up_out = []
    for label, vals in up_rows:
        v = fill_and_clip(vals)
        v = apply_bias_curve(v, bias_curve)
        v = ensure_monotonic_nondec(v)
        up_out.append((label, v))

    dn_map = {label: vals for label, vals in dn_rows}
    dn_out = []
    for label, up_vals in up_out:
        if label in dn_map:
            vdn = fill_and_clip(dn_map[label])
            vdn = enforce_hysteresis(vdn, up_vals, shift_gap)
            vdn = ensure_monotonic_nondec(vdn)
            dn_out.append((label, vdn))

    tcc_bias = {0:0.0, 12:0.2, 19:0.4, 25:0.6, 31:0.8, 37:1.0, 44:1.2, 50:1.4, 56:1.6, 62:1.6, 69:1.6, 75:1.4, 81:1.0, 87:0.6, 94:0.3, 100:0.0}
    tcca_out = []
    for label, vals in tcca_rows:
        v = fill_and_clip(vals)
        v = apply_bias_curve(v, tcc_bias)
        v = ensure_monotonic_nondec(v)
        tcca_out.append((label, v))

    def _apply_row_for_release(label):
        return label.replace(" Release"," Apply")
    tcca_map = {label: vals for label, vals in tcca_out}
    tccr_out = []
    for label, vals in tccr_rows:
        v = fill_and_clip(vals)
        ref = tcca_map.get(_apply_row_for_release(label))
        if ref:
            v = enforce_hysteresis(v, ref, tcc_gap)
        v = ensure_monotonic_nondec(v)
        tccr_out.append((label, v))

    out_up   = os.path.join(args.dir, "SHIFT_TABLES__UP__Throttle17__OVERLAY.tsv")
    out_dn   = os.path.join(args.dir, "SHIFT_TABLES__DOWN__Throttle17__OVERLAY.tsv")
    out_tcca = os.path.join(args.dir, "TCC_APPLY__Throttle17__OVERLAY.tsv")
    out_tccr = os.path.join(args.dir, "TCC_RELEASE__Throttle17__OVERLAY.tsv")

    write_tsv(out_up,   up_out)
    write_tsv(out_dn,   dn_out)
    write_tsv(out_tcca, tcca_out)
    write_tsv(out_tccr, tccr_out)

    print("[OK] Overlay wrote:")
    print(" ", out_up)
    print(" ", out_dn)
    print(" ", out_tcca)
    print(" ", out_tccr)

if __name__ == "__main__":
    main()
