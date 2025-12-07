# blend_tables.py (patched)
# Blend multilog overlay tables with a logless pack (Throttle17 format).
# Robust to DOWN label styles ("2 -> 1 Shift" vs "1 -> 2 Shift").
import argparse, os, math, re

TPS_AXIS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
HEADER = ["mph"] + [str(x) for x in TPS_AXIS] + ["%"]

DATA_W_BY_TPS = {
  0:0.40, 6:0.50, 12:0.65, 19:0.70, 25:0.75, 31:0.80, 37:0.80, 44:0.80,
  50:0.80, 56:0.80, 62:0.75, 69:0.70, 75:0.60, 81:0.55, 87:0.50, 94:0.45, 100:0.40
}

shift_re = re.compile(r'^\s*(\d)\s*->\s*(\d)\s*Shift\s*$', re.I)

def read_tsv(path):
    rows = []
    with open(path,"r",encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines: raise ValueError(f"{os.path.basename(path)} is empty")
    hdr = lines[0].split("\t")
    if hdr != HEADER:
        raise ValueError(f"{os.path.basename(path)}: header must be Throttle17 'mph … %'")
    for ln in lines[1:]:
        parts = ln.split("\t")
        label = parts[0].strip()
        vals = []
        for s in parts[1:-1]:
            s = s.strip()
            if s=="":
                vals.append(float("nan"))
            else:
                try: vals.append(float(s))
                except: vals.append(float("nan"))
        rows.append((label, vals))
    return rows

def write_tsv(path, data):
    with open(path,"w",encoding="utf-8") as f:
        f.write("\t".join(HEADER)+"\n")
        for label, vals in data:
            out = [label] + [("" if (v is None or (isinstance(v,float) and math.isnan(v))) else f"{v:.1f}") for v in vals] + ["%"]
            f.write("\t".join(out)+"\n")

def ensure_monotonic(vals):
    out = vals[:]
    for i in range(1,len(out)):
        if out[i] < out[i-1]:
            out[i] = out[i-1]
    return out

def enforce_hysteresis(down_vals, up_vals, gap_floor=1.0):
    out = down_vals[:]
    for i in range(len(TPS_AXIS)):
        uv = up_vals[i]
        dv = out[i]
        if uv==uv:
            tgt = uv - gap_floor
            if (dv!=dv) or (dv>tgt):
                out[i] = tgt
    return ensure_monotonic(out)

def blend_row(data_vals, logless_vals, row_name, data_weight_bias=0.0):
    out = []
    for i,tps in enumerate(TPS_AXIS):
        d = data_vals[i]; l = logless_vals[i]
        if d!=d and l!=l:
            out.append(float('nan')); continue
        if d!=d: out.append(l); continue
        if l!=l: out.append(d); continue
        w = DATA_W_BY_TPS[tps]
        if "->" in row_name:
            if row_name.startswith("4"): w = min(1.0, w+0.05)
            if row_name.startswith("5"): w = min(1.0, w+0.10)
        else:
            if row_name.startswith("5") or row_name.startswith("6"):
                w = min(1.0, w+0.05)
        w = min(1.0, max(0.0, w + data_weight_bias))
        out.append(w*d + (1.0-w)*l)
    return ensure_monotonic(out)

def up_key_for_down(down_label):
    m = shift_re.match(down_label)
    if not m: return down_label  # not a shift label
    a,b = m.group(1), m.group(2)
    # Map "2 -> 1 Shift" -> "1 -> 2 Shift"
    return f"{b} -> {a} Shift"

def build_dual_key_map(rows):
    """Return a dict mapping both UP-style and DOWN-style keys to the same row values."""
    m = {}
    for lbl, vals in rows:
        m[lbl] = vals
        mm = shift_re.match(lbl)
        if mm:
            a,b = mm.group(1), mm.group(2)
            # add reverse key as well
            rev = f"{b} -> {a} Shift"
            m[rev] = vals
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay-dir", required=True, help="Folder with multilog __OVERLAY.tsv files")
    ap.add_argument("--logless-dir", required=True, help="Folder with logless TSVs")
    ap.add_argument("--out-dir", required=True, help="Output folder for blended TSVs")
    ap.add_argument("--data-weight-bias", type=float, default=0.0, help="Shift all TPS weights by this amount (-0.2..+0.2)")
    ap.add_argument("--down-gap-floor", type=float, default=1.6, help="Minimum mph hysteresis DOWN vs UP after blend")
    ap.add_argument("--tcc-gap-floor", type=float, default=1.1, help="Minimum mph gap RELEASE vs APPLY after blend")
    args = ap.parse_args()

    def overlay_path(name):
        return os.path.join(args.overlay_dir, f"{name}__Throttle17__OVERLAY.tsv")
    up_p   = overlay_path("SHIFT_TABLES__UP")
    dn_p   = overlay_path("SHIFT_TABLES__DOWN")
    tcca_p = overlay_path("TCC_APPLY")
    tccr_p = overlay_path("TCC_RELEASE")

    lup_p  = os.path.join(args.logless_dir, "SHIFT_TABLES__UP__Throttle17.tsv")
    ldn_p  = os.path.join(args.logless_dir, "SHIFT_TABLES__DOWN__Throttle17.tsv")
    ltca_p = os.path.join(args.logless_dir, "TCC_APPLY__Throttle17.tsv")
    ltcr_p = os.path.join(args.logless_dir, "TCC_RELEASE__Throttle17.tsv")

    up_rows   = read_tsv(up_p)
    dn_rows   = read_tsv(dn_p)
    tcca_rows = read_tsv(tcca_p)
    tccr_rows = read_tsv(tccr_p)
    lup_rows  = read_tsv(lup_p)
    ldn_rows  = read_tsv(ldn_p)
    ltca_rows = read_tsv(ltca_p)
    ltcr_rows = read_tsv(ltcr_p)

    # Dual-key maps allow looking up by either "1 -> 2 Shift" or "2 -> 1 Shift"
    up_map    = build_dual_key_map(up_rows)
    dn_map    = build_dual_key_map(dn_rows)
    lup_map   = build_dual_key_map(lup_rows)
    ldn_map   = build_dual_key_map(ldn_rows)
    tcca_map  = {lbl:vals for lbl,vals in tcca_rows}
    tccr_map  = {lbl:vals for lbl,vals in tccr_rows}
    ltca_map  = {lbl:vals for lbl,vals in ltca_rows}
    ltcr_map  = {lbl:vals for lbl,vals in ltcr_rows}

    up_out, dn_out, tcca_out, tccr_out = [], [], [], []

    # UP blending (labels as-is)
    for lbl, dvals in up_rows:
        up_out.append((lbl, blend_row(dvals, lup_map.get(lbl, [float('nan')]*len(TPS_AXIS)), lbl, args.data_weight_bias)))

    # DOWN blending: allow DOWN labels like "2 -> 1 Shift", but enforce hysteresis vs corresponding UP
    for lbl, dvals in dn_rows:
        # For logless reference, use same-form key if present, else try reverse
        lvals = ldn_map.get(lbl)
        if lvals is None:
            lvals = ldn_map.get(up_key_for_down(lbl), [float('nan')]*len(TPS_AXIS))
        dn_vals = blend_row(dvals, lvals, lbl, args.data_weight_bias)
        up_label = up_key_for_down(lbl)  # map to UP side for hysteresis
        up_vals = dict(up_out).get(up_label)
        if up_vals is None:
            up_vals = up_map.get(up_label, [float('nan')]*len(TPS_AXIS))
        dn_vals = enforce_hysteresis(dn_vals, up_vals, gap_floor=args.down_gap_floor)
        dn_out.append((lbl, dn_vals))

    # TCC
    for lbl, dvals in tcca_rows:
        tcca_out.append((lbl, blend_row(dvals, ltca_map.get(lbl, [float('nan')]*len(TPS_AXIS)), lbl, args.data_weight_bias)))
    for lbl, dvals in tccr_rows:
        rel_vals = blend_row(dvals, ltcr_map.get(lbl, [float('nan')]*len(TPS_AXIS)), lbl, args.data_weight_bias)
        app_vals = dict(tcca_out).get(lbl.replace("Release","Apply"), tcca_map.get(lbl.replace("Release","Apply"), [float('nan')]*len(TPS_AXIS)))
        rel_vals = enforce_hysteresis(rel_vals, app_vals, gap_floor=args.tcc_gap_floor)
        tccr_out.append((lbl, rel_vals))

    os.makedirs(args.out_dir, exist_ok=True)
    def w(name, rows):
        path = os.path.join(args.out_dir, f"{name}__Throttle17__OVERLAY__BLENDED.tsv")
        write_tsv(path, rows); return path
    for name, rows in [("SHIFT_TABLES__UP", up_out),
                       ("SHIFT_TABLES__DOWN", dn_out),
                       ("TCC_APPLY", tcca_out),
                       ("TCC_RELEASE", tccr_out)]:
        print("[OK] wrote", w(name, rows))

if __name__ == "__main__":
    main()
