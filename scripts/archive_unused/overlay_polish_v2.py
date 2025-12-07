# overlay_polish_v2.py — tolerant headers/labels, always writes 4 overlays
import argparse, os, math, re

TPS_AXIS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
HEADER_OK = ["mph"] + [str(x) for x in TPS_AXIS] + ["%"]

def _norm_header(cells):
    # strip, collapse spaces
    return [re.sub(r"\s+"," ",x.strip()) for x in cells]

def _parse_header(ln):
    cells = ln.rstrip("\n\r").split("\t")
    cells = _norm_header(cells)
    # accept exact or forgiving match (mph / numbers / %)
    if len(cells) != len(HEADER_OK): return None
    if cells[0].lower() != "mph": return None
    if cells[-1] != "%": return None
    # middle must be the TPS axis, but allow e.g. " 50 " etc.
    try:
        mids = [int(float(x)) for x in cells[1:-1]]
    except:
        return None
    if mids != TPS_AXIS: return None
    return cells

def read_tsv(path):
    rows = []
    with open(path,"r",encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines: raise ValueError(f"{os.path.basename(path)} is empty")
    if _parse_header(lines[0]) is None:
        raise ValueError(f"{os.path.basename(path)} bad/mismatched header (expect 17-pt TPS axis + mph … %)")
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
        f.write("\t".join(HEADER_OK)+"\n")
        for label, vals in data:
            out = [label] + [("" if (v is None or math.isnan(v)) else f"{v:.1f}") for v in vals] + ["%"]
            f.write("\t".join(out)+"\n")

def fill_and_clip(vals, lo=0.0, hi=140.0):
    x = vals[:]
    last=None
    for i,v in enumerate(x):
        if v!=v:  # NaN
            if last is not None: x[i]=last
        else:
            last=v
    last=None
    for i in range(len(x)-1,-1,-1):
        v=x[i]
        if v!=v:
            if last is not None: x[i]=last
        else:
            last=v
    for i,v in enumerate(x):
        if v==v:
            x[i]=max(lo,min(hi,v))
    return x

def ensure_monotonic_nondec(vals):
    out = vals[:]
    for i in range(1,len(out)):
        a,b = out[i-1], out[i]
        if b!=b: continue
        if a!=a: out[i-1]=b
        out[i] = max(out[i], out[i-1])
    return out

def apply_bias_curve(vals, curve):
    out = vals[:]
    keys = sorted(curve.keys())
    for i,tps in enumerate(TPS_AXIS):
        v = out[i]
        if v!=v: continue
        if tps in curve: delta = curve[tps]
        else:
            # linear interp
            L = max([k for k in keys if k<=tps], default=keys[0])
            R = min([k for k in keys if k>=tps], default=keys[-1])
            if L==R: delta = curve[L]
            else:
                frac=(tps-L)/(R-L)
                delta = curve[L]*(1-frac)+curve[R]*frac
        out[i]=v+delta
    return out

def enforce_hysteresis(down_vals, up_vals, gap_curve):
    out = down_vals[:]
    keys = sorted(gap_curve.keys())
    for i,tps in enumerate(TPS_AXIS):
        uv = up_vals[i]
        if uv!=uv: continue
        gap = gap_curve.get(tps, gap_curve[keys[0]])
        target = uv - gap
        dv = out[i]
        if dv!=dv: out[i]=target
        else:      out[i]=min(dv,target)
    return out

def ord_key(label):
    # normalize row labels like "1st Apply", "1st Release", "1 -> 2 Shift"
    s = label.strip()
    # For SHIFT rows, keep full label match later
    if "Shift" in s: return s
    m = re.match(r"(\d+)(st|nd|rd|th)", s)
    if m: return m.group(1)  # "1", "2", etc.
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    up_p   = os.path.join(args.dir,"SHIFT_TABLES__UP__Throttle17.tsv")
    dn_p   = os.path.join(args.dir,"SHIFT_TABLES__DOWN__Throttle17.tsv")
    tcca_p = os.path.join(args.dir,"TCC_APPLY__Throttle17.tsv")
    tccr_p = os.path.join(args.dir,"TCC_RELEASE__Throttle17.tsv")

    up_rows   = read_tsv(up_p)
    dn_rows   = read_tsv(dn_p)
    tcca_rows = read_tsv(tcca_p)
    tccr_rows = read_tsv(tccr_p)

    # overlay profiles
    bias_curve = {0:0.0,12:0.5,19:1.0,25:1.8,31:2.2,37:2.5,44:2.5,50:2.5,56:2.5,62:2.2,69:1.8,75:1.2,81:0.7,87:0.4,94:0.2,100:0.0}
    shift_gap = {0:2.0,12:2.0,19:2.5,25:3.0,31:3.0,37:3.5,44:3.5,50:4.0,56:4.0,62:4.0,69:4.5,75:4.5,81:5.0,87:5.0,94:5.0,100:5.0}
    tcc_bias  = {0:0.0,12:0.2,19:0.4,25:0.6,31:0.8,37:1.0,44:1.2,50:1.4,56:1.6,62:1.6,69:1.6,75:1.4,81:1.0,87:0.6,94:0.3,100:0.0}
    tcc_gap   = {0:1.0,12:1.0,19:1.5,25:2.0,31:2.0,37:2.0,44:2.0,50:2.5,56:2.5,62:2.5,69:3.0,75:3.0,81:3.0,87:3.0,94:3.0,100:3.0}

    # SHIFT UP
    up_out=[]
    for label,vals in up_rows:
        v = ensure_monotonic_nondec(apply_bias_curve(fill_and_clip(vals), bias_curve))
        up_out.append((label,v))

    # SHIFT DOWN — enforce hysteresis vs same label as UP, tolerate label spacing
    dn_map = {lbl.strip():vals for lbl,vals in dn_rows}
    dn_out=[]
    for label, up_vals in up_out:
        base = label.strip()
        if base in dn_map:
            vdn = ensure_monotonic_nondec(enforce_hysteresis(fill_and_clip(dn_map[base]), up_vals, shift_gap))
            dn_out.append((label, vdn))

    # TCC APPLY
    tcca_out=[]
    for label,vals in tcca_rows:
        v = ensure_monotonic_nondec(apply_bias_curve(fill_and_clip(vals), tcc_bias))
        tcca_out.append((label,v))

    # TCC RELEASE — match by ordinal key (e.g., "1st" → "1")
    tcca_by_ord = {}
    for lbl,vals in tcca_out:
        k = ord_key(lbl)  # "1"
        tcca_by_ord.setdefault(k, vals)
    tccr_out=[]
    for lbl,vals in tccr_rows:
        k = ord_key(lbl)
        v = fill_and_clip(vals)
        ref = tcca_by_ord.get(k)
        if ref:
            v = enforce_hysteresis(v, ref, tcc_gap)
        v = ensure_monotonic_nondec(v)
        tccr_out.append((lbl,v))

    # write
    out_up   = os.path.join(args.dir,"SHIFT_TABLES__UP__Throttle17__OVERLAY.tsv")
    out_dn   = os.path.join(args.dir,"SHIFT_TABLES__DOWN__Throttle17__OVERLAY.tsv")
    out_tcca = os.path.join(args.dir,"TCC_APPLY__Throttle17__OVERLAY.tsv")
    out_tccr = os.path.join(args.dir,"TCC_RELEASE__Throttle17__OVERLAY.tsv")
    for p,d in [(out_up,up_out),(out_dn,dn_out),(out_tcca,tcca_out),(out_tccr,tccr_out)]:
        write_tsv(p,d)
        print("[OK] wrote", p, "rows:", len(d))

if __name__ == "__main__":
    main()
