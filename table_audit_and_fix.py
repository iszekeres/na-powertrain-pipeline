#!/usr/bin/env python3
import argparse, os, pandas as pd, numpy as np

TPS_CANON = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]

def _read(path): return pd.read_csv(path, sep='\t')
def _tps_cols(df):
    cols = df.columns.tolist()
    return cols[1:-1] if cols[-1] == '%' else cols[1:]

def _round01(df, tps_cols):
    num = df[tps_cols].apply(pd.to_numeric, errors='coerce')
    num = num.applymap(lambda x: round(x,1) if pd.notna(x) else x)
    df[tps_cols] = num
    return df

def audit_shift(up_p, dn_p, out_dir):
    up = _read(up_p); dn = _read(dn_p)
    up.columns = ['mph'] + list(map(str, TPS_CANON)) + (['%'] if up.columns[-1]=='%' else [])
    dn.columns = ['mph'] + list(map(str, TPS_CANON)) + (['%'] if dn.columns[-1]=='%' else [])
    up = _round01(up, _tps_cols(up)); dn = _round01(dn, _tps_cols(dn))
    tps_cols = _tps_cols(up)
    upv = up[tps_cols].apply(pd.to_numeric, errors='coerce').values
    dnv = dn[tps_cols].apply(pd.to_numeric, errors='coerce').values
    dnv = np.minimum(dnv, upv - 1.0)
    dn[tps_cols] = dnv
    oup = os.path.join(out_dir, os.path.basename(up_p)); odn = os.path.join(out_dir, os.path.basename(dn_p))
    up.to_csv(oup, sep='\t', index=False); dn.to_csv(odn, sep='\t', index=False)
    print(f'[AUDIT] wrote {oup} and {odn}')

def audit_tcc(apply_p, release_p, out_dir):
    ap = _read(apply_p); rp = _read(release_p)
    ap.columns = ['mph'] + list(map(str, TPS_CANON)) + (['%'] if ap.columns[-1]=='%' else [])
    rp.columns = ['mph'] + list(map(str, TPS_CANON)) + (['%'] if rp.columns[-1]=='%' else [])
    tps_cols = _tps_cols(ap)
    def to_num_or_keep(x):
        try: return float(x)
        except: return x
    apv = ap[tps_cols].applymap(to_num_or_keep)
    rpv = rp[tps_cols].applymap(to_num_or_keep)
    ap_mask = apv.applymap(lambda x: isinstance(x, (int,float)) and x < 317.0)
    rp_mask = rpv.applymap(lambda x: isinstance(x, (int,float)) and x < 317.0)
    ap_num = apv.where(ap_mask, np.nan).astype(float).round(1)
    rp_num = rpv.where(rp_mask, np.nan).astype(float).round(1)
    rp_num = np.maximum(rp_num, ap_num + 1.1)
    for c in tps_cols:
        ap[c] = np.where(ap_mask[c], ap_num[c], apv[c])
        rp[c] = np.where(rp_mask[c], rp_num[c], rpv[c])
    oap = os.path.join(out_dir, os.path.basename(apply_p)); orp = os.path.join(out_dir, os.path.basename(release_p))
    ap.to_csv(oap, sep='\t', index=False); rp.to_csv(orp, sep='\t', index=False)
    print(f'[AUDIT] wrote {oap} and {orp}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--up'); ap.add_argument('--down')
    ap.add_argument('--tcc-apply'); ap.add_argument('--tcc-release')
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    if a.up and a.down: audit_shift(a.up, a.down, a.out_dir)
    if a.tcc_apply and a.tcc_release: audit_tcc(a.tcc_apply, a.tcc_release, a.out_dir)

if __name__ == '__main__':
    main()
# --- AUTO CLAMP APPEND (DOWN ≤ UP − 1.0 mph; preserves 317/318) ---
try:
    import os, sys, numpy as np, pandas as pd  # reuse modules already present
    # Find --out-dir in argv
    out_dir = None
    for i, a in enumerate(sys.argv):
        if a == '--out-dir' and i + 1 < len(sys.argv):
            out_dir = sys.argv[i + 1]
            break
    if out_dir and os.path.isdir(out_dir):
        up_p   = os.path.join(out_dir, 'SHIFT_TABLES__UP__Throttle17.tsv')
        down_p = os.path.join(out_dir, 'SHIFT_TABLES__DOWN__Throttle17.tsv')
        if os.path.exists(up_p) and os.path.exists(down_p):
            up_df   = pd.read_csv(up_p, sep='\t', dtype=str)
            down_df = pd.read_csv(down_p, sep='\t', dtype=str)

            tps_cols = [c for c in up_df.columns if c not in ('mph','%')]
            # numeric copies
            up_num   = up_df.copy()
            down_num = down_df.copy()
            for c in tps_cols:
                up_num[c]   = pd.to_numeric(up_df[c],   errors='coerce')
                down_num[c] = pd.to_numeric(down_df[c], errors='coerce')

            for c in tps_cols:
                a = up_num[c]
                b = down_num[c]
                mask = a.notna() & b.notna()
                # do not clamp sentinels
                mask &= ~a.isin([317,318]) & ~b.isin([317,318])
                # enforce gap
                b.loc[mask] = np.minimum(b.loc[mask], a.loc[mask] - 1.0)
                down_num[c] = b

            # write clamped DOWN back to TSV (keep original header/order)
            for c in tps_cols:
                down_df[c] = down_num[c]
            down_df.to_csv(down_p, sep='\t', index=False)
            print('[CLAMP] Enforced DOWN ≤ UP − 1.0 mph in', down_p)
except Exception as _e:
    print('[CLAMP WARN]', _e)
# --- END AUTO CLAMP APPEND ---
