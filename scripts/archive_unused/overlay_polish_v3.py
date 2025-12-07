#!/usr/bin/env python3
import argparse, os, pandas as pd

def enforce_monotonic_row(vals):
    out = vals[:]
    for i in range(1, len(out)):
        if out[i] < out[i-1]:
            out[i] = out[i-1]
    return out

def polish(in_up, in_dn, out_up, out_dn):
    for src, dst in [(in_up,out_up),(in_dn,out_dn)]:
        df = pd.read_csv(src, sep='\t')
        cols = df.columns.tolist()
        tps_cols = cols[1:-1] if cols[-1] == '%' else cols[1:]
        vs = df[tps_cols].apply(pd.to_numeric, errors='coerce').values.tolist()
        vs = [enforce_monotonic_row(row) for row in vs]
        vs = [[round(x,1) if pd.notna(x) else x for x in row] for row in vs]
        df[tps_cols] = vs
        df.to_csv(dst, sep='\t', index=False)
    print('[POLISH] wrote:', out_up, 'and', out_dn)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', help='Folder that contains SHIFT_TABLES__UP/DOWN__Throttle17.tsv')
    ap.add_argument('--in-up')
    ap.add_argument('--in-down')
    ap.add_argument('--out-up')
    ap.add_argument('--out-down')
    a = ap.parse_args()
    if a.dir:
        in_up = os.path.join(a.dir, 'SHIFT_TABLES__UP__Throttle17.tsv')
        in_dn = os.path.join(a.dir, 'SHIFT_TABLES__DOWN__Throttle17.tsv')
        out_up, out_dn = in_up, in_dn
    else:
        in_up, in_dn, out_up, out_dn = a.in_up, a.in_down, a.out_up, a.out_down
    polish(in_up, in_dn, out_up, out_dn)

if __name__ == '__main__':
    main()
