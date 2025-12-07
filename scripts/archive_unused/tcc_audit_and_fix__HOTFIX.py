import os, io, re, argparse, numpy as np, pandas as pd

TPS_CANON = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
HDR_CANON = ['mph'] + list(map(str, TPS_CANON)) + ['%']

def read_tsv_clean(path):
    with open(path,'r',encoding='utf-8', newline='') as f:
        txt = f.read()
    # fix literal-escaped tabs/newlines if present
    if ('\\t' in txt and '\t' not in txt) or ('\\n' in txt and '\n' not in txt):
        txt = txt.replace('\\t','\t').replace('\\n','\n')
    # parse as TSV
    df = pd.read_csv(io.StringIO(txt), sep='\t', engine='python', dtype=str)
    # if it collapsed to a single column, try splitting manually
    if df.shape[1] == 1:
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        rows  = [ln.split('\t') for ln in lines]
        df = pd.DataFrame(rows[1:], columns=rows[0])
    # normalize header: accept 18 or 19 cols (with or without trailing '%')
    cols = list(df.columns)
    if cols and cols[0].strip().lower() == 'mph':
        # if missing %, add it; if extra cols, trim
        if len(cols) == 18:
            cols = ['mph'] + list(map(str, TPS_CANON))
            df.columns = cols
            df['%'] = ''
        elif len(cols) >= 19:
            df = df.iloc[:, :19]
            df.columns = HDR_CANON
        else:
            raise ValueError(f"header has {len(cols)} cols; expected 18 or 19")
    else:
        # last resort: force canonical header length
        if df.shape[1] == 19:
            df.columns = HDR_CANON
        else:
            raise ValueError(f"could not normalize header: {cols}")
    return df

def to_num(v):
    try:
        x = float(v)
        return x
    except Exception:
        return np.nan

def round_1dp_preserve(x):
    if pd.isna(x): return x
    # preserve lockout sentinels exactly if present (317/318)
    if abs(x-317.0) < 1e-9 or abs(x-318.0) < 1e-9: return int(x)
    return round(float(x), 1)

def audit_tcc(apply_df, release_df):
    ap = apply_df.copy()
    rl = release_df.copy()

    # clean row labels (3th -> 3rd, stray spaces)
    ap['mph'] = ap['mph'].str.replace(r'\b3th\b','3rd',regex=True).str.strip()
    rl['mph'] = rl['mph'].str.replace(r'\b3th\b','3rd',regex=True).str.strip()

    cols = [c for c in ap.columns if c not in ('mph','%')]

    # numeric conversion
    for c in cols:
        ap[c] = ap[c].map(to_num)
        rl[c] = rl[c].map(to_num)

    # enforce RELEASE ≥ APPLY + 1.1 where both present and suffixes match (3rd/4th/5th/6th)
    viol = 0
    for g in ('3rd','4th','5th','6th'):
        ra = ap[ap['mph'].str.startswith(g + ' Apply', na=False)]
        rr = rl[rl['mph'].str.startswith(g + ' Release', na=False)]
        if ra.empty or rr.empty: continue
        ia, ir = ra.index[0], rr.index[0]
        for c in cols:
            a = ap.at[ia,c]; r = rl.at[ir,c]
            if pd.notna(a) and pd.notna(r):
                need = a + 1.1
                if (r + 1e-9) < need:
                    rl.at[ir,c] = need
                    viol += 1

    # rounding (1dp) with sentinel preservation
    for c in cols:
        ap[c] = ap[c].map(round_1dp_preserve)
        rl[c] = rl[c].map(round_1dp_preserve)

    return ap, rl, viol

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--apply', required=True)
    p.add_argument('--release', required=True)
    p.add_argument('--out-dir', required=True)
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    ap = read_tsv_clean(a.apply)
    rl = read_tsv_clean(a.release)
    ap2, rl2, viol = audit_tcc(ap, rl)

    ap_out = os.path.join(a.out_dir, 'TCC_APPLY__Throttle17.tsv')
    rl_out = os.path.join(a.out_dir, 'TCC_RELEASE__Throttle17.tsv')
    ap2.to_csv(ap_out, sep='\t', index=False, float_format='%.1f')
    rl2.to_csv(rl_out, sep='\t', index=False, float_format='%.1f')
    print("TCC gap violations fixed:", viol)
    print("Wrote:", ap_out)
    print("Wrote:", rl_out)

if __name__ == "__main__":
    main()
