import os, pandas as pd

ROOT = r'.\newlogs\output'
SHIFT_DIR = os.path.join(ROOT,'01_tables','shift')
TCC_DIR   = os.path.join(ROOT,'01_tables','tcc')
PASS_DIR  = os.path.join(ROOT,'02_passes')

OUT_ROOT  = os.path.join(ROOT,'01_tables','BLENDED_CANDIDATE')
OUT_SHIFT = os.path.join(OUT_ROOT,'shift')
OUT_TCC   = os.path.join(OUT_ROOT,'tcc')
os.makedirs(OUT_SHIFT, exist_ok=True); os.makedirs(OUT_TCC, exist_ok=True)

def r(path): return pd.read_csv(path, sep='\t', dtype=str)

def r_opt(path):
    return r(path) if os.path.exists(path) else None

def add_deltas(base, delta):
    if delta is None: return base
    out = base.copy()
    for c in out.columns[1:-1]:
        a = pd.to_numeric(out[c], errors='coerce')
        if c in delta.columns:
            b = pd.to_numeric(delta[c], errors='coerce')
            b = b.reindex(a.index).fillna(0)
        else:
            b = 0
        out[c] = (a + b).astype(float)
    return out

# Load neutral base
up   = r(os.path.join(SHIFT_DIR,'SHIFT_TABLES__UP__Throttle17.tsv'))
down = r(os.path.join(SHIFT_DIR,'SHIFT_TABLES__DOWN__Throttle17.tsv'))
tccA = r(os.path.join(TCC_DIR,'TCC_APPLY__Throttle17.tsv'))
tccR = r(os.path.join(TCC_DIR,'TCC_RELEASE__Throttle17.tsv'))

# Load DELTAs (optional files handled)
d_CONS_UP  = r_opt(os.path.join(PASS_DIR,'CONSIST','CONSIST__SHIFT_UP__DELTA.tsv'))
d_CONS_DN  = r_opt(os.path.join(PASS_DIR,'CONSIST','CONSIST__SHIFT_DOWN__DELTA.tsv'))
d_LAT_UP   = r_opt(os.path.join(PASS_DIR,'LAT','LAT__SHIFT_UP__DELTA.tsv'))
d_COR_DN   = r_opt(os.path.join(PASS_DIR,'CORNER','CORNER__SHIFT_DOWN__DELTA__COMBINED.tsv'))
d_STP_DN   = r_opt(os.path.join(PASS_DIR,'STOPGO','STOPGO__SHIFT_DOWN__DELTA.tsv'))
d_KDN_DN   = r_opt(os.path.join(PASS_DIR,'KICKDOWN','KICKDOWN__SHIFT_DOWN__DELTA.tsv'))
d_INT_UP   = r_opt(os.path.join(PASS_DIR,'CRUISE_TIPIN','INTENT__SHIFT_UP__DELTA.tsv'))
d_INT_TCCR = r_opt(os.path.join(PASS_DIR,'CRUISE_TIPIN','INTENT__TCC_RELEASE__DELTA.tsv'))

# Blend SHIFT
UP_blend = up.copy()
for d in (d_CONS_UP, d_LAT_UP, d_INT_UP):
    UP_blend = add_deltas(UP_blend, d)

DOWN_blend = down.copy()
for d in (d_CONS_DN, d_COR_DN, d_STP_DN, d_KDN_DN):
    DOWN_blend = add_deltas(DOWN_blend, d)

# Blend TCC (only RELEASE has INTENT deltas; APPLY remains base)
TCC_A_blend = tccA.copy()
TCC_R_blend = add_deltas(tccR.copy(), d_INT_TCCR)

# Write BLENDED_CANDIDATE
UP_blend.to_csv( os.path.join(OUT_SHIFT,'SHIFT_TABLES__UP__Throttle17.tsv'),   sep='\t', index=False)
DOWN_blend.to_csv( os.path.join(OUT_SHIFT,'SHIFT_TABLES__DOWN__Throttle17.tsv'), sep='\t', index=False)
TCC_A_blend.to_csv(os.path.join(OUT_TCC,'TCC_APPLY__Throttle17.tsv'),   sep='\t', index=False)
TCC_R_blend.to_csv(os.path.join(OUT_TCC,'TCC_RELEASE__Throttle17.tsv'), sep='\t', index=False)
print('[WRITE] BLENDED_CANDIDATE ->', OUT_ROOT)
