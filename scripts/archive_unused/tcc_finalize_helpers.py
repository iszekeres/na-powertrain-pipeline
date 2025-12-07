import pandas as pd, numpy as np, math

TPS_COLS_CANON = ["0","6","12","19","25","31","37","44","50","56","62","69","75","81","87","94","100"]

def fmt_1dp_keep_sentinel(x):
    if x is None:
        return x
    try:
        v = float(x)
    except Exception:
        return x
    if not math.isfinite(v):
        return ""
    if v in (317.0, 318.0):   # EC3 lockout sentinels
        return int(v)
    return float(f"{v:.1f}")

def finalize_tcc_table(df):
    # Ensure all 17 TPS columns exist (add missing as NaN), and in canonical order
    present = [c for c in df.columns if c in TPS_COLS_CANON]
    missing = [c for c in TPS_COLS_CANON if c not in df.columns]
    for m in missing:
        df[m] = np.nan

    left = [df.columns[0]]
    right = [df.columns[-1]] if df.columns[-1] == "%" else []
    df = df[left + TPS_COLS_CANON + right]

    # Convert to numeric and clean zeros (<317) to NaN (zeros mean "no data" here)
    num = df[TPS_COLS_CANON].apply(pd.to_numeric, errors="coerce")
    zero_mask = (num < 317) & (num == 0)
    num = num.mask(zero_mask, np.nan)

    # Strict 1-dp with sentinel preservation
    df[TPS_COLS_CANON] = num.applymap(fmt_1dp_keep_sentinel)
    return df
