import argparse, pandas as pd, decimal

# Canonical TPS columns (do not touch 'mph' or '%')
CANON_TPS = ["0","6","12","19","25","31","37","44","50","56","62","69","75","81","87","94","100"]
SENTINELS = (decimal.Decimal("317"), decimal.Decimal("318"))

def snap_cell(x:str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s == "" or s.lower() in ("nan","na","none"):
        return ""  # keep blanks as blanks
    try:
        d = decimal.Decimal(s)
        if d in SENTINELS:
            return str(d)  # keep 317/318 exactly
        q = d.quantize(decimal.Decimal("0.1"), rounding=decimal.ROUND_HALF_UP)
        return f"{q}"
    except Exception:
        # Non-numeric payloads (row labels, '%', etc.) pass through unchanged
        return s

def process(path:str):
    # Read as strings so blanks don't become NaN
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    for col in [c for c in df.columns if c in CANON_TPS]:
        df[col] = df[col].map(snap_cell)
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")
    print(f"[1dp OK] {path}")

def main():
    ap = argparse.ArgumentParser(description="Strict 0.1 mph rounding for TCC tables (NaN-safe; preserves 317/318).")
    ap.add_argument("--apply",   required=True, help="Path to TCC_APPLY__Throttle17.tsv")
    ap.add_argument("--release", required=True, help="Path to TCC_RELEASE__Throttle17.tsv")
    args = ap.parse_args()
    process(args.apply)
    process(args.release)

if __name__ == "__main__":
    main()
