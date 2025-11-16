import argparse, pandas as pd, decimal

CANON_TPS = ["0","6","12","19","25","31","37","44","50","56","62","69","75","81","87","94","100"]
SENTINELS = (decimal.Decimal("317"), decimal.Decimal("318"))

def snap_cell(x:str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s == "" or s.lower() in ("nan","na","none"):
        return ""  # leave blanks as blanks
    # keep sentinels exactly
    try:
        d = decimal.Decimal(s)
        if d in SENTINELS:
            return str(d)
        q = d.quantize(decimal.Decimal("0.1"), rounding=decimal.ROUND_HALF_UP)
        return f"{q}"
    except Exception:
        # non-numeric payloads (row labels, %, etc.) pass through
        return s

def process(path:str):
    # Read as strings so blanks don't become NaN
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    # Only round TPS columns (leave 'mph' and '%' alone)
    for col in [c for c in df.columns if c in CANON_TPS]:
        df[col] = df[col].map(snap_cell)
    # Write back with LF newlines
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")
    print(f"[1dp OK] {path}")

def main():
    ap = argparse.ArgumentParser(description="Strict 0.1 mph rounding for SHIFT tables (NaN-safe; preserves 317/318).")
    ap.add_argument("--up",   required=True, help="Path to SHIFT_TABLES__UP__Throttle17.tsv")
    ap.add_argument("--down", required=True, help="Path to SHIFT_TABLES__DOWN__Throttle17.tsv")
    args = ap.parse_args()
    process(args.up)
    process(args.down)

if __name__ == "__main__":
    main()
