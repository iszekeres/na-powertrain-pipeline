import pandas as pd
import numpy as np
from pathlib import Path

TPS_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]


def smooth_monotonic(vals, active_mask=None):
    """
    Enforce non-decreasing values across TPS for indices where active_mask is True.
    vals: list of floats/NaN
    active_mask: list[bool] or None (True = participate in smoothing)
    """
    vals = list(vals)
    if active_mask is None:
        active_mask = [not pd.isna(v) for v in vals]

    idxs = [i for i, a in enumerate(active_mask) if a and not pd.isna(vals[i])]
    if not idxs:
        return vals

    for k in range(1, len(idxs)):
        i_prev = idxs[k - 1]
        i_cur = idxs[k]
        prev_val = vals[i_prev]
        cur_val = vals[i_cur]
        if pd.isna(prev_val) or pd.isna(cur_val):
            continue
        if cur_val < prev_val:
            vals[i_cur] = prev_val
    return vals


def smooth_shift_table(in_path: Path, out_path: Path):
    """
    For each row labeled '* Shift', enforce monotonic vs TPS (0..100).
    """
    print(f"[SHIFT] Smoothing {in_path} -> {out_path}")
    df = pd.read_csv(in_path, sep="\t")

    all_tps_cols = [str(x) for x in TPS_AXIS]
    tps_cols = [c for c in all_tps_cols if c in df.columns]

    if "mph" not in df.columns:
        raise SystemExit(f"[SHIFT] No 'mph' column in {in_path}")

    for idx in range(len(df)):
        label = str(df.at[idx, "mph"])
        if "Shift" not in label:
            continue

        vals = []
        for c in tps_cols:
            v = df.at[idx, c]
            vals.append(v if not (isinstance(v, str) and v.strip() == "") else np.nan)

        vals_s = smooth_monotonic(vals)

        for c, v in zip(tps_cols, vals_s):
            df.at[idx, c] = v

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[SHIFT] Wrote {out_path}")


def smooth_tcc_table(in_path: Path, out_path: Path):
    """
    For each row labeled '* Apply' or '* Release':
      - Leave sentinel cells (>=300) alone.
      - Smooth the non-sentinel cells monotonic vs TPS.
    """
    print(f"[TCC] Smoothing {in_path} -> {out_path}")
    df = pd.read_csv(in_path, sep="\t")

    all_tps_cols = [str(x) for x in TPS_AXIS]
    tps_cols = [c for c in all_tps_cols if c in df.columns]

    if "mph" not in df.columns:
        raise SystemExit(f"[TCC] No 'mph' column in {in_path}")

    for idx in range(len(df)):
        label = str(df.at[idx, "mph"])
        if ("Apply" not in label) and ("Release" not in label):
            continue

        vals_raw = [df.at[idx, c] for c in tps_cols]
        vals = []
        active_mask = []

        for v in vals_raw:
            if pd.isna(v):
                vals.append(np.nan)
                active_mask.append(False)
                continue

            try:
                f = float(v)
            except Exception:  # noqa: BLE001
                vals.append(np.nan)
                active_mask.append(False)
                continue

            if f >= 300.0:
                vals.append(f)
                active_mask.append(False)
            else:
                vals.append(f)
                active_mask.append(True)

        vals_s = smooth_monotonic(vals, active_mask)

        for c, v in zip(tps_cols, vals_s):
            df.at[idx, c] = v

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[TCC] Wrote {out_path}")


def main():
    base = Path("newlogs") / "output" / "01_tables" / "BLENDED_LOGOVERBASE"
    in_dir = base / "AUDITFIX"
    out_dir = base / "SMOOTH"

    up_in = in_dir / "SHIFT_TABLES__UP__Throttle17.tsv"
    down_in = in_dir / "SHIFT_TABLES__DOWN__Throttle17.tsv"
    app_in = in_dir / "TCC_APPLY__Throttle17.tsv"
    rel_in = in_dir / "TCC_RELEASE__Throttle17.tsv"

    up_out = out_dir / "SHIFT_TABLES__UP__Throttle17.tsv"
    down_out = out_dir / "SHIFT_TABLES__DOWN__Throttle17.tsv"
    app_out = out_dir / "TCC_APPLY__Throttle17.tsv"
    rel_out = out_dir / "TCC_RELEASE__Throttle17.tsv"

    for p in [up_in, down_in, app_in, rel_in]:
        if not p.is_file():
            raise SystemExit(f"[ERROR] Missing input table: {p}")

    print("[INFO] Smoothing SHIFT UP/DOWN tables...")
    smooth_shift_table(up_in, up_out)
    smooth_shift_table(down_in, down_out)

    print("[INFO] Smoothing TCC APPLY/RELEASE tables...")
    smooth_tcc_table(app_in, app_out)
    smooth_tcc_table(rel_in, rel_out)

    print("[DONE] SMOOTH tables written under:", out_dir)


if __name__ == "__main__":
    main()

