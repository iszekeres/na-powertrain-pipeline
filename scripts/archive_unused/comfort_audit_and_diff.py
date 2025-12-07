import io
import math
import re
import sys
import zipfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

REPO_PACK = Path("bundles") / "Tahoe_6L80_Pack__ComfortGT_RowdyPerf__v4_FULL__2025-11-18.zip"
CUR_DIR = Path("newlogs") / "_truck_current"
OUT_DIR = Path("newlogs") / "output" / "02_passes" / "COMFORT_AUDIT"
OVERLAY_DIR = OUT_DIR / "CURRENT_TO_FINAL_OVERLAY"

SHIFT_FINAL_UP = Path(
    "newlogs/output/01_tables/shift/SHIFT_TABLES__UP__Throttle17__COMFORT_FINAL.tsv"
)
SHIFT_FINAL_DOWN = Path(
    "newlogs/output/01_tables/shift/SHIFT_TABLES__DOWN__Throttle17__COMFORT_FINAL.tsv"
)
TCC_FINAL_APPLY = Path(
    "newlogs/output/01_tables/tcc/TCC_APPLY__Throttle17__COMFORT_FINAL.tsv"
)
TCC_FINAL_REL = Path(
    "newlogs/output/01_tables/tcc/TCC_RELEASE__Throttle17__COMFORT_FINAL.tsv"
)

TPS_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]
SHIFT_ROWS_UP = [
    "1 -> 2 Shift",
    "2 -> 3 Shift",
    "3 -> 4 Shift",
    "4 -> 5 Shift",
    "5 -> 6 Shift",
]


def read_zip_tables(zip_path: Path, out_dir: Path):
    found = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            lo = name.lower()
            if lo.endswith(".tsv") and any(
                key in lo
                for key in [
                    "shift_tables__up__throttle17",
                    "shift_tables__down__throttle17",
                    "tcc_apply__throttle17",
                    "tcc_release__throttle17",
                ]
            ):
                target = out_dir / Path(name).name
                with zf.open(name) as f:
                    data = f.read()
                target.write_bytes(data)
                found.append(target)
    return found


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def header_is_17pt(df: pd.DataFrame):
    cols = list(df.columns)
    if len(cols) != 18:
        return False
    if cols[0].strip().lower() not in ["mph", "row", "shift", "label"]:
        return False
    try:
        parsed = [int(str(x).strip()) for x in cols[1:]]
    except Exception:
        return False
    return parsed == TPS_AXIS


def round_01(x):
    if pd.isna(x):
        return x
    d = Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return float(d)


def enforce_01(df: pd.DataFrame):
    for c in df.columns[1:]:
        df[c] = df[c].map(round_01)
    return df


def fix_rowlabels_for_down(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "2 -> 1 Shift": "1 -> 2 Shift",
        "3 -> 2 Shift": "2 -> 3 Shift",
        "4 -> 3 Shift": "3 -> 4 Shift",
        "5 -> 4 Shift": "4 -> 5 Shift",
        "6 -> 5 Shift": "5 -> 6 Shift",
    }
    df = df.copy()
    first = df.columns[0]
    df[first] = df[first].map(lambda s: mapping.get(str(s).strip(), s))
    return df


def sort_shift_rows(df: pd.DataFrame) -> pd.DataFrame:
    order = {r: i for i, r in enumerate(SHIFT_ROWS_UP)}
    f = df.copy()
    f["_ord"] = f[f.columns[0]].map(lambda s: order.get(str(s).strip(), 999))
    f = f.sort_values("_ord").drop(columns=["_ord"])
    return f


def audit_shift_tables(up: pd.DataFrame, down: pd.DataFrame):
    issues = []
    if not header_is_17pt(up):
        issues.append("UP header not 17-pt TPS axis")
    if not header_is_17pt(down):
        issues.append("DOWN header not 17-pt TPS axis")
    for name, tbl in [("UP", up), ("DOWN", down)]:
        for _, row in tbl.iterrows():
            vals = row.iloc[1:].astype(float).values
            if any(pd.isna(vals)):
                continue
            if any(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
                issues.append(f'{name} row "{row.iloc[0]}" is not monotonic across TPS')
    merged = up.merge(down, how="inner", on=up.columns[0], suffixes=("_up", "_down"))
    for _, r in merged.iterrows():
        for tps in TPS_AXIS:
            cu = r[f"{tps}_up"]
            cd = r[f"{tps}_down"]
            if pd.isna(cu) or pd.isna(cd):
                continue
            if not (cd <= cu - 1.0 + 1e-9):
                issues.append(
                    f'Hysteresis: row {r.iloc[0]} TPS {tps}: DOWN={cd} not <= UP-1.0 (UP={cu})'
                )
    return issues


def audit_tcc_tables(apply: pd.DataFrame, release: pd.DataFrame):
    issues = []
    if not header_is_17pt(apply):
        issues.append("APPLY header not 17-pt TPS axis")
    if not header_is_17pt(release):
        issues.append("RELEASE header not 17-pt TPS axis")
    merged = apply.merge(release, how="inner", on=apply.columns[0], suffixes=("_apply", "_release"))
    for _, r in merged.iterrows():
        for tps in TPS_AXIS:
            a = r[f"{tps}_apply"]
            rel = r[f"{tps}_release"]
            if pd.isna(a) or pd.isna(rel):
                continue
            if not (rel >= a + 1.1 - 1e-9):
                issues.append(
                    f"TCC gap: row {r.iloc[0]} TPS {tps}: RELEASE={rel} not >= APPLY+1.1 (APPLY={a})"
                )
    return issues


def diff_tables(cur: pd.DataFrame, fin: pd.DataFrame):
    k = cur.columns[0]
    cols = [k] + [c for c in cur.columns[1:] if c in fin.columns]
    cur2 = cur[cols].copy()
    fin2 = fin[cols].copy()
    deltas = [k] + [c for c in cur2.columns[1:]]
    out = pd.DataFrame(columns=deltas)
    out[k] = cur2[k]
    for c in cur2.columns[1:]:
        out[c] = fin2[c] - cur2[c]
    return out


def summarize_deltas(df: pd.DataFrame):
    cells = df.iloc[:, 1:].values.flatten()
    cells = cells[~pd.isna(cells)]
    if len(cells) == 0:
        return {"mean": None, "max_abs": None, "gt_2mph": 0, "gt_5mph": 0}
    import numpy as np

    mean = float(np.mean(cells))
    max_abs = float(np.max(np.abs(cells)))
    gt2 = int(np.sum(np.abs(cells) > 2.0))
    gt5 = int(np.sum(np.abs(cells) > 5.0))
    return {"mean": round(mean, 3), "max_abs": round(max_abs, 3), "gt_2mph": gt2, "gt_5mph": gt5}


def save_tsv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def load_shift(path: Path) -> pd.DataFrame:
    df = read_tsv(path)
    return enforce_01(df)


def load_tcc(path: Path) -> pd.DataFrame:
    df = read_tsv(path)
    return enforce_01(df)


def overlay_from_delta(delta_path: Path, overlay_path: Path):
    if not delta_path.exists():
        return
    d = read_tsv(delta_path)
    k = d.columns[0]
    body = d.copy()
    for c in body.columns[1:]:
        body[c] = body[c].where(~body[c].isna() & (body[c] != 0.0), other=pd.NA)
    keep = body.iloc[:, 1:].notna().any(axis=1)
    body = body.loc[keep]
    if body.empty:
        return
    save_tsv(body, overlay_path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    if not REPO_PACK.exists():
        print(f"[ERROR] Missing pack zip: {REPO_PACK}")
        sys.exit(1)

    found = read_zip_tables(REPO_PACK, CUR_DIR)
    print("[INFO] Extracted from pack:", *[f.name for f in found], sep="\n  - ")

    cur_shift_up = next((p for p in CUR_DIR.glob("SHIFT_TABLES__UP__Throttle17*.tsv")), None)
    cur_shift_dn = next((p for p in CUR_DIR.glob("SHIFT_TABLES__DOWN__Throttle17*.tsv")), None)
    cur_tcc_apply = next((p for p in CUR_DIR.glob("TCC_APPLY__Throttle17*.tsv")), None)
    cur_tcc_rel = next((p for p in CUR_DIR.glob("TCC_RELEASE__Throttle17*.tsv")), None)

    missing = []
    for need, path in [
        ("CUR SHIFT UP", cur_shift_up),
        ("CUR SHIFT DOWN", cur_shift_dn),
        ("CUR TCC APPLY", cur_tcc_apply),
        ("CUR TCC REL", cur_tcc_rel),
        ("FINAL SHIFT UP", SHIFT_FINAL_UP),
        ("FINAL SHIFT DOWN", SHIFT_FINAL_DOWN),
        ("FINAL TCC APPLY", TCC_FINAL_APPLY),
        ("FINAL TCC REL", TCC_FINAL_REL),
    ]:
        if path is None or not Path(path).exists():
            missing.append(need)
    if missing:
        print("[WARN] Missing expected files:", ", ".join(missing))

    cur_up = load_shift(cur_shift_up) if cur_shift_up and cur_shift_up.exists() else None
    cur_down = load_shift(cur_shift_dn) if cur_shift_dn and cur_shift_dn.exists() else None
    fin_up = load_shift(SHIFT_FINAL_UP) if SHIFT_FINAL_UP.exists() else None
    fin_down = load_shift(SHIFT_FINAL_DOWN) if SHIFT_FINAL_DOWN.exists() else None

    cur_apply = load_tcc(cur_tcc_apply) if cur_tcc_apply and cur_tcc_apply.exists() else None
    cur_rel = load_tcc(cur_tcc_rel) if cur_tcc_rel and cur_tcc_rel.exists() else None
    fin_apply = load_tcc(TCC_FINAL_APPLY) if TCC_FINAL_APPLY.exists() else None
    fin_rel = load_tcc(TCC_FINAL_REL) if TCC_FINAL_REL.exists() else None

    summary = []
    def add(line):
        summary.append(line)

    add("=== AUDIT: SHIFT (CURRENT) ===")
    if cur_up is not None and cur_down is not None:
        for p in (cur_up, cur_down):
            p.columns = [p.columns[0]] + [int(c) for c in p.columns[1:]]
        cur_shift_issues = audit_shift_tables(cur_up, cur_down)
        add("\n".join(["OK (no issues)"] if not cur_shift_issues else cur_shift_issues))
    else:
        add("Missing current UP/DOWN, skip audit.")

    add("\n=== AUDIT: SHIFT (FINAL COMFORT) ===")
    if fin_up is not None and fin_down is not None:
        for p in (fin_up, fin_down):
            p.columns = [p.columns[0]] + [int(c) for c in p.columns[1:]]
        fin_shift_issues = audit_shift_tables(fin_up, fin_down)
        add("\n".join(["OK (no issues)"] if not fin_shift_issues else fin_shift_issues))
    else:
        add("Missing final UP/DOWN, skip audit.")

    add("\n=== AUDIT: TCC (CURRENT) ===")
    if cur_apply is not None and cur_rel is not None:
        for p in (cur_apply, cur_rel):
            p.columns = [p.columns[0]] + [int(c) for c in p.columns[1:]]
        cur_tcc_issues = audit_tcc_tables(cur_apply, cur_rel)
        add("\n".join(["OK (no issues)"] if not cur_tcc_issues else cur_tcc_issues))
    else:
        add("Missing current TCC APPLY/RELEASE, skip audit.")

    add("\n=== AUDIT: TCC (FINAL COMFORT) ===")
    if fin_apply is not None and fin_rel is not None:
        for p in (fin_apply, fin_rel):
            p.columns = [p.columns[0]] + [int(c) for c in p.columns[1:]]
        fin_tcc_issues = audit_tcc_tables(fin_apply, fin_rel)
        add("\n".join(["OK (no issues)"] if not fin_tcc_issues else fin_tcc_issues))
    else:
        add("Missing final TCC APPLY/RELEASE, skip audit.")

    if cur_up is not None and fin_up is not None:
        for p in (cur_up, fin_up):
            p.columns = [p.columns[0]] + [int(c) for c in p.columns[1:]]
        d_up = diff_tables(cur_up, fin_up)
        save_tsv(d_up, OUT_DIR / "DELTA__SHIFT_UP__COMFORT_FINAL_minus_CURRENT.tsv")
        add("\nSHIFT UP deltas summary: " + str(summarize_deltas(d_up)))
    else:
        add("\nNo SHIFT UP diff (missing one side)")

    if cur_down is not None and fin_down is not None:
        for p in (cur_down, fin_down):
            p.columns = [p.columns[0]] + [int(c) for c in p.columns[1:]]
        cur_down_n = fix_rowlabels_for_down(cur_down)
        fin_down_n = fix_rowlabels_for_down(fin_down)
        d_dn = diff_tables(cur_down_n, fin_down_n)
        save_tsv(d_dn, OUT_DIR / "DELTA__SHIFT_DOWN__COMFORT_FINAL_minus_CURRENT.tsv")
        add("SHIFT DOWN deltas summary: " + str(summarize_deltas(d_dn)))
    else:
        add("No SHIFT DOWN diff (missing one side)")

    if cur_apply is not None and fin_apply is not None:
        for p in (cur_apply, fin_apply):
            p.columns = [p.columns[0]] + [int(c) for c in p.columns[1:]]
        d_ap = diff_tables(cur_apply, fin_apply)
        save_tsv(d_ap, OUT_DIR / "DELTA__TCC_APPLY__COMFORT_FINAL_minus_CURRENT.tsv")
        add("TCC APPLY deltas summary: " + str(summarize_deltas(d_ap)))
    else:
        add("No TCC APPLY diff (missing one side)")

    if cur_rel is not None and fin_rel is not None:
        for p in (cur_rel, fin_rel):
            p.columns = [p.columns[0]] + [int(c) for c in p.columns[1:]]
        d_rl = diff_tables(cur_rel, fin_rel)
        save_tsv(d_rl, OUT_DIR / "DELTA__TCC_RELEASE__COMFORT_FINAL_minus_CURRENT.tsv")
        add("TCC RELEASE deltas summary: " + str(summarize_deltas(d_rl)))
    else:
        add("No TCC RELEASE diff (missing one side)")

    overlay_from_delta(
        OUT_DIR / "DELTA__SHIFT_UP__COMFORT_FINAL_minus_CURRENT.tsv",
        OVERLAY_DIR / "OVERLAY__SHIFT_UP__CURRENT_to_COMFORT_FINAL.tsv",
    )
    overlay_from_delta(
        OUT_DIR / "DELTA__SHIFT_DOWN__COMFORT_FINAL_minus_CURRENT.tsv",
        OVERLAY_DIR / "OVERLAY__SHIFT_DOWN__CURRENT_to_COMFORT_FINAL.tsv",
    )
    overlay_from_delta(
        OUT_DIR / "DELTA__TCC_APPLY__COMFORT_FINAL_minus_CURRENT.tsv",
        OVERLAY_DIR / "OVERLAY__TCC_APPLY__CURRENT_to_COMFORT_FINAL.tsv",
    )
    overlay_from_delta(
        OUT_DIR / "DELTA__TCC_RELEASE__COMFORT_FINAL_minus_CURRENT.tsv",
        OVERLAY_DIR / "OVERLAY__TCC_RELEASE__CURRENT_to_COMFORT_FINAL.tsv",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "SUMMARY.txt").write_text("\n".join(summary), encoding="utf-8")

    print("\n[OK] Wrote outputs to:", OUT_DIR)
    for p in sorted(OUT_DIR.rglob("*")):
        print("  -", p.relative_to(Path.cwd()))


if __name__ == "__main__":
    main()
