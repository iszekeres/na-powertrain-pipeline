from pathlib import Path

script = """import math
import sys
import zipfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

REPO_PACK = Path("bundles") / "Tahoe_6L80_Pack__ComfortGT_RowdyPerf__v4_FULL__2025-11-18.zip"
CURRENT_DIR = Path("newlogs") / "_truck_current"
OUTPUT_DIR = Path("newlogs") / "output" / "02_passes" / "COMFORT"
SUMMARY_PATH = OUTPUT_DIR / "COMFORT_VS_ONTRUCK__SUMMARY.txt"
TPS_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]
SENTINELS = {317.0, 318.0}
SHIFT_ROWS = [
    "1 -> 2 Shift",
    "2 -> 3 Shift",
    "3 -> 4 Shift",
    "4 -> 5 Shift",
    "5 -> 6 Shift",
]

TABLE_SPEC = {
    "shift_up": {
        "keyword": "shift_tables__up__throttle17",
        "extra": ["comfort", "comforgt"],
    },
    "shift_down": {
        "keyword": "shift_tables__down__throttle17",
        "extra": ["comfort", "comforgt"],
    },
    "tcc_apply": {
        "keyword": "tcc_apply__throttle17",
        "extra": ["comfort", "comforgt"],
    },
    "tcc_release": {
        "keyword": "tcc_release__throttle17",
        "extra": ["comfort", "comforgt"],
    },
}

DOWN_LABEL_MAP = {
    "2 -> 1 Shift": "1 -> 2 Shift",
    "3 -> 2 Shift": "2 -> 3 Shift",
    "4 -> 3 Shift": "3 -> 4 Shift",
    "5 -> 4 Shift": "4 -> 5 Shift",
    "6 -> 5 Shift": "5 -> 6 Shift",
}

FINAL_TABLES = {
    "shift_up": Path(
        "newlogs/output/01_tables/shift/SHIFT_TABLES__UP__Throttle17__COMFORT_FINAL.tsv"
    ),
    "shift_down": Path(
        "newlogs/output/01_tables/shift/SHIFT_TABLES__DOWN__Throttle17__COMFORT_FINAL.tsv"
    ),
    "tcc_apply": Path(
        "newlogs/output/01_tables/tcc/TCC_APPLY__Throttle17__COMFORT_FINAL.tsv"
    ),
    "tcc_release": Path(
        "newlogs/output/01_tables/tcc/TCC_RELEASE__Throttle17__COMFORT_FINAL.tsv"
    ),
}

DELTA_FILES = {
    "shift_up": OUTPUT_DIR / "COMFORT_VS_ONTRUCK__SHIFT_UP_DELTA.tsv",
    "shift_down": OUTPUT_DIR / "COMFORT_VS_ONTRUCK__SHIFT_DOWN_DELTA.tsv",
    "tcc_apply": OUTPUT_DIR / "COMFORT_VS_ONTRUCK__TCC_APPLY_DELTA.tsv",
    "tcc_release": OUTPUT_DIR / "COMFORT_VS_ONTRUCK__TCC_RELEASE_DELTA.tsv",
}

SUMMARY_FILES = {
    "shift_up": OUTPUT_DIR / "COMFORT_VS_ONTRUCK__SHIFT_UP_SUMMARY.tsv",
    "shift_down": OUTPUT_DIR / "COMFORT_VS_ONTRUCK__SHIFT_DOWN_SUMMARY.tsv",
    "tcc_apply": OUTPUT_DIR / "COMFORT_VS_ONTRUCK__TCC_APPLY_SUMMARY.tsv",
    "tcc_release": OUTPUT_DIR / "COMFORT_VS_ONTRUCK__TCC_RELEASE_SUMMARY.tsv",
}


def ensure_dirs():
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def choose_best(candidates):
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    for cand in candidates:
        if "comforgt" in cand.lower():
            return cand
    print(f"[WARN] Multiple candidates for comfort table: {', '.join(candidates)}; picking first.")
    return candidates[0]


def extract_current_tables():
    if not REPO_PACK.exists():
        print(f"[ERROR] Missing pack: {REPO_PACK}")
        sys.exit(1)
    found = {}
    with zipfile.ZipFile(REPO_PACK, "r") as zf:
        names = zf.namelist()
        for key, spec in TABLE_SPEC.items():
            matches = [
                name
                for name in names
                if name.lower().endswith(".tsv")
                and spec["keyword"] in name.lower()
                and any(extra in name.lower() for extra in spec["extra"])
            ]
            selected = choose_best(matches)
            if selected:
                target = CURRENT_DIR / Path(selected).name
                with zf.open(selected) as src:
                    target.write_bytes(src.read())
                found[key] = target
    print("[INFO] Current comfort tables:")
    for key in ["shift_up", "shift_down", "tcc_apply", "tcc_release"]:
        print(f"  {key.replace('_', ' ').upper()}: {found.get(key)}")
    return found


def round_value(value, is_tcc=False):
    if pd.isna(value):
        return value
    try:
        num = float(value)
    except (ValueError, TypeError):
        return value
    if is_tcc and num in SENTINELS:
        return num
    rounded = Decimal(str(num)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return float(rounded)


def load_table(path, is_tcc=False, is_down=False):
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={df.columns[0]: "row_label"})
    axis_values = []
    names = ["row_label"]
    axis_valid = True
    for col in df.columns[1:]:
        label = str(col).strip()
        axis_val = None
        try:
            axis_val = int(float(label))
        except ValueError:
            axis_val = label
            axis_valid = False
        axis_values.append(axis_val)
        names.append(axis_val)
    df.columns = names
    if is_down:
        df["row_label"] = df["row_label"].map(lambda x: DOWN_LABEL_MAP.get(str(x).strip(), x))
    for col in df.columns[1:]:
        df[col] = df[col].apply(lambda v: round_value(v, is_tcc=is_tcc))
    if len(axis_values) != len(TPS_AXIS) or axis_values != TPS_AXIS:
        axis_valid = False
    return df, axis_valid


def compute_delta(current, final, is_tcc=False, is_down=False):
    cur_df = current.set_index("row_label")
    fin_df = final.set_index("row_label")
    all_rows = sorted(set(cur_df.index) | set(fin_df.index))
    if not is_tcc:
        order = {label: idx for idx, label in enumerate(SHIFT_ROWS)}
        all_rows = sorted(all_rows, key=lambda r: order.get(r, len(order)))
    common_cols = sorted(
        {int(c) for c in cur_df.columns if c in fin_df.columns and isinstance(c, (int, float))}
    )
    data = {"row_label": all_rows}
    for col in common_cols:
        cur_series = cur_df[col].reindex(all_rows)
        fin_series = fin_df[col].reindex(all_rows)
        data[col] = (fin_series - cur_series).values
    sentinel_dict = {
        col: (
            cur_df[col].reindex(all_rows).isin(SENTINELS)
            | fin_df[col].reindex(all_rows).isin(SENTINELS)
        ).values
        for col in common_cols
    }
    delta_df = pd.DataFrame(data)
    sentinel_df = pd.DataFrame(sentinel_dict)
    return delta_df, sentinel_df


def summarize_delta(delta_df, sentinel_df=None):
    axis_cols = [c for c in delta_df.columns if c != "row_label"]
    rows = []
    for idx, row in delta_df.iterrows():
        values = row[axis_cols].astype(float)
        if sentinel_df is not None and not sentinel_df.empty:
            mask = sentinel_df.iloc[idx].astype(bool).values
            values = values.mask(mask)
        valid = values.dropna()
        mean_delta = valid.mean() if not valid.empty else float("nan")
        max_abs = valid.abs().max() if not valid.empty else float("nan")
        count_gt2 = int((valid.abs() > 2.0).sum())
        count_gt5 = int((valid.abs() > 5.0).sum())
        rows.append(
            {
                "row_label": row["row_label"],
                "mean_delta": mean_delta,
                "max_abs_delta": max_abs,
                "count_gt_2": count_gt2,
                "count_gt_5": count_gt5,
            }
        )
    return pd.DataFrame(rows)


def write_summary_text(summaries, notes=None):
    lines = []
    for kind, df in summaries.items():
        lines.append(f"=== {kind.replace('_', ' ').upper()} ===")
        for _, row in df.iterrows():
            mean_val = row["mean_delta"]
            max_val = row["max_abs_delta"]
            mean_str = "N/A" if math.isnan(mean_val) else f"{mean_val:+.1f} mph"
            max_str = "N/A" if math.isnan(max_val) else f"{max_val:.1f} mph"
            lines.append(
                f"{row['row_label']}: mean Δ={mean_str}, max |Δ|={max_str}, "
                f"{row['count_gt_2']} cells >2 mph, {row['count_gt_5']} cells >5 mph"
            )
        lines.append("")
    if notes:
        lines.append("=== NOTES ===")
        lines.extend(notes)
    SUMMARY_PATH.write_text("\n".join(lines).strip(), encoding="utf-8")


def print_final_summary(current_files):
    print("\n[INFO] CURRENT comfort tables:")
    for key in ["shift_up", "shift_down", "tcc_apply", "tcc_release"]:
        print(f"  {key.replace('_', ' ').upper()}: {current_files.get(key)}")
    print("\n[INFO] FINAL COMFORT tables:")
    for key, path in FINAL_TABLES.items():
        print(f"  {key.replace('_', ' ').upper()}: {path}")
    print("\n[OK] Wrote:")
    outputs = list(DELTA_FILES.values()) + list(SUMMARY_FILES.values()) + [SUMMARY_PATH]
    for out in outputs:
        print(f"  {out}")


def main():
    ensure_dirs()
    current_files = extract_current_tables()
    loaded = {}
    issues = []
    for key in TABLE_SPEC:
        current = current_files.get(key)
        final = FINAL_TABLES.get(key)
        is_tcc = key.startswith("tcc")
        is_down = key == "shift_down"
        if current is None or final is None or not final.exists():
            issues.append(f"[WARN] Missing table for {key}")
            continue
        cur_df, axis_ok_cur = load_table(current, is_tcc=is_tcc, is_down=is_down)
        fin_df, axis_ok_fin = load_table(final, is_tcc=is_tcc, is_down=is_down)
        if not axis_ok_cur:
            issues.append(f"[ISSUE] Current {key} axis differs from TPS axis")
        if not axis_ok_fin:
            issues.append(f"[ISSUE] Final {key} axis differs from TPS axis")
        loaded[key] = (cur_df, fin_df)
    summaries = {}
    for key, (cur_df, fin_df) in loaded.items():
        is_tcc = key.startswith("tcc")
        delta_df, sentinel_df = compute_delta(
            cur_df, fin_df, is_tcc=is_tcc, is_down=(key == "shift_down")
        )
        delta_df.to_csv(DELTA_FILES[key], sep="\t", index=False)
        summary_df = summarize_delta(delta_df, sentinel_df if is_tcc else None)
        summary_df.to_csv(SUMMARY_FILES[key], sep="\t", index=False)
        summaries[key] = summary_df
    write_summary_text(summaries, notes=issues)
    if issues:
        print("\n".join(issues))
    print_final_summary(current_files)


if __name__ == "__main__":
    main()
"""

Path("tools/comfort_vs_ontruck_diff.py").write_text(script, encoding="utf-8")
