from pathlib import Path

import pandas as pd

TPS_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]
TPS_COLS = [str(v) for v in TPS_AXIS]
TCC_SENTINELS = {317.0, 318.0}


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[ERROR] Table not found: {path}")
    return pd.read_csv(path, sep="\t")


def assert_tps_axis(df: pd.DataFrame, label: str) -> None:
    """
    Ensure the table has the full 17-pt TPS axis.
    """
    missing = [c for c in TPS_COLS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"[ERROR] Missing TPS columns in {label} table: {missing}. "
            "Expected full 17-point TPS axis."
        )


def main():
    base = Path("newlogs") / "output" / "01_tables"
    shift_dir = base / "shift"
    tcc_dir = base / "tcc"
    baseline_dir = base / "BASELINE_LOGFIRST"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # SHIFT: RAWEDGES -> guarded baseline UP/DOWN
    # -------------------------------------------------------------------------
    up_raw = load_table(shift_dir / "SHIFT_TABLES__UP__Throttle17__RAWEDGES.tsv")
    down_raw = load_table(shift_dir / "SHIFT_TABLES__DOWN__Throttle17__RAWEDGES.tsv")

    # Enforce full 17-pt TPS axis for SHIFT tables
    assert_tps_axis(up_raw, "SHIFT UP")
    assert_tps_axis(down_raw, "SHIFT DOWN")

    tps_cols = [c for c in TPS_COLS if c in up_raw.columns]

    df_up = up_raw.copy()
    df_down = down_raw.copy()

    # 1) Monotonic across TPS for each row (UP and DOWN)
    for df in (df_up, df_down):
        for idx in range(len(df)):
            label = str(df.at[idx, "mph"])
            if "Shift" not in label:
                continue
            last = None
            for c in tps_cols:
                v = df.at[idx, c]
                if isinstance(v, str) and not v.strip():
                    continue
                try:
                    f = float(v)
                except Exception:  # noqa: BLE001
                    continue
                if last is None:
                    last = f
                else:
                    if f < last:
                        f = last
                        df.at[idx, c] = f
                    last = f

    # 2) Cross-gear constraint on UP: 1->2 < 2->3 < ... < 5->6 with +1.5 mph min gap
    up_order = [
        "1 -> 2 Shift",
        "2 -> 3 Shift",
        "3 -> 4 Shift",
        "4 -> 5 Shift",
        "5 -> 6 Shift",
    ]
    row_index = {str(df_up.at[i, "mph"]): i for i in range(len(df_up))}
    for c in tps_cols:
        prev = None
        for label in up_order:
            i = row_index.get(label)
            if i is None:
                continue
            v = df_up.at[i, c]
            if isinstance(v, str) and not v.strip():
                continue
            try:
                f = float(v)
            except Exception:  # noqa: BLE001
                continue
            if prev is None:
                prev = f
            else:
                min_allowed = prev + 1.5
                if f < min_allowed:
                    f = min_allowed
                    df_up.at[i, c] = f
                prev = f

    # 3) DOWN ≤ UP - 1.0 mph wherever both exist
    down_order = [
        "2 -> 1 Shift",
        "3 -> 2 Shift",
        "4 -> 3 Shift",
        "5 -> 4 Shift",
        "6 -> 5 Shift",
    ]

    for up_label, down_label in zip(up_order, down_order):
        up_row_df = df_up[df_up["mph"] == up_label]
        down_row_df = df_down[df_down["mph"] == down_label]
        if up_row_df.empty or down_row_df.empty:
            continue
        up_row = up_row_df.iloc[0]
        down_idx = down_row_df.index[0]
        for c in tps_cols:
            up_v = up_row[c]
            down_v = df_down.at[down_idx, c]
            try:
                u = float(up_v)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(down_v, str) and not down_v.strip():
                continue
            try:
                d = float(down_v)
            except Exception:  # noqa: BLE001
                continue
            max_allowed = u - 1.0
            if d > max_allowed:
                d = max_allowed
                df_down.at[down_idx, c] = d

    # Write baseline SHIFT tables (0.1 mph resolution)
    up_out = baseline_dir / "SHIFT_TABLES__UP__Throttle17.tsv"
    down_out = baseline_dir / "SHIFT_TABLES__DOWN__Throttle17.tsv"
    df_up.to_csv(up_out, sep="\t", index=False, float_format="%.1f")
    df_down.to_csv(down_out, sep="\t", index=False, float_format="%.1f")

    # -------------------------------------------------------------------------
    # TCC: enforce Release ≥ Apply + 1.1 mph, preserving 317/318 sentinels
    # -------------------------------------------------------------------------
    app_df = load_table(tcc_dir / "TCC_APPLY__Throttle17.tsv")
    rel_df = load_table(tcc_dir / "TCC_RELEASE__Throttle17.tsv")

    # Enforce full 17-pt TPS axis for TCC tables
    assert_tps_axis(app_df, "TCC APPLY")
    assert_tps_axis(rel_df, "TCC RELEASE")

    tps_cols_tcc = [c for c in TPS_COLS if c in app_df.columns]

    for idx in range(len(app_df)):
        label = str(app_df.at[idx, "mph"])
        if "Apply" not in label:
            continue
        target = label.replace("Apply", "Release")
        rel_row_df = rel_df[rel_df["mph"] == target]
        if rel_row_df.empty:
            continue
        rel_idx = rel_row_df.index[0]
        for c in tps_cols_tcc:
            av = app_df.at[idx, c]
            rv = rel_df.at[rel_idx, c]

            # Skip empty cells
            if isinstance(av, str) and not av.strip():
                continue
            if isinstance(rv, str) and not rv.strip():
                continue

            try:
                a = float(av)
            except Exception:  # noqa: BLE001
                continue
            try:
                r = float(rv)
            except Exception:  # noqa: BLE001
                continue

            # Preserve 317/318 sentinels exactly (no gap enforcement)
            if a in TCC_SENTINELS or r in TCC_SENTINELS:
                continue

            min_rel = a + 1.1
            if r < min_rel:
                rel_df.at[rel_idx, c] = min_rel

    app_out = baseline_dir / "TCC_APPLY__Throttle17.tsv"
    rel_out = baseline_dir / "TCC_RELEASE__Throttle17.tsv"
    app_df.to_csv(app_out, sep="\t", index=False, float_format="%.1f")
    rel_df.to_csv(rel_out, sep="\t", index=False, float_format="%.1f")

    print(f"[OK] LOG-FIRST baseline tables written to {baseline_dir}")


if __name__ == "__main__":
    main()

