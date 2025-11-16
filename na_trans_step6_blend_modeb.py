import os

import numpy as np
import pandas as pd


BASE_DIR = os.path.join("newlogs", "baseline_current")
SHIFT_DIR = os.path.join("newlogs", "output", "01_tables", "shift")
TCC_DIR = os.path.join("newlogs", "output", "01_tables", "tcc")
PASS_ROOT = os.path.join("newlogs", "output", "02_passes")
OUT_DIR = os.path.join("newlogs", "output", "01_tables", "BLENDED_LOGOVERBASE")


def load_tsv(path: str, label: str, required: bool = True) -> pd.DataFrame | None:
    if not os.path.exists(path):
        if required:
            raise SystemExit(f"[ERROR] Required table missing ({label}): {path}")
        print(f"[WARN] Optional table missing ({label}): {path}")
        return None
    return pd.read_csv(path, sep="\t")


def override_shift(base_df: pd.DataFrame, log_df: pd.DataFrame, label: str) -> pd.DataFrame:
    print(f"[BLEND] Override SHIFT {label} with RAWEDGES where present")
    base = base_df.copy()
    log = log_df.copy()
    if "mph" not in base.columns or "mph" not in log.columns:
        raise SystemExit(f"[ERROR] SHIFT {label}: expected 'mph' column in both tables.")
    base_idx = base.set_index("mph")
    log_idx = log.set_index("mph")
    tps_cols = [c for c in base_idx.columns if c != "%"]
    for row_label in base_idx.index:
        if row_label not in log_idx.index:
            continue
        for col in tps_cols:
            l_raw = log_idx.at[row_label, col] if col in log_idx.columns else ""
            try:
                l_val = float(l_raw)
            except (TypeError, ValueError):
                l_val = np.nan
            if np.isnan(l_val):
                continue
            base_idx.at[row_label, col] = l_val
    return base_idx.reset_index()


def override_tcc(base_df: pd.DataFrame, log_df: pd.DataFrame, label: str) -> pd.DataFrame:
    print(f"[BLEND] Override TCC {label} with log TCC where present (preserve 317/318)")
    base = base_df.copy()
    log = log_df.copy()
    if "mph" not in base.columns or "mph" not in log.columns:
        raise SystemExit(f"[ERROR] TCC {label}: expected 'mph' column in both tables.")
    base_idx = base.set_index("mph")
    log_idx = log.set_index("mph")
    tps_cols = [c for c in base_idx.columns if c != "%"]
    for row_label in base_idx.index:
        if row_label not in log_idx.index:
            continue
        for col in tps_cols:
            b_raw = base_idx.at[row_label, col]
            l_raw = log_idx.at[row_label, col] if col in log_idx.columns else ""
            try:
                b_val = float(b_raw)
            except (TypeError, ValueError):
                b_val = np.nan
            try:
                l_val = float(l_raw)
            except (TypeError, ValueError):
                l_val = np.nan
            if not np.isnan(b_val) and int(round(b_val)) in (317, 318):
                continue
            if np.isnan(l_val):
                continue
            base_idx.at[row_label, col] = l_val
    return base_idx.reset_index()


def apply_delta(table_df: pd.DataFrame, delta_df: pd.DataFrame | None, label: str) -> pd.DataFrame:
    if delta_df is None:
        print(f"[DELTA] {label}: no delta (None)")
        return table_df
    print(f"[DELTA] Applying {label}")
    base_idx = table_df.set_index("mph")
    delta_idx = delta_df.set_index("mph")
    tps_cols = [c for c in base_idx.columns if c != "%"]
    for row_label in base_idx.index:
        if row_label not in delta_idx.index:
            continue
        for col in tps_cols:
            d_raw = delta_idx.at[row_label, col] if col in delta_idx.columns else ""
            try:
                d_val = float(d_raw)
            except (TypeError, ValueError):
                d_val = np.nan
            if np.isnan(d_val) or d_val == 0.0:
                continue
            b_raw = base_idx.at[row_label, col]
            try:
                b_val = float(b_raw)
            except (TypeError, ValueError):
                continue
            base_idx.at[row_label, col] = b_val + d_val
    return base_idx.reset_index()


def finalize_shift(up_df: pd.DataFrame, down_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    up = up_df.copy()
    down = down_df.copy()
    tps_cols = [c for c in up.columns if c not in ("mph", "%")]
    for df in (up, down):
        for col in tps_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    pairs = [
        ("1 -> 2 Shift", "2 -> 1 Shift"),
        ("2 -> 3 Shift", "3 -> 2 Shift"),
        ("3 -> 4 Shift", "4 -> 3 Shift"),
        ("4 -> 5 Shift", "5 -> 4 Shift"),
        ("5 -> 6 Shift", "6 -> 5 Shift"),
    ]
    for up_label, down_label in pairs:
        if up_label not in up["mph"].values or down_label not in down["mph"].values:
            continue
        for col in tps_cols:
            u = up.loc[up["mph"] == up_label, col].values[0]
            d = down.loc[down["mph"] == down_label, col].values[0]
            if pd.isna(u) or pd.isna(d):
                continue
            if d > u - 1.0:
                new_d = max(0.0, u - 1.0)
                down.loc[down["mph"] == down_label, col] = new_d

    def snap(v):
        if pd.isna(v):
            return ""
        v = max(0.0, float(v))
        return f"{round(v, 1):.1f}"

    for df in (up, down):
        for col in tps_cols:
            df[col] = df[col].apply(snap)

    return up, down


def apply_tcc_release_delta(
    apply_df: pd.DataFrame,
    rel_df: pd.DataFrame,
    delta_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if delta_df is None:
        print("[DELTA] No INTENT TCC_RELEASE delta.")
        return rel_df
    print("[DELTA] Applying INTENT TCC_RELEASE delta")
    app_idx = apply_df.set_index("mph")
    rel_idx = rel_df.set_index("mph")
    d_idx = delta_df.set_index("mph")
    tps_cols = [c for c in app_idx.columns if c != "%"]
    for row_label in rel_idx.index:
        if row_label not in d_idx.index:
            continue
        for col in tps_cols:
            d_raw = d_idx.at[row_label, col] if col in d_idx.columns else ""
            try:
                d_val = float(d_raw)
            except (TypeError, ValueError):
                d_val = np.nan
            if np.isnan(d_val) or d_val == 0.0:
                continue
            r_raw = rel_idx.at[row_label, col]
            try:
                r_val = float(r_raw)
            except (TypeError, ValueError):
                continue
            if int(round(r_val)) in (317, 318):
                continue
            rel_idx.at[row_label, col] = r_val + d_val
    return rel_idx.reset_index()


def finalize_tcc(app_df: pd.DataFrame, rel_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    app = app_df.copy()
    rel = rel_df.copy()
    tps_cols = [c for c in app.columns if c not in ("mph", "%")]
    for df in (app, rel):
        for col in tps_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def is_sentinel(v):
        return not pd.isna(v) and int(round(v)) in (317, 318)

    for gear in ["3rd", "4th", "5th", "6th"]:
        a_label = f"{gear} Apply"
        r_label = f"{gear} Release"
        if a_label not in app["mph"].values or r_label not in rel["mph"].values:
            continue
        for col in tps_cols:
            a = app.loc[app["mph"] == a_label, col].values[0]
            r = rel.loc[rel["mph"] == r_label, col].values[0]
            if pd.isna(a) or pd.isna(r) or is_sentinel(r) or is_sentinel(a):
                continue
            if r < a + 1.1:
                rel.loc[rel["mph"] == r_label, col] = a + 1.1

    def snap_tcc(v):
        if pd.isna(v):
            return ""
        v = float(v)
        if 316.5 <= v <= 317.5:
            return "317"
        if 317.5 <= v <= 318.5:
            return "318"
        return f"{round(v, 1):.1f}"

    for df in (app, rel):
        for col in tps_cols:
            df[col] = df[col].apply(snap_tcc)

    return app, rel


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    base_shift_up = os.path.join(BASE_DIR, "SHIFT_TABLES__UP__Throttle17.tsv")
    base_shift_down = os.path.join(BASE_DIR, "SHIFT_TABLES__DOWN__Throttle17.tsv")
    base_tcc_apply = os.path.join(BASE_DIR, "TCC_APPLY__Throttle17.tsv")
    base_tcc_rel = os.path.join(BASE_DIR, "TCC_RELEASE__Throttle17.tsv")

    base_shift_up_df = load_tsv(base_shift_up, "baseline SHIFT UP")
    base_shift_down_df = load_tsv(base_shift_down, "baseline SHIFT DOWN")
    base_tcc_apply_df = load_tsv(base_tcc_apply, "baseline TCC APPLY")
    base_tcc_rel_df = load_tsv(base_tcc_rel, "baseline TCC RELEASE")

    log_shift_up = os.path.join(SHIFT_DIR, "SHIFT_TABLES__UP__Throttle17__RAWEDGES.tsv")
    log_shift_down = os.path.join(SHIFT_DIR, "SHIFT_TABLES__DOWN__Throttle17__RAWEDGES.tsv")
    log_tcc_apply = os.path.join(TCC_DIR, "TCC_APPLY__Throttle17.tsv")
    log_tcc_rel = os.path.join(TCC_DIR, "TCC_RELEASE__Throttle17.tsv")

    log_shift_up_df = load_tsv(log_shift_up, "RAWEDGES SHIFT UP")
    log_shift_down_df = load_tsv(log_shift_down, "RAWEDGES SHIFT DOWN")
    log_tcc_apply_df = load_tsv(log_tcc_apply, "log TCC APPLY")
    log_tcc_rel_df = load_tsv(log_tcc_rel, "log TCC RELEASE")

    shift_up_neutral = override_shift(base_shift_up_df, log_shift_up_df, "UP")
    shift_down_neutral = override_shift(base_shift_down_df, log_shift_down_df, "DOWN")
    tcc_apply_neutral = override_tcc(base_tcc_apply_df, log_tcc_apply_df, "APPLY")
    tcc_rel_neutral = override_tcc(base_tcc_rel_df, log_tcc_rel_df, "RELEASE")

    def load_delta(path: str, label: str) -> pd.DataFrame | None:
        return load_tsv(path, label, required=False)

    lat_up_path = os.path.join(PASS_ROOT, "LAT", "LAT__SHIFT_UP__DELTA.tsv")
    intent_up_path = os.path.join(PASS_ROOT, "INTENT", "INTENT__SHIFT_UP__DELTA.tsv")
    consist_up_path = os.path.join(PASS_ROOT, "CONSIST", "CONSIST__SHIFT_UP__DELTA.tsv")

    lat_up_df = load_delta(lat_up_path, "LAT UP")
    intent_up_df = load_delta(intent_up_path, "INTENT UP")
    consist_up_df = load_delta(consist_up_path, "CONSIST UP")

    shift_up_delta = shift_up_neutral.copy()
    for name, df_delta in [
        ("LAT UP", lat_up_df),
        ("INTENT UP", intent_up_df),
        ("CONSIST UP", consist_up_df),
    ]:
        if df_delta is not None:
            shift_up_delta = apply_delta(shift_up_delta, df_delta, name)

    stopgo_down_path = os.path.join(PASS_ROOT, "STOPGO", "STOPGO__SHIFT_DOWN__DELTA.tsv")
    kick_down_path = os.path.join(PASS_ROOT, "KICKDOWN", "KICKDOWN__SHIFT_DOWN__DELTA.tsv")
    corner_comb_path = os.path.join(PASS_ROOT, "CORNER", "CORNER__SHIFT_DOWN__DELTA__COMBINED.tsv")
    consist_down_path = os.path.join(PASS_ROOT, "CONSIST", "CONSIST__SHIFT_DOWN__DELTA.tsv")

    stopgo_down_df = load_delta(stopgo_down_path, "STOPGO DOWN")
    kick_down_df = load_delta(kick_down_path, "KICKDOWN DOWN")
    corner_comb_df = load_delta(corner_comb_path, "CORNER COMBINED DOWN")
    consist_down_df = load_delta(consist_down_path, "CONSIST DOWN")

    shift_down_delta = shift_down_neutral.copy()
    for name, df_delta in [
        ("STOPGO DOWN", stopgo_down_df),
        ("KICKDOWN DOWN", kick_down_df),
        ("CORNER COMBINED DOWN", corner_comb_df),
        ("CONSIST DOWN", consist_down_df),
    ]:
        if df_delta is not None:
            shift_down_delta = apply_delta(shift_down_delta, df_delta, name)

    shift_up_final, shift_down_final = finalize_shift(shift_up_delta, shift_down_delta)

    intent_tcc_rel_path = os.path.join(PASS_ROOT, "INTENT", "INTENT__TCC_RELEASE__DELTA.tsv")
    intent_tcc_rel_df = load_delta(intent_tcc_rel_path, "INTENT TCC_RELEASE")

    tcc_rel_delta = apply_tcc_release_delta(tcc_apply_neutral, tcc_rel_neutral, intent_tcc_rel_df)
    tcc_apply_final, tcc_rel_final = finalize_tcc(tcc_apply_neutral, tcc_rel_delta)

    out_shift_up = os.path.join(OUT_DIR, "SHIFT_TABLES__UP__Throttle17.tsv")
    out_shift_down = os.path.join(OUT_DIR, "SHIFT_TABLES__DOWN__Throttle17.tsv")
    out_tcc_apply = os.path.join(OUT_DIR, "TCC_APPLY__Throttle17.tsv")
    out_tcc_rel = os.path.join(OUT_DIR, "TCC_RELEASE__Throttle17.tsv")

    shift_up_final.to_csv(out_shift_up, sep="\t", index=False)
    shift_down_final.to_csv(out_shift_down, sep="\t", index=False)
    tcc_apply_final.to_csv(out_tcc_apply, sep="\t", index=False)
    tcc_rel_final.to_csv(out_tcc_rel, sep="\t", index=False)

    print("\n[OK] Blended + pass-adjusted tables written to", OUT_DIR)
    for p in (out_shift_up, out_shift_down, out_tcc_apply, out_tcc_rel):
        print("  -", p)


if __name__ == "__main__":
    main()

