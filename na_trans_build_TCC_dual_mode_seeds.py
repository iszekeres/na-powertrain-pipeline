import os, math, shutil
import pandas as pd

ROOT = "newlogs"
TABLE_ROOT = os.path.join(ROOT, "output", "01_tables")

# Neutral, polished tables we treat as "middle ground"
NEUTRAL_DIR = os.path.join(TABLE_ROOT, "BLENDED_LOGOVERBASE", "AUDITFIX_SMOOTH")

SHIFT_UP_NAME   = "SHIFT_TABLES__UP__Throttle17.tsv"
SHIFT_DOWN_NAME = "SHIFT_TABLES__DOWN__Throttle17.tsv"
TCC_APPLY_NAME  = "TCC_APPLY__Throttle17.tsv"
TCC_REL_NAME    = "TCC_RELEASE__Throttle17.tsv"

COMFORT_DIR = os.path.join(TABLE_ROOT, "COMFORT_SEED")
PERF_DIR    = os.path.join(TABLE_ROOT, "PERF_SEED")

os.makedirs(COMFORT_DIR, exist_ok=True)
os.makedirs(PERF_DIR,    exist_ok=True)

FD = 3.08
TIRE_DIA_IN = 32.5

# 6L80 gear ratios
GEAR_RATIOS = {
    3: 1.532,
    4: 1.152,
    5: 0.852,
    6: 0.667,
}

def rpm_per_mph_for_gear(gear:int) -> float:
    """
    engine_rpm = mph * 1056 * gear_ratio * FD / (pi * tire_dia_in)
    """
    gr = GEAR_RATIOS[gear]
    return 1056.0 * gr * FD / (math.pi * TIRE_DIA_IN)

RPM_PER_MPH = {g: rpm_per_mph_for_gear(g) for g in GEAR_RATIOS}

def read_tsv(path:str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"[ERROR] Missing table: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)
    # keep everything as string initially; we'll parse numerics per-cell
    return df

def snap_1dp(x: float) -> float:
    return float(f"{round(x, 1):.1f}")

def is_sentinel(v: str) -> bool:
    try:
        f = float(v)
    except Exception:
        return False
    return f in (317.0, 318.0)

def adjust_tcc_tables(df_apply: pd.DataFrame,
                      df_rel: pd.DataFrame,
                      rpm_floor: float,
                      lockout_tps_bins) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clone APPLY/RELEASE and:
      - Raise APPLY mph so rpm at lock >= rpm_floor (for non-sentinel cells).
      - Force lockout (318/317) in specified TPS bins.
      - Enforce RELEASE >= APPLY + 1.1 mph everywhere.
    """
    apply = df_apply.copy(deep=True)
    rel   = df_rel.copy(deep=True)

    cols = list(apply.columns)
    if "mph" not in cols:
        raise SystemExit("[ERROR] TCC tables must have 'mph' as first column.")
    tps_cols = [c for c in cols if c not in ("mph", "%")]

    # Build a quick map from "3rd Apply" -> idx, "3rd Release" -> idx, etc.
    apply_idx = {row_label: idx for idx, row_label in enumerate(apply["mph"])}
    rel_idx   = {row_label: idx for idx, row_label in enumerate(rel["mph"])}

    # Pass 1: adjust APPLY cells and lockout TPS regions
    for row_label, idx in apply_idx.items():
        if "Apply" not in str(row_label):
            continue
        # row_label like "3rd Apply"
        token = str(row_label).split()[0]  # "3rd"
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            continue
        gear = int(digits)
        if gear not in RPM_PER_MPH:
            # ignore 1st/2nd or anything weird; they should stay sentinel
            continue
        k = RPM_PER_MPH[gear]

        for col in tps_cols:
            tps_str = col
            try:
                tps_val = float(tps_str)
            except Exception:
                # non-numeric header (shouldn't happen)
                continue

            cell = apply.at[idx, col]
            if isinstance(cell, float) or isinstance(cell, int):
                cell_str = f"{cell}"
            else:
                cell_str = str(cell).strip()

            if cell_str == "":
                # no data, skip
                continue

            if is_sentinel(cell_str):
                # keep sentinels as-is unless we're in an explicit lockout bin
                if tps_val in lockout_tps_bins:
                    # make sure it's 318.0 for APPLY
                    apply.at[idx, col] = "318.0"
                    rel_label = row_label.replace("Apply", "Release")
                    if rel_label in rel_idx:
                        ridx = rel_idx[rel_label]
                        rel.at[ridx, col] = "317.0"
                continue

            # If we're in a lockout TPS region, force 318/317 regardless of rpm
            if tps_val in lockout_tps_bins:
                apply.at[idx, col] = "318.0"
                rel_label = row_label.replace("Apply", "Release")
                if rel_label in rel_idx:
                    ridx = rel_idx[rel_label]
                    rel.at[ridx, col] = "317.0"
                continue

            try:
                mph = float(cell_str)
            except Exception:
                continue

            rpm = mph * k
            if rpm < rpm_floor:
                mph_new = rpm_floor / k
                mph_new = snap_1dp(mph_new)
                apply.at[idx, col] = f"{mph_new:.1f}"

    # Pass 2: enforce RELEASE >= APPLY + 1.1 and sentinel pairing
    for row_label, idx in rel_idx.items():
        if "Release" not in str(row_label):
            continue
        apply_label = row_label.replace("Release", "Apply")
        if apply_label not in apply_idx:
            continue
        a_idx = apply_idx[apply_label]

        for col in tps_cols:
            a_cell = str(apply.at[a_idx, col]).strip()
            r_cell = str(rel.at[idx, col]).strip()

            # If APPLY is sentinel, force RELEASE sentinel
            if is_sentinel(a_cell):
                rel.at[idx, col] = "317.0"
                continue

            # If RELEASE is sentinel but APPLY is numeric, bump RELEASE above APPLY
            try:
                a_val = float(a_cell)
            except Exception:
                continue

            if r_cell == "" or is_sentinel(r_cell):
                # create a RELEASE >= APPLY + 1.1
                r_val = snap_1dp(a_val + 1.1)
                rel.at[idx, col] = f"{r_val:.1f}"
                continue

            try:
                r_val = float(r_cell)
            except Exception:
                continue

            min_rel = a_val + 1.1
            if r_val < min_rel:
                r_val = snap_1dp(min_rel)
                rel.at[idx, col] = f"{r_val:.1f}"

    return apply, rel

def write_tsv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")


def main():
    # --- Load neutral SHIFT + TCC tables ---
    shift_up_path   = os.path.join(NEUTRAL_DIR, SHIFT_UP_NAME)
    shift_dn_path   = os.path.join(NEUTRAL_DIR, SHIFT_DOWN_NAME)
    tcc_apply_path  = os.path.join(NEUTRAL_DIR, TCC_APPLY_NAME)
    tcc_release_path= os.path.join(NEUTRAL_DIR, TCC_REL_NAME)

    up_neutral   = read_tsv(shift_up_path)
    down_neutral = read_tsv(shift_dn_path)
    tcc_apply    = read_tsv(tcc_apply_path)
    tcc_rel      = read_tsv(tcc_release_path)

    # --- Copy SHIFT tables unchanged into both modes for now ---
    for out_dir in (COMFORT_DIR, PERF_DIR):
        os.makedirs(out_dir, exist_ok=True)
        write_tsv(up_neutral,   os.path.join(out_dir, SHIFT_UP_NAME))
        write_tsv(down_neutral, os.path.join(out_dir, SHIFT_DOWN_NAME))

    # --- Build COMFORT TCC (soft ~1250 rpm lock floor) ---
    comfort_floor_rpm = 1250.0
    # Lockout region: TPS >= 81% (81, 87, 94, 100 bins)
    comfort_lockout_bins = [81.0, 87.0, 94.0, 100.0]

    c_apply, c_rel = adjust_tcc_tables(
        tcc_apply,
        tcc_rel,
        rpm_floor=comfort_floor_rpm,
        lockout_tps_bins=comfort_lockout_bins,
    )

    write_tsv(c_apply, os.path.join(COMFORT_DIR, TCC_APPLY_NAME))
    write_tsv(c_rel,   os.path.join(COMFORT_DIR, TCC_REL_NAME))

    # --- Build PERFORMANCE TCC (rowdy ~1500 rpm lock floor) ---
    perf_floor_rpm = 1500.0
    # Use same high-TPS lockout for now (we WANT it open under big throttle)
    perf_lockout_bins = [81.0, 87.0, 94.0, 100.0]

    p_apply, p_rel = adjust_tcc_tables(
        tcc_apply,
        tcc_rel,
        rpm_floor=perf_floor_rpm,
        lockout_tps_bins=perf_lockout_bins,
    )

    write_tsv(p_apply, os.path.join(PERF_DIR, TCC_APPLY_NAME))
    write_tsv(p_rel,   os.path.join(PERF_DIR, TCC_REL_NAME))

    print("[OK] Wrote COMFORT_SEED tables to:", COMFORT_DIR)
    print("[OK] Wrote PERF_SEED tables to:", PERF_DIR)
    print("      (SHIFT = neutral blend, TCC floors: comfort ~1250 rpm, perf ~1500 rpm)")

if __name__ == "__main__":
    main()
