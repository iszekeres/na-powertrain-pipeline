#!/usr/bin/env python3
"""
TCC Truth Check
---------------
Given a single HP Tuners log CSV, build a canonical view of TCC slip and state
by gear, using flexible column detection and simple lock/partial/open rules.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for cand in candidates:
        if cand in df.columns:
            return cand
    return None


def require_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c not in df.columns]


def build_time(df: pd.DataFrame, time_col: str) -> Tuple[pd.Series, pd.Series]:
    time = pd.to_numeric(df[time_col], errors="coerce")
    time = time - time.min()
    dt = time.shift(-1) - time
    med_dt = dt.median(skipna=True)
    if pd.isna(med_dt):
        med_dt = 0.0
    dt = dt.fillna(med_dt)
    dt[dt < 0] = 0.0
    return time, dt


def build_slip(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    slip_col = pick_col(df, ["TCC Slip", "Trans Slip RPM"])
    if slip_col:
        slip = pd.to_numeric(df[slip_col], errors="coerce")
        src = slip_col
    else:
        eng_col = pick_col(df, ["Engine RPM (SAE)", "Engine RPM"])
        turb_col = pick_col(df, ["Trans Turbine RPM", "Turbine Speed"])
        if eng_col and turb_col:
            eng = pd.to_numeric(df[eng_col], errors="coerce")
            turb = pd.to_numeric(df[turb_col], errors="coerce")
            slip = eng - turb
            src = f"{eng_col} - {turb_col}"
        else:
            missing = []
            if eng_col is None:
                missing.append("Engine RPM (SAE)")
            if turb_col is None:
                missing.append("Trans Turbine RPM")
            raise RuntimeError(f"Missing slip sources: {missing}")
    slip[np.abs(slip) > 5000] = np.nan
    return slip, src


def build_line_pressure(df: pd.DataFrame) -> Tuple[pd.Series, Optional[str]]:
    cand = ["TCC Line Pressure", "TCC Line Pressure (SAE)", "TCC Line", "TCC Line kPa"]
    lp_col = pick_col(df, cand)
    if lp_col:
        line = pd.to_numeric(df[lp_col], errors="coerce")
    else:
        line = pd.Series(np.nan, index=df.index)
    return line, lp_col


def classify_tcc_state(slip: pd.Series, line: pd.Series) -> pd.Series:
    state = pd.Series("PARTIAL", index=slip.index, dtype=object)
    abs_slip = slip.abs()
    locked_mask = abs_slip <= 50
    open_mask = abs_slip >= 100
    if line.notna().any():
        locked_mask = locked_mask & ((line >= 200) | line.isna())
        open_mask = open_mask & ((line <= 50) | line.isna())
    state[locked_mask] = "LOCKED"
    state[open_mask] = "OPEN"
    return state


def summarize_by_gear(df: pd.DataFrame, out_csv: Path, out_txt: Path) -> None:
    records = []
    gears = range(1, 7)
    states = ["LOCKED", "PARTIAL", "OPEN"]
    abs_slip = df["tcc_slip_calc_rpm"].abs()
    for g in gears:
        df_g = df[df["gear_actual"] == g]
        for st in states:
            sub = df_g[df_g["tcc_state"] == st]
            records.append(
                {
                    "gear": g,
                    "tcc_state": st,
                    "n_samples": len(sub),
                    "total_time_s": sub["dt_s"].sum(),
                    "median_abs_slip_rpm": abs_slip.loc[sub.index].median(),
                    "max_abs_slip_rpm": abs_slip.loc[sub.index].max(),
                }
            )
    pd.DataFrame(records).to_csv(out_csv, index=False)

    lines = []
    for g in gears:
        df_g = df[df["gear_actual"] == g]
        if df_g.empty:
            lines.append(f"Gear {g}: no samples.")
            continue
        total_time = df_g["dt_s"].sum()
        parts = []
        for st in states:
            sub = df_g[df_g["tcc_state"] == st]
            time_s = sub["dt_s"].sum()
            pct = (time_s / total_time * 100.0) if total_time > 0 else 0.0
            slip_weighted = (sub["tcc_slip_calc_rpm"].abs() * sub["dt_s"]).sum()
            avg_slip = slip_weighted / time_s if time_s > 0 else np.nan
            median_slip = sub["tcc_slip_calc_rpm"].abs().median()
            max_slip = sub["tcc_slip_calc_rpm"].abs().max()
            parts.append(f"{st}: {pct:.1f}% (w-avg slip {avg_slip:.1f} rpm, med {median_slip:.1f}, max {max_slip:.1f})")
        lines.append(f"Gear {g}: " + "; ".join(parts))
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Quick console summary for highway gears
    for g in [3, 4, 5, 6]:
        df_g = df[df["gear_actual"] == g]
        if df_g.empty:
            continue
        total_time = df_g["dt_s"].sum()
        locked_time = df_g[df_g["tcc_state"] == "LOCKED"]["dt_s"].sum()
        partial_time = df_g[df_g["tcc_state"] == "PARTIAL"]["dt_s"].sum()
        open_time = df_g[df_g["tcc_state"] == "OPEN"]["dt_s"].sum()
        print(
            f"Gear {g}: LOCKED {locked_time/total_time*100 if total_time>0 else 0:.1f}% | "
            f"PARTIAL {partial_time/total_time*100 if total_time>0 else 0:.1f}% | "
            f"OPEN {open_time/total_time*100 if total_time>0 else 0:.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Canonical TCC slip/state checker for HP Tuners logs.")
    parser.add_argument("--log", required=True, help="Path to a single HP Tuners CSV log")
    args = parser.parse_args()

    log_path = Path(args.log).resolve()
    if not log_path.exists():
        print(f"[ERROR] Log not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(log_path, low_memory=False)

    required_backbone = ["Offset"]
    missing = require_columns(df, required_backbone)

    gear_col = pick_col(df, ["Trans Current Gear", "Trans Current Gear  ", "TransCurGear"])
    if gear_col is None:
        missing.append("Trans Current Gear")

    slip_candidates = ["TCC Slip", "Trans Slip RPM"]
    slip_available = any(c in df.columns for c in slip_candidates)
    eng_present = pick_col(df, ["Engine RPM (SAE)", "Engine RPM"])
    turb_present = pick_col(df, ["Trans Turbine RPM", "Turbine Speed"])

    if not slip_available and not (eng_present and turb_present):
        missing.append("TCC Slip / Trans Slip RPM or Engine RPM (SAE) + Trans Turbine RPM")

    if missing:
        print("[ERROR] Missing required columns:", ", ".join(missing), file=sys.stderr)
        sys.exit(1)

    time_s, dt_s = build_time(df, "Offset")
    gear_actual = pd.to_numeric(df[gear_col], errors="coerce").astype("Int64")

    slip, slip_src = build_slip(df)
    line, line_src = build_line_pressure(df)

    tcc_state = classify_tcc_state(slip, line)

    df_out = pd.DataFrame(
        {
            "time_s": time_s,
            "dt_s": dt_s,
            "gear_actual": gear_actual,
            "tcc_slip_calc_rpm": slip,
            "tcc_line_kpa": line,
            "tcc_state": tcc_state,
        }
    )

    df_out = df_out[
        df_out["gear_actual"].between(1, 6)
        & df_out["dt_s"].notna()
        & df_out["dt_s"].astype(float).pipe(np.isfinite)
    ].copy()

    print(f"[INFO] Gear column: {gear_col}")
    print(f"[INFO] Slip source: {slip_src}")
    if line_src:
        print(f"[INFO] TCC line pressure: {line_src}")
    else:
        print("[INFO] TCC line pressure: not available (using slip-only classification)")

    out_csv = log_path.parent / "tcc_truth_by_gear.csv"
    out_txt = log_path.parent / "TCC_TRUTH_SUMMARY.txt"
    summarize_by_gear(df_out, out_csv, out_txt)
    print(f"[OK] Wrote {out_csv}")
    print(f"[OK] Wrote {out_txt}")


if __name__ == "__main__":
    main()
