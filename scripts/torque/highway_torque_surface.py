import argparse
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

"""
Highway torque surface with air/spark/knock and shift overlays.

Outputs (under --out-dir, default newlogs/highway_torque_surface):
  - torque_air_spark_surface__by_gear.csv
  - torque_gain__downshift_map.csv
  - torque_air_spark_gain__downshift_map.csv
  - TORQUE_GAIN__SUMMARY.txt
  - TORQUE_AIR_SPARK__SUMMARY.txt
  - ALL__shift_points.csv
  - shift_overlay__rpm_vs_pedal.csv
  - highway_torque_surface_outputs_small.zip (small bundle)
"""

RPM_BINS = np.arange(1200, 5000 + 250, 250)
PEDAL_BINS = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100], dtype=float)
GEAR_RATIOS = {4: 1.152, 5: 0.852, 6: 0.667}
MIN_SAMPLES_BIN = 20


def fail_missing(cols: List[str], df: pd.DataFrame, ctx: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{ctx}: missing required columns: {missing}")


def first_non_null(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    for c in cols:
        if c in df.columns:
            series = df[c].astype(float)
            out = out.fillna(series)
    return out


def load_prepped(prepped_dir: Path) -> pd.DataFrame:
    csvs = sorted(prepped_dir.glob("*__prepped.csv"))
    if not csvs:
        raise RuntimeError(f"No prepped logs found in {prepped_dir}")
    frames = []
    for p in csvs:
        df = pd.read_csv(p, low_memory=False)
        fail_missing(
            [
                "time_s",
                "speed_mph",
                "gear_actual__canon",
                "pedal_pct",
                "throttle_pct",
                "engine_rpm",
            ],
            df,
            p.name,
        )
        df = df.rename(columns={"gear_actual__canon": "gear_actual"})
        # Normalize slip column
        if "tcc_slip_rpm" in df.columns:
            df["tcc_slip"] = df["tcc_slip_rpm"].astype(float)
        elif "tcc_slip_rpm_fused" in df.columns:
            df["tcc_slip"] = df["tcc_slip_rpm_fused"].astype(float)
        elif "tcc_slip" in df.columns:
            df["tcc_slip"] = df["tcc_slip"].astype(float)
        else:
            raise RuntimeError(f"{p.name}: missing TCC slip column (tcc_slip_rpm[_fused])")
        df["file_name"] = p.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def unify_torque(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    torque_cols = [
        "Delivered Engine Torque",
        "Engine Torque",
        "Trans Engine Torque",
    ]
    axle_cols = [
        "Actual Axle Torque",
        "Driver Final Axle Torque Req",
        "Immediate Axle Torque Cmd",
    ]
    eng = first_non_null(df, torque_cols)
    if eng.isna().all():
        raise RuntimeError(f"No engine torque column found; tried {torque_cols}")
    axle = first_non_null(df, axle_cols)
    return eng, axle


def build_surface(df_all: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    eng_tq, axle_tq = unify_torque(df_all)
    df = df_all.copy()
    df["eng_torque"] = eng_tq
    df["axle_torque"] = axle_tq
    df["cyl_airmass"] = df["Cylinder Airmass"].astype(float) if "Cylinder Airmass" in df.columns else np.nan
    df["airflow_mass"] = first_non_null(df, ["Mass Airflow (SAE)", "Dynamic Airflow"])
    df["map_kpa"] = first_non_null(df, ["Intake Manifold Absolute Pressure (SAE)", "Manifold Absolute Pressure - Hi-Res"])
    df["ve_mgk_kpa"] = df["Volumetric Efficiency (mg•K/kPa)"].astype(float) if "Volumetric Efficiency (mg•K/kPa)" in df.columns else np.nan
    df["ve_airflow"] = df["Volumetric Efficiency Airflow"].astype(float) if "Volumetric Efficiency Airflow" in df.columns else np.nan
    df["spark_deg"] = df["Timing Advance (SAE)"].astype(float) if "Timing Advance (SAE)" in df.columns else np.nan
    df["knock_retard"] = df["Knock Retard"].astype(float) if "Knock Retard" in df.columns else np.nan
    df["lambda_eq"] = first_non_null(df, ["WB EQ Ratio 2 (SAE)", "Equivalence Ratio Commanded (SAE)"])

    # Drop rows with critical NaNs
    df = df.dropna(
        subset=[
            "eng_torque",
            "axle_torque",
            "cyl_airmass",
            "spark_deg",
            "knock_retard",
            "engine_rpm",
            "pedal_pct",
            "gear_actual",
        ]
    )

    # Highway filter
    df = df[
        (df["speed_mph"].between(60, 85))
        & (df["gear_actual"].isin([3, 4, 5, 6]))
    ].copy()
    if "Brake Pressure" in df.columns:
        df = df[df["Brake Pressure"].astype(float) <= 15]
    df["rpm_bin"] = pd.cut(df["engine_rpm"], RPM_BINS)
    df["pedal_bin"] = pd.cut(df["pedal_pct"], PEDAL_BINS)
    records = []
    for g in [3, 4, 5, 6]:
        df_g = df[df["gear_actual"] == g]
        if df_g.empty:
            continue
        grp = df_g.groupby(["rpm_bin", "pedal_bin"])
        for (rb, pb), sub in grp:
            n = len(sub)
            if n < MIN_SAMPLES_BIN:
                continue
            rpm_center = (rb.left + rb.right) / 2
            pedal_center = (pb.left + pb.right) / 2
            records.append(
                {
                    "gear": g,
                    "rpm_center": rpm_center,
                    "pedal_center": pedal_center,
                    "n_samples": n,
                    "eng_torque_mean": sub["eng_torque"].mean(),
                    "eng_torque_p50": sub["eng_torque"].median(),
                    "eng_torque_p75": np.nanpercentile(sub["eng_torque"], 75),
                    "axle_torque_mean": sub["axle_torque"].mean(),
                    "cyl_airmass_mean": sub["cyl_airmass"].mean(),
                    "cyl_airmass_p50": sub["cyl_airmass"].median(),
                    "airflow_mass_mean": sub["airflow_mass"].mean(),
                    "map_kpa_mean": sub["map_kpa"].mean(),
                    "ve_mgk_kpa_mean": sub["ve_mgk_kpa"].mean(),
                    "ve_airflow_mean": sub["ve_airflow"].mean(),
                    "spark_mean": sub["spark_deg"].mean(),
                    "spark_p50": sub["spark_deg"].median(),
                    "knock_p95": np.nanpercentile(sub["knock_retard"], 95),
                    "lambda_eq_mean": sub["lambda_eq"].mean(),
                    "coverage_ok": True,
                }
            )
    out_df = pd.DataFrame(records)
    out_df.to_csv(out_dir / "torque_air_spark_surface__by_gear.csv", index=False)
    # keep legacy surface name for compatibility
    out_df.to_csv(out_dir / "torque_surface__by_gear.csv", index=False)
    return out_df


def detect_shift_points(df_all: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    df_all = df_all.sort_values(["file_name", "time_s"]).copy()
    rows = []
    for fname, g in df_all.groupby("file_name"):
        gear = g["gear_actual"].astype(int).to_numpy()
        time = g["time_s"].to_numpy()
        speed = g["speed_mph"].astype(float).to_numpy()
        rpm = g["engine_rpm"].astype(float).to_numpy()
        pedal = g["pedal_pct"].astype(float).to_numpy()
        thr = g["throttle_pct"].astype(float).to_numpy()
        tcc = g["tcc_state"].astype(str).to_numpy() if "tcc_state" in g.columns else np.array([""] * len(g))
        mode = g["trans_mode"].astype(str).to_numpy() if "trans_mode" in g.columns else np.array([""] * len(g))
        for i in range(1, len(g)):
            if gear[i] != gear[i - 1]:
                rows.append(
                    {
                        "file_name": fname,
                        "time_shift": time[i],
                        "from_gear": int(gear[i - 1]),
                        "to_gear": int(gear[i]),
                        "speed_mph": speed[i],
                        "engine_rpm": rpm[i],
                        "pedal_pct": pedal[i],
                        "throttle_pct": thr[i],
                        "tcc_state": tcc[i],
                        "mode": mode[i],
                    }
                )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "ALL__shift_points.csv", index=False)
    return out_df


def build_shift_overlay(pts: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    pts = pts.copy()
    # Highway focus filters
    mask = (
        ((pts["from_gear"] == 6) & (pts["speed_mph"].between(60, 85)))
        | ((pts["from_gear"] == 5) & (pts["speed_mph"].between(40, 75)))
    )
    pts = pts[mask]
    pts["pedal_bin"] = pd.cut(pts["pedal_pct"], PEDAL_BINS)
    records = []
    for (fg, tg, pb), sub in pts.groupby(["from_gear", "to_gear", "pedal_bin"]):
        pedal_center = (pb.left + pb.right) / 2
        records.append(
            {
                "from_gear": fg,
                "to_gear": tg,
                "pedal_center": pedal_center,
                "shift_rpm_mean": sub["engine_rpm"].mean(),
                "shift_rpm_p50": sub["engine_rpm"].median(),
                "shift_speed_mean": sub["speed_mph"].mean(),
                "n_events": len(sub),
            }
        )
    out_df = pd.DataFrame(records)
    out_df.to_csv(out_dir / "shift_overlay__rpm_vs_pedal.csv", index=False)
    return out_df


def nearest_surface(surface: pd.DataFrame, gear: int, rpm_target: float, pedal_target: float):
    sub = surface[(surface["gear"] == gear) & (surface["coverage_ok"])]
    if sub.empty:
        return None
    # closest pedal then closest rpm
    idx_pedal = (sub["pedal_center"] - pedal_target).abs().idxmin()
    pedal_val = sub.loc[idx_pedal, "pedal_center"]
    sub_p = sub[sub["pedal_center"] == pedal_val]
    if sub_p.empty:
        return None
    idx_rpm = (sub_p["rpm_center"] - rpm_target).abs().idxmin()
    row = sub_p.loc[idx_rpm]
    return row.to_dict()


def torque_gain(surface: pd.DataFrame, overlay: pd.DataFrame, out_dir: Path):
    rows = []
    for _, r in overlay.iterrows():
        fg = int(r["from_gear"])
        tg = int(r["to_gear"])
        if (fg, tg) not in [(6, 5), (5, 4)]:
            continue
        pedal = float(r["pedal_center"])
        rpm_after = float(r["shift_rpm_mean"])
        # estimate rpm before if stayed in tall gear
        if fg in GEAR_RATIOS and tg in GEAR_RATIOS:
            rpm_before = rpm_after * (GEAR_RATIOS[tg] / GEAR_RATIOS[fg])
        else:
            rpm_before = rpm_after
        before = nearest_surface(surface, fg, rpm_before, pedal)
        after = nearest_surface(surface, tg, rpm_after, pedal)
        coverage_before = before is not None and bool(before.get("coverage_ok", True))
        coverage_after = after is not None and bool(after.get("coverage_ok", True))
        coverage_ok = coverage_before and coverage_after
        rows.append(
            {
                "from_gear": fg,
                "to_gear": tg,
                "pedal_center": pedal,
                "rpm_before_center": before.get("rpm_center") if coverage_before else np.nan,
                "rpm_after_center": after.get("rpm_center") if coverage_after else np.nan,
                "eng_torque_before": before.get("eng_torque_mean") if coverage_before else np.nan,
                "eng_torque_after": after.get("eng_torque_mean") if coverage_after else np.nan,
                "delta_eng_torque": (after.get("eng_torque_mean") - before.get("eng_torque_mean")) if coverage_ok else np.nan,
                "axle_torque_before": before.get("axle_torque_mean") if coverage_before else np.nan,
                "axle_torque_after": after.get("axle_torque_mean") if coverage_after else np.nan,
                "delta_axle_torque": (after.get("axle_torque_mean") - before.get("axle_torque_mean")) if coverage_ok else np.nan,
                "coverage_before_ok": coverage_before,
                "coverage_after_ok": coverage_after,
                "coverage_ok": coverage_ok,
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "torque_gain__downshift_map.csv", index=False)
    return out_df


def summarize_gain(gain_df: pd.DataFrame, out_dir: Path):
    lines = []
    for pair, speed_band in [((6, 5), "70-85"), ((5, 4), "50-70")]:
        sub = gain_df[
            (gain_df["from_gear"] == pair[0])
            & (gain_df["to_gear"] == pair[1])
            & (gain_df["pedal_center"].between(15, 40))
            & (gain_df["coverage_ok"])
        ]
        if sub.empty:
            lines.append(f"{pair[0]}->{pair[1]}: no valid bands (coverage_ok) in pedal 15-40%.")
            continue
        med_eng = sub["delta_eng_torque"].median()
        med_axle = sub["delta_axle_torque"].median()
        min_eng = sub["delta_eng_torque"].min()
        max_eng = sub["delta_eng_torque"].max()
        neg = (sub["delta_axle_torque"] < 0).sum()
        lines.append(
            f"{pair[0]}->{pair[1]} ({speed_band} mph, pedal 15-40%): "
            f"{len(sub)} bands, median Delta_eng={med_eng:.1f}, median Delta_axle={med_axle:.1f}, "
            f"min/max Delta_eng={min_eng:.1f}/{max_eng:.1f}, negative axle bands={neg}"
        )
    with open(out_dir / "TORQUE_GAIN__SUMMARY.txt", "w") as f:
        f.write("\n".join(lines))


def torque_air_spark_gain(surface: pd.DataFrame, overlay: pd.DataFrame, out_dir: Path):
    rows = []
    for _, r in overlay.iterrows():
        fg, tg = int(r["from_gear"]), int(r["to_gear"])
        if (fg, tg) not in [(6, 5), (5, 4)]:
            continue
        pedal = float(r["pedal_center"])
        rpm_after = float(r["shift_rpm_mean"])
        rpm_before = rpm_after * (GEAR_RATIOS.get(tg, 1) / GEAR_RATIOS.get(fg, 1)) if (fg in GEAR_RATIOS and tg in GEAR_RATIOS) else rpm_after
        before = nearest_surface(surface, fg, rpm_before, pedal)
        after = nearest_surface(surface, tg, rpm_after, pedal)
        ok_b = before is not None and bool(before.get("coverage_ok", True))
        ok_a = after is not None and bool(after.get("coverage_ok", True))
        ok = ok_b and ok_a
        def val(row, key):
            return row.get(key) if row and key in row else np.nan
        rows.append(
            {
                "from_gear": fg,
                "to_gear": tg,
                "pedal_center": pedal,
                "rpm_before_center": val(before, "rpm_center"),
                "rpm_after_center": val(after, "rpm_center"),
                "eng_torque_before": val(before, "eng_torque_mean"),
                "eng_torque_after": val(after, "eng_torque_mean"),
                "delta_eng_torque": val(after, "eng_torque_mean") - val(before, "eng_torque_mean") if ok else np.nan,
                "axle_torque_before": val(before, "axle_torque_mean"),
                "axle_torque_after": val(after, "axle_torque_mean"),
                "delta_axle_torque": val(after, "axle_torque_mean") - val(before, "axle_torque_mean") if ok else np.nan,
                "cyl_before": val(before, "cyl_airmass_mean"),
                "cyl_after": val(after, "cyl_airmass_mean"),
                "delta_cyl_air": val(after, "cyl_airmass_mean") - val(before, "cyl_airmass_mean") if ok else np.nan,
                "airflow_before": val(before, "airflow_mass_mean"),
                "airflow_after": val(after, "airflow_mass_mean"),
                "delta_airflow": val(after, "airflow_mass_mean") - val(before, "airflow_mass_mean") if ok else np.nan,
                "map_before": val(before, "map_kpa_mean"),
                "map_after": val(after, "map_kpa_mean"),
                "delta_map_kpa": val(after, "map_kpa_mean") - val(before, "map_kpa_mean") if ok else np.nan,
                "ve_before": val(before, "ve_mgk_kpa_mean"),
                "ve_after": val(after, "ve_mgk_kpa_mean"),
                "delta_ve": val(after, "ve_mgk_kpa_mean") - val(before, "ve_mgk_kpa_mean") if ok else np.nan,
                "spark_before": val(before, "spark_mean"),
                "spark_after": val(after, "spark_mean"),
                "delta_spark": val(after, "spark_mean") - val(before, "spark_mean") if ok else np.nan,
                "knock_before": val(before, "knock_p95"),
                "knock_after": val(after, "knock_p95"),
                "delta_knock": val(after, "knock_p95") - val(before, "knock_p95") if ok else np.nan,
                "lambda_before": val(before, "lambda_eq_mean"),
                "lambda_after": val(after, "lambda_eq_mean"),
                "delta_lambda": val(after, "lambda_eq_mean") - val(before, "lambda_eq_mean") if ok else np.nan,
                "coverage_ok": ok,
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "torque_air_spark_gain__downshift_map.csv", index=False)
    return out_df


def summarize_air_spark_gain(gain_df: pd.DataFrame, out_dir: Path):
    lines = []
    for pair, speed_band in [((6, 5), "70-85"), ((5, 4), "50-70")]:
        sub = gain_df[
            (gain_df["from_gear"] == pair[0])
            & (gain_df["to_gear"] == pair[1])
            & (gain_df["pedal_center"].between(15, 40))
            & (gain_df["coverage_ok"])
        ]
        if sub.empty:
            lines.append(f"{pair[0]}->{pair[1]}: no valid bands (coverage_ok) in pedal 15-40%.")
            continue
        def stat(col):
            return (sub[col].median(), sub[col].min(), sub[col].max())
        med_axle, min_axle, max_axle = stat("delta_axle_torque")
        med_cyl, min_cyl, max_cyl = stat("delta_cyl_air")
        med_map, min_map, max_map = stat("delta_map_kpa")
        med_spark, min_spark, max_spark = stat("delta_spark")
        med_knock, min_knock, max_knock = stat("delta_knock")
        neg_axle = sub[sub["delta_axle_torque"] < 0]["pedal_center"].tolist()
        worse_knock = sub[sub["delta_knock"] > 0]["pedal_center"].tolist()
        lines.append(
            f"{pair[0]}->{pair[1]} ({speed_band} mph, pedal 15-40%): "
            f"{len(sub)} bands, median Delta_axle={med_axle:.1f} (min/max {min_axle:.1f}/{max_axle:.1f}); "
            f"Delta_cyl_air med {med_cyl:.3f} (min/max {min_cyl:.3f}/{max_cyl:.3f}); "
            f"Delta_MAP med {med_map:.1f} (min/max {min_map:.1f}/{max_map:.1f}); "
            f"Delta_spark med {med_spark:.1f} (min/max {min_spark:.1f}/{max_spark:.1f}); "
            f"Delta_knock med {med_knock:.2f} (min/max {min_knock:.2f}/{max_knock:.2f})"
        )
        if neg_axle:
            lines.append(f"  Bands with negative Delta_axle_torque: {neg_axle}")
        if worse_knock:
            lines.append(f"  Bands with increased knock_p95: {worse_knock}")
    with open(out_dir / "TORQUE_AIR_SPARK__SUMMARY.txt", "w") as f:
        f.write("\n".join(lines))


def build_surface_speedspace(df_all: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    df = df_all[
        (df_all["speed_mph"].between(45, 90))
        & (df_all["gear_actual"].isin([3, 4, 5, 6]))
    ].copy()
    if "Brake Pressure" in df.columns:
        df = df[df["Brake Pressure"].astype(float) <= 15]
    df["speed_bin"] = pd.cut(df["speed_mph"], np.arange(45, 90 + 2.5, 2.5))
    df["pedal_bin"] = pd.cut(df["pedal_pct"], PEDAL_BINS)
    df["speed_center"] = df["speed_bin"].apply(lambda x: x.mid if pd.notna(x) else np.nan)
    df["pedal_center"] = df["pedal_bin"].apply(lambda x: x.mid if pd.notna(x) else np.nan)
    records = []
    for g in [3, 4, 5, 6]:
        df_g = df[df["gear_actual"] == g]
        grp = df_g.groupby(["speed_bin", "pedal_bin"])
        for (sb, pb), d in grp:
            if pd.isna(sb) or pd.isna(pb):
                continue
            if len(d) < MIN_SAMPLES_BIN:
                continue
            records.append(
                {
                    "gear": g,
                    "speed_center": sb.mid,
                    "pedal_center": pb.mid,
                    "n_samples": len(d),
                    "eng_torque_mean": d["eng_torque"].mean(),
                    "eng_torque_p50": d["eng_torque"].median(),
                    "axle_torque_mean": d["axle_torque"].mean(),
                    "cyl_airmass_mean": d["cyl_airmass"].mean(),
                    "cyl_airmass_p50": d["cyl_airmass"].median(),
                    "airflow_mass_mean": d["airflow_mass"].mean(),
                    "map_kpa_mean": d["map_kpa"].mean(),
                    "ve_mgk_kpa_mean": d["ve_mgk_kpa"].mean(),
                    "spark_mean": d["spark_deg"].mean(),
                    "spark_p50": d["spark_deg"].median(),
                    "knock_p95": d["knock_retard"].quantile(0.95),
                    "lambda_eq_mean": d["lambda_eq"].mean(),
                    "coverage_ok": True,
                }
            )
    surf = pd.DataFrame(records)
    surf.to_csv(out_dir / "torque_air_spark_surface__SPEEDSPACE.csv", index=False)
    return surf


def lookup_speed(surface: pd.DataFrame, gear: int, speed: float, pedal: float):
    sub = surface[(surface["gear"] == gear) & (surface["coverage_ok"])]
    if sub.empty:
        return None, False
    idx_speed = (sub["speed_center"] - speed).abs().idxmin()
    speed_val = sub.loc[idx_speed, "speed_center"]
    sub_s = sub[sub["speed_center"] == speed_val]
    if sub_s.empty:
        return None, False
    idx_p = (sub_s["pedal_center"] - pedal).abs().idxmin()
    row = sub_s.loc[idx_p]
    if not row["coverage_ok"]:
        return None, False
    return row.to_dict(), True


def torque_air_spark_gain_speed(surface: pd.DataFrame, overlay: pd.DataFrame, out_dir: Path):
    rows = []
    mask65 = (overlay["from_gear"] == 6) & (overlay["to_gear"] == 5) & overlay["shift_speed_mean"].between(70, 85)
    mask54 = (overlay["from_gear"] == 5) & (overlay["to_gear"] == 4) & overlay["shift_speed_mean"].between(50, 70)
    overlay_use = overlay[mask65 | mask54]
    for _, r in overlay_use.iterrows():
        fg, tg = int(r["from_gear"]), int(r["to_gear"])
        sp = float(r["shift_speed_mean"])
        pc = float(r["pedal_center"])
        before, okb = lookup_speed(surface, fg, sp, pc)
        after, oka = lookup_speed(surface, tg, sp, pc)
        if not (okb and oka):
            continue
        def val(row, key):
            return row.get(key) if row and key in row else np.nan
        rows.append(
            {
                "from_gear": fg,
                "to_gear": tg,
                "speed_center": sp,
                "pedal_center": pc,
                "eng_tq_before": val(before, "eng_torque_mean"),
                "eng_tq_after": val(after, "eng_torque_mean"),
                "delta_eng_tq": val(after, "eng_torque_mean") - val(before, "eng_torque_mean"),
                "axle_tq_before": val(before, "axle_torque_mean"),
                "axle_tq_after": val(after, "axle_torque_mean"),
                "delta_axle_tq": val(after, "axle_torque_mean") - val(before, "axle_torque_mean"),
                "cyl_before": val(before, "cyl_airmass_mean"),
                "cyl_after": val(after, "cyl_airmass_mean"),
                "delta_cyl": val(after, "cyl_airmass_mean") - val(before, "cyl_airmass_mean"),
                "airflow_before": val(before, "airflow_mass_mean"),
                "airflow_after": val(after, "airflow_mass_mean"),
                "delta_airflow": val(after, "airflow_mass_mean") - val(before, "airflow_mass_mean"),
                "map_before": val(before, "map_kpa_mean"),
                "map_after": val(after, "map_kpa_mean"),
                "delta_map": val(after, "map_kpa_mean") - val(before, "map_kpa_mean"),
                "ve_before": val(before, "ve_mgk_kpa_mean"),
                "ve_after": val(after, "ve_mgk_kpa_mean"),
                "delta_ve": val(after, "ve_mgk_kpa_mean") - val(before, "ve_mgk_kpa_mean"),
                "spark_before": val(before, "spark_mean"),
                "spark_after": val(after, "spark_mean"),
                "delta_spark": val(after, "spark_mean") - val(before, "spark_mean"),
                "knock_before": val(before, "knock_p95"),
                "knock_after": val(after, "knock_p95"),
                "delta_knock": val(after, "knock_p95") - val(before, "knock_p95"),
                "lambda_before": val(before, "lambda_eq_mean"),
                "lambda_after": val(after, "lambda_eq_mean"),
                "delta_lambda": val(after, "lambda_eq_mean") - val(before, "lambda_eq_mean"),
                "coverage_ok": True,
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "torque_air_spark_gain__SPEEDSPACE.csv", index=False)
    return out_df


def summarize_air_spark_gain_speed(gain_df: pd.DataFrame, out_dir: Path):
    lines = []
    for pair, label in [((6, 5), "6->5 (70-85 mph)"), ((5, 4), "5->4 (50-70 mph)")]:
        sub = gain_df[
            (gain_df["from_gear"] == pair[0])
            & (gain_df["to_gear"] == pair[1])
            & (gain_df["pedal_center"].between(15, 40))
            & (gain_df["coverage_ok"])
        ]
        lines.append(f"==== {label} ====")
        if sub.empty:
            lines.append("No valid bands.")
            lines.append("")
            continue
        for col in ["delta_axle_tq", "delta_cyl", "delta_map", "delta_spark", "delta_knock", "delta_lambda"]:
            arr = sub[col].dropna()
            if arr.empty:
                continue
            lines.append(f"{col}: median={arr.median():.3f}, min={arr.min():.3f}, max={arr.max():.3f}")
        neg_axle = sub[sub["delta_axle_tq"] < 0]["pedal_center"].tolist()
        if neg_axle:
            lines.append(f"Bands with negative Delta_axle_tq: {neg_axle}")
        lines.append("")
    (out_dir / "TORQUE_AIR_SPARK__SUMMARY__SPEEDSPACE.txt").write_text("\n".join(lines), encoding="utf-8")


def make_zip(out_dir: Path):
    version_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    small = out_dir / f"highway_torque_surface_outputs_small_v{version_stamp}.zip"
    with zipfile.ZipFile(small, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in [
            "torque_surface__by_gear.csv",
            "torque_air_spark_surface__by_gear.csv",
            "torque_air_spark_surface__SPEEDSPACE.csv",
            "ALL__shift_points.csv",
            "shift_overlay__rpm_vs_pedal.csv",
            "torque_gain__downshift_map.csv",
            "torque_air_spark_gain__downshift_map.csv",
            "torque_air_spark_gain__SPEEDSPACE.csv",
            "TORQUE_GAIN__SUMMARY.txt",
            "TORQUE_AIR_SPARK__SUMMARY.txt",
            "TORQUE_AIR_SPARK__SUMMARY__SPEEDSPACE.txt",
        ]:
            p = out_dir / name
            if p.exists():
                zf.write(p, p.name)
    print(f"[OK] Wrote {small} ({small.stat().st_size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torque surface + shift overlay + torque gain")
    parser.add_argument("--prepped-dir", default="newlogs/highway_MAX_analysis/prepped", help="Folder with *_prepped.csv")
    parser.add_argument("--out-dir", default="newlogs/highway_torque_surface", help="Output folder")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all = load_prepped(Path(args.prepped_dir))
    # Add unified torque columns once so every downstream function (including the
    # speed-space builder) has them available.
    eng_tq_all, axle_tq_all = unify_torque(df_all)
    df_all["eng_torque"] = eng_tq_all
    df_all["axle_torque"] = axle_tq_all
    df_all["cyl_airmass"] = df_all["Cylinder Airmass"].astype(float) if "Cylinder Airmass" in df_all.columns else np.nan
    df_all["airflow_mass"] = first_non_null(df_all, ["Mass Airflow (SAE)", "Dynamic Airflow"])
    df_all["map_kpa"] = first_non_null(df_all, ["Intake Manifold Absolute Pressure (SAE)", "Manifold Absolute Pressure - Hi-Res"])
    df_all["ve_mgk_kpa"] = (
        df_all["Volumetric Efficiency (mg�?�K/kPa)"].astype(float)
        if "Volumetric Efficiency (mg�?�K/kPa)" in df_all.columns
        else np.nan
    )
    df_all["ve_airflow"] = (
        df_all["Volumetric Efficiency Airflow"].astype(float)
        if "Volumetric Efficiency Airflow" in df_all.columns
        else np.nan
    )
    df_all["spark_deg"] = df_all["Timing Advance (SAE)"].astype(float) if "Timing Advance (SAE)" in df_all.columns else np.nan
    df_all["knock_retard"] = df_all["Knock Retard"].astype(float) if "Knock Retard" in df_all.columns else np.nan
    df_all["lambda_eq"] = first_non_null(df_all, ["WB EQ Ratio 2 (SAE)", "Equivalence Ratio Commanded (SAE)"])
    surface_path = out_dir / "torque_surface__by_gear.csv"
    surface_air_path = out_dir / "torque_air_spark_surface__by_gear.csv"
    surface = None
    if surface_air_path.exists():
        surface = pd.read_csv(surface_air_path)
    elif surface_path.exists():
        surface = pd.read_csv(surface_path)
    # Rebuild if missing enriched columns
    needed_cols = {"cyl_airmass_mean", "airflow_mass_mean", "map_kpa_mean", "spark_mean", "knock_p95"}
    if (surface is None) or (not needed_cols.issubset(set(surface.columns))):
        surface = build_surface(df_all, out_dir)
    else:
        # also save enriched copy for consistency
        surface.to_csv(surface_air_path, index=False)

    shift_pts = detect_shift_points(df_all, out_dir)
    overlay = build_shift_overlay(shift_pts, out_dir)
    gain = torque_gain(surface, overlay, out_dir)
    summarize_gain(gain, out_dir)
    gain_air = torque_air_spark_gain(surface, overlay, out_dir)
    summarize_air_spark_gain(gain_air, out_dir)
    surface_speed = build_surface_speedspace(df_all, out_dir)
    gain_air_speed = torque_air_spark_gain_speed(surface_speed, overlay, out_dir)
    summarize_air_spark_gain_speed(gain_air_speed, out_dir)
    make_zip(out_dir)
    print("[DONE] Torque surface and shift overlay complete.")
