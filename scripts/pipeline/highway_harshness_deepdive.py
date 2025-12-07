import argparse
import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np


def load_heatmap(path: Path) -> pd.DataFrame:
    print(f"[info] Loading harshness heatmap: {path}")
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    expected_cols = ["speed_bin_center", "pedal_bin_center"]
    for col in expected_cols:
        if col not in df.columns:
            raise SystemExit(f"[error] Expected column '{col}' in heatmap, found: {df.columns.tolist()}")
    # Try to detect from/to gear columns for later
    lower = {c.lower(): c for c in df.columns}
    from_col = None
    to_col = None
    for key, col in lower.items():
        if "gear" in key and "from" in key:
            from_col = col
        if "gear" in key and "to" in key:
            to_col = col
    return df, from_col, to_col


def build_bin_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.sort(np.array(centers, dtype=float))
    if centers.size == 0:
        raise ValueError("No bin centers provided")
    if centers.size == 1:
        step = 10.0
        return np.array([centers[0] - step / 2.0, centers[0] + step / 2.0])
    diffs = np.diff(centers)
    mean_step = float(np.median(diffs))
    left = centers[0] - mean_step / 2.0
    right = centers[-1] + mean_step / 2.0
    internal = (centers[:-1] + centers[1:]) / 2.0
    return np.concatenate([[left], internal, [right]])


def compute_occupancy(prepped_dir: Path, heatmap_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap, _, _ = load_heatmap(heatmap_path)

    speed_centers = np.sort(heatmap["speed_bin_center"].dropna().unique().astype(float))
    pedal_centers = np.sort(heatmap["pedal_bin_center"].dropna().unique().astype(float))

    speed_edges = build_bin_edges(speed_centers)
    pedal_edges = build_bin_edges(pedal_centers)

    print(f"[info] Speed centers: {speed_centers}")
    print(f"[info] Pedal centers: {pedal_centers}")

    occ_accum = {}

    csv_files = sorted(prepped_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"[error] No CSV files found in prepped dir: {prepped_dir}")

    for fpath in csv_files:
        print(f"[info] Scanning for occupancy in: {fpath.name}")
        try:
            for chunk in pd.read_csv(fpath, chunksize=200000):
                cols_lower = {c.lower(): c for c in chunk.columns}
                if "speed_mph" not in cols_lower or "pedal_pct" not in cols_lower:
                    print(f"[warn] Skipping chunk in {fpath.name}: missing speed_mph/pedal_pct")
                    continue
                speed_col = cols_lower["speed_mph"]
                pedal_col = cols_lower["pedal_pct"]

                speed = pd.to_numeric(chunk[speed_col], errors="coerce")
                pedal = pd.to_numeric(chunk[pedal_col], errors="coerce")

                mask = speed.notna() & pedal.notna()
                if not mask.any():
                    continue

                # Time delta
                time_col = None
                for key in ("time_s", "offset", "time", "elapsed_time"):
                    if key in cols_lower:
                        time_col = cols_lower[key]
                        break

                if time_col is not None:
                    t = pd.to_numeric(chunk[time_col], errors="coerce")
                    dt = t.diff().abs()
                    median_dt = dt.median()
                    if not pd.isna(median_dt) and median_dt > 0:
                        dt = dt.fillna(median_dt)
                    else:
                        dt = dt.fillna(0.1)
                    dt = dt.clip(lower=0.0, upper=1.0)
                else:
                    dt = pd.Series(0.1, index=chunk.index)

                speed_v = speed[mask].to_numpy()
                pedal_v = pedal[mask].to_numpy()
                dt_v = dt[mask].to_numpy()

                s_idx = np.digitize(speed_v, speed_edges) - 1
                p_idx = np.digitize(pedal_v, pedal_edges) - 1

                valid = (
                    (s_idx >= 0)
                    & (s_idx < speed_centers.size)
                    & (p_idx >= 0)
                    & (p_idx < pedal_centers.size)
                )
                if not np.any(valid):
                    continue

                s_idx = s_idx[valid]
                p_idx = p_idx[valid]
                dt_v = dt_v[valid]

                s_bins = speed_centers[s_idx]
                p_bins = pedal_centers[p_idx]

                occ_df = pd.DataFrame(
                    {"speed_bin_center": s_bins, "pedal_bin_center": p_bins, "dt_s": dt_v}
                )
                grouped = occ_df.groupby(["speed_bin_center", "pedal_bin_center"]).agg(
                    total_time_s=("dt_s", "sum"),
                    n_samples=("dt_s", "size"),
                )
                for (s_bc, p_bc), row in grouped.iterrows():
                    key = (float(s_bc), float(p_bc))
                    total_time_s, n_samples = occ_accum.get(key, (0.0, 0))
                    occ_accum[key] = (
                        total_time_s + float(row["total_time_s"]),
                        n_samples + int(row["n_samples"]),
                    )
        except Exception as e:
            print(f"[warn] Failed while reading {fpath.name}: {e}")

    if not occ_accum:
        raise SystemExit("[error] No occupancy data accumulated; check inputs/columns.")

    occ_rows = []
    for (s_bc, p_bc), (total_time_s, n_samples) in sorted(occ_accum.items()):
        occ_rows.append(
            {
                "speed_bin_center": s_bc,
                "pedal_bin_center": p_bc,
                "total_time_s": total_time_s,
                "n_samples": n_samples,
            }
        )
    occ_df = pd.DataFrame(occ_rows)
    occ_path = out_dir / "shift_harshness_occupancy_from_logs.csv"
    occ_df.to_csv(occ_path, index=False)
    print(f"[info] Wrote occupancy grid: {occ_path}")

    merged = heatmap.merge(
        occ_df, on=["speed_bin_center", "pedal_bin_center"], how="left"
    )
    merged["total_time_s"] = merged["total_time_s"].fillna(0.0)
    merged["n_samples"] = merged["n_samples"].fillna(0).astype(int)
    merged_path = out_dir / "shift_harshness_heatmap__with_occupancy.csv"
    merged.to_csv(merged_path, index=False)
    print(f"[info] Wrote heatmap+occupancy: {merged_path}")


def load_tcc_truth_table(path: Path) -> pd.DataFrame:
    """
    Load tcc_truth_by_gear.csv that may be formatted with rows per gear/state
    (gear, tcc_state, total_time_s, ...) and produce a wide table with
    pct_locked/partial/open per gear.
    """
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    gear_col = None
    for key, col in cols_lower.items():
        if "gear" in key:
            gear_col = col
            break
    if gear_col is None:
        raise SystemExit("[error] Could not find 'gear' column in tcc_truth file.")

    # If already has pct_locked columns, return as-is
    locked_col = None
    partial_col = None
    open_col = None
    for key, col in cols_lower.items():
        if "locked" in key:
            locked_col = col
        if "partial" in key:
            partial_col = col
        if "open" in key:
            open_col = col
    if locked_col and partial_col and open_col:
        out = df[[gear_col, locked_col, partial_col, open_col]].copy()
        out = out.rename(
            columns={
                gear_col: "gear",
                locked_col: "pct_locked",
                partial_col: "pct_partial",
                open_col: "pct_open",
            }
        )
        out["gear"] = pd.to_numeric(out["gear"], errors="coerce").astype("Int64")
        out = out.dropna(subset=["gear"])
        out["gear"] = out["gear"].astype(int)
        return out

    # Otherwise expect tall format with tcc_state and total_time_s
    state_col = cols_lower.get("tcc_state")
    time_col = cols_lower.get("total_time_s")
    if state_col is None or time_col is None:
        raise SystemExit(
            "[error] tcc_truth file missing locked/partial/open pct columns AND missing tcc_state/total_time_s fields. "
            f"Columns seen: {df.columns.tolist()}"
        )
    df = df[[gear_col, state_col, time_col]].copy()
    df[gear_col] = pd.to_numeric(df[gear_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[gear_col])
    df[gear_col] = df[gear_col].astype(int)
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    pivot = (
        df.pivot_table(
            index=gear_col,
            columns=state_col,
            values=time_col,
            aggfunc="sum",
        )
        .fillna(0.0)
        .reset_index()
    )
    pivot.columns = [str(c) for c in pivot.columns]
    pivot = pivot.rename(columns={gear_col: "gear"})
    for st in ["LOCKED", "PARTIAL", "OPEN", "locked", "partial", "open"]:
        if st in pivot.columns:
            pivot[st.upper()] = pivot[st]
    pivot["total"] = pivot[[c for c in pivot.columns if c.upper() in ["LOCKED", "PARTIAL", "OPEN"]]].sum(axis=1)
    for st in ["LOCKED", "PARTIAL", "OPEN"]:
        col = st
        if col in pivot.columns:
            pivot[f"pct_{st.lower()}"] = np.where(pivot["total"] > 0, pivot[col] / pivot["total"] * 100.0, np.nan)
        else:
            pivot[f"pct_{st.lower()}"] = math.nan
    return pivot[["gear", "pct_locked", "pct_partial", "pct_open"]]


def tcc_overlay(heatmap_path: Path, tcc_truth_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap, from_col, to_col = load_heatmap(heatmap_path)
    if to_col is None:
        # Fallback: try a generic 'gear' column in heatmap
        candidates = [c for c in heatmap.columns if "gear" in c.lower()]
        if len(candidates) == 1:
            to_col = candidates[0]
            print(f"[info] Using '{to_col}' as gear column for TCC overlay.")
        else:
            raise SystemExit(
                "[error] Could not detect 'to_gear' column in heatmap for TCC overlay."
            )

    print(f"[info] Using '{to_col}' as post-shift gear column for TCC overlay.")

    tcc_small = load_tcc_truth_table(tcc_truth_path)
    print(f"[info] Loaded TCC truth: {tcc_truth_path}")

    heatmap["gear_for_tcc"] = pd.to_numeric(heatmap[to_col], errors="coerce").astype(
        "Int64"
    )

    merged = heatmap.merge(
        tcc_small, left_on="gear_for_tcc", right_on="gear", how="left"
    )
    merged_path = out_dir / "shift_harshness_heatmap__tcc_overlay.csv"
    merged.to_csv(merged_path, index=False)
    print(f"[info] Wrote TCC overlay heatmap: {merged_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Deep-dive tools for highway harshness: occupancy and TCC overlay."
    )
    parser.add_argument(
        "--prepped-dir",
        type=str,
        default="newlogs\\highway_MAX_analysis\\prepped",
        help="Directory with prepped logs (for occupancy).",
    )
    parser.add_argument(
        "--harshness-heatmap",
        type=str,
        required=True,
        help="Path to shift_harshness_heatmap.csv.",
    )
    parser.add_argument(
        "--tcc-truth",
        type=str,
        help="Path to tcc_truth_by_gear.csv for TCC overlay.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="highway_super_analysis__HARSHNESS",
        help="Output directory for derived CSVs.",
    )
    parser.add_argument(
        "--do-occupancy",
        action="store_true",
        help="Compute speed/pedal occupancy from prepped logs and merge into heatmap.",
    )
    parser.add_argument(
        "--do-tcc-overlay",
        action="store_true",
        help="Overlay TCC locked/partial/open percentages onto heatmap bins (by post-shift gear).",
    )

    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    heatmap_path = Path(args.harshness_heatmap)

    did_anything = False

    if args.do_occupancy:
        prepped_dir = Path(args.prepped_dir)
        compute_occupancy(prepped_dir, heatmap_path, out_dir)
        did_anything = True

    if args.do_tcc_overlay:
        if not args.tcc_truth:
            raise SystemExit(
                "[error] --do-tcc-overlay requires --tcc-truth=path\\to\\tcc_truth_by_gear.csv"
            )
        tcc_truth_path = Path(args.tcc_truth)
        tcc_overlay(heatmap_path, tcc_truth_path, out_dir)
        did_anything = True

    if not did_anything:
        parser.print_help()
        print(
            "\n[error] You must pass at least one of: --do-occupancy, --do-tcc-overlay"
        )


if __name__ == "__main__":
    main()
