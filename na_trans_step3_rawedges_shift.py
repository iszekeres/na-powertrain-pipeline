import glob
import os

import numpy as np
import pandas as pd

# LOG-FIRST RAWEDGES builder for SHIFT tables.
#
# Gating:
#   - speed_mph__canon >= MIN_SPEED_MPH
#   - brake <= 0.5 (brake OFF; binary 0/1)
#   - gears 1–6 only
#   - ±1 gear steps (1->2, 2->3, ..., and the matching downs)
#
# Strict/no-fallback on required columns.

TPS_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]
TPS_COLS = [str(v) for v in TPS_AXIS]

MIN_HITS_PER_BIN = 1
MIN_SPEED_MPH = 3.0  # ignore ultra-low-speed junk

REQ_COLS = [
    "time_s",
    "speed_mph__canon",
    "throttle_pct__canon",
    "gear_actual__canon",
    "brake",  # binary 0/1 from CLEAN_FULL
]


def snap_tps(val):
    if pd.isna(val):
        return None
    v = float(val)
    if v < 0.0:
        v = 0.0
    if v > 100.0:
        v = 100.0
    return min(TPS_AXIS, key=lambda a: abs(a - v))


def build_shift_table(edges_df: pd.DataFrame, direction: str) -> pd.DataFrame:
    if direction == "UP":
        pairs = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    else:
        pairs = [(2, 1), (3, 2), (4, 3), (5, 4), (6, 5)]

    rows = []
    for g_from, g_to in pairs:
        label = f"{g_from} -> {g_to} Shift"
        row = {"mph": label}

        sub = edges_df[
            (edges_df["direction"] == direction)
            & (edges_df["from_gear"] == g_from)
            & (edges_df["to_gear"] == g_to)
        ]

        for tps in TPS_AXIS:
            col = str(tps)
            sub_bin = sub[sub["tps_bin"] == tps]
            if len(sub_bin) < MIN_HITS_PER_BIN:
                row[col] = ""
            else:
                row[col] = float(sub_bin["speed_mph"].median())

        row["%"] = ""
        rows.append(row)

    return pd.DataFrame(rows, columns=["mph"] + TPS_COLS + ["%"])


def main() -> None:
    clean_dir = os.path.join("newlogs", "cleaned")
    shift_dir = os.path.join("newlogs", "output", "01_tables", "shift")
    debug_dir = os.path.join("newlogs", "output", "DEBUG")

    os.makedirs(shift_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    pattern = os.path.join(clean_dir, "__trans_focus__clean_FULL__*.csv")
    files = sorted(glob.glob(pattern))

    print(f"[INFO] CLEAN_FULL dir: {clean_dir}")
    if not files:
        raise SystemExit("[ERROR] No CLEAN_FULL files found in newlogs/cleaned")

    print(f"[INFO] Found {len(files)} CLEAN_FULL file(s):")
    for f in files:
        print("   ", os.path.basename(f))

    # Strict preflight
    missing_any = False
    missing_report = []
    for f in files:
        head = pd.read_csv(f, nrows=0)
        missing = [c for c in REQ_COLS if c not in head.columns]
        if missing:
            missing_any = True
            missing_report.append((os.path.basename(f), missing))

    if missing_any:
        print("[ERROR] Missing required columns in CLEAN_FULL files:")
        for fname, missing in missing_report:
            print(f"   {fname}: missing {missing}")
        raise SystemExit("[ERROR] Aborting RAWEDGES build due to missing columns.")

    edges = []
    for f in files:
        fname = os.path.basename(f)
        df = pd.read_csv(f)

        n_rows = len(df)
        gears = df["gear_actual__canon"].dropna().unique()
        brk_vals = df["brake"].dropna().unique()
        print(
            f"\n[DEBUG] {fname}: rows={n_rows}, "
            f"unique gear_actual__canon={sorted(gears.tolist()) if len(gears) else []}, "
            f"unique brake={sorted(brk_vals.tolist()) if len(brk_vals) else []}"
        )

        # Speed + brake gate: moving AND brake off
        brk = pd.to_numeric(df["brake"], errors="coerce").fillna(0.0)
        mask = (df["speed_mph__canon"] >= MIN_SPEED_MPH) & (brk <= 0.5)
        df2 = df.loc[
            mask,
            ["time_s", "speed_mph__canon", "throttle_pct__canon", "gear_actual__canon"],
        ].copy()

        if df2.empty:
            print(
                f"  [WARN] {fname}: no rows after speed/brake gate "
                f"(speed >= {MIN_SPEED_MPH} mph and brake <= 0.5); skipping."
            )
            continue

        df2 = df2.sort_values("time_s")
        df2["gear_ff"] = df2["gear_actual__canon"].ffill()
        df2["speed_ff"] = df2["speed_mph__canon"].ffill()

        n2 = len(df2)
        per_file_edges = 0

        for i in range(1, n2):
            prev_g = df2["gear_ff"].iat[i - 1]
            cur_g = df2["gear_ff"].iat[i]

            if pd.isna(prev_g) or pd.isna(cur_g):
                continue

            try:
                prev_g = int(prev_g)
                cur_g = int(cur_g)
            except Exception:
                continue

            if not (1 <= prev_g <= 6 and 1 <= cur_g <= 6):
                continue

            step = cur_g - prev_g
            if step == 1:
                direction = "UP"
            elif step == -1:
                direction = "DOWN"
            else:
                continue  # skip multi-gear jumps

            speed = float(df2["speed_ff"].iat[i])
            tps_raw = df2["throttle_pct__canon"].iat[i]
            tps_bin = snap_tps(tps_raw)

            edges.append(
                {
                    "file": fname,
                    "direction": direction,
                    "from_gear": prev_g,
                    "to_gear": cur_g,
                    "speed_mph": speed,
                    "tps_raw": float(tps_raw) if not pd.isna(tps_raw) else np.nan,
                    "tps_bin": tps_bin,
                }
            )
            per_file_edges += 1

        print(f"  [OK] {fname}: {per_file_edges} edges after speed+brake gate")

    if not edges:
        debug_path = os.path.join(debug_dir, "SHIFT_EDGES__ZERO_CASE__GEAR_TIMELINE.tsv")
        timelines = []
        for f in files:
            fname = os.path.basename(f)
            df = pd.read_csv(f)
            df_small = df[["time_s", "speed_mph__canon", "gear_actual__canon", "brake"]].copy()
            df_small["file"] = fname
            timelines.append(df_small.head(500))
        if timelines:
            out_df = pd.concat(timelines, ignore_index=True)
            out_df.to_csv(debug_path, sep="\t", index=False)
            print(f"\n[ERROR] No valid gear edges found even with speed+brake gating.")
            print(f"       Gear timeline dumped to: {debug_path}")
        else:
            print("\n[ERROR] No valid gear edges found and no rows to dump.")
        raise SystemExit("[ERROR] RAWEDGES cannot proceed without any gear edges.")

    edges_df = pd.DataFrame(edges)
    debug_edges_path = os.path.join(debug_dir, "SHIFT_EDGES__RAW_SCAN.tsv")
    edges_df.to_csv(debug_edges_path, sep="\t", index=False)
    print(f"\n[OK] Wrote raw edge list to {debug_edges_path}")

    print("\n[SUMMARY] Edge counts by direction/from_gear:")
    print(edges_df.groupby(["direction", "from_gear"])["speed_mph"].count())

    print("\n[SUMMARY] Edge counts per (direction, from_gear, tps_bin):")
    print(edges_df.groupby(["direction", "from_gear", "tps_bin"])["speed_mph"].count())

    print("\n[STEP] Building RAWEDGES SHIFT UP table...")
    up_table = build_shift_table(edges_df, "UP")

    print("[STEP] Building RAWEDGES SHIFT DOWN table...")
    down_table = build_shift_table(edges_df, "DOWN")

    # Snap TPS columns to 0.1 mph, preserve blanks
    for df in (up_table, down_table):
        for col in df.columns:
            if col in TPS_COLS:
                df[col] = df[col].apply(
                    lambda v: "" if (pd.isna(v) or v == "") else f"{float(v):.1f}"
                )

    up_out = os.path.join(shift_dir, "SHIFT_TABLES__UP__Throttle17__RAWEDGES.tsv")
    down_out = os.path.join(shift_dir, "SHIFT_TABLES__DOWN__Throttle17__RAWEDGES.tsv")

    up_table.to_csv(up_out, sep="\t", index=False)
    down_table.to_csv(down_out, sep="\t", index=False)

    print("\n[OK] RAWEDGES SHIFT tables written:")
    print("  ", up_out)
    print("  ", down_out)


if __name__ == "__main__":
    main()

