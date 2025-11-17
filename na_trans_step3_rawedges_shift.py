import glob
import os
import statistics

import pandas as pd

# LOG-FIRST RAWEDGES builder for SHIFT tables using shift_events from the cleaner.
# Strict/no-fallback on required columns.

TPS_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]
SHIFT_UP_PAIRS = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
SHIFT_DN_PAIRS = [(2, 1), (3, 2), (4, 3), (5, 4), (6, 5)]
MIN_SPEED_MPH = 3.0  # ignore ultra-low-speed junk


def ensure_canonical_columns(df: pd.DataFrame):
    """
    SHIFT_EVENTS v2 canonicalization.

    Require (strict/no-fallback):
      - time_s
      - from_gear
      - to_gear
      - speed_mph_event
      - throttle_pct_event

    Then normalize to internal names:
      - speed_mph       := speed_mph_event
      - throttle_pct    := throttle_pct_event
    """
    required_v2 = [
        "time_s",
        "from_gear",
        "to_gear",
        "speed_mph_event",
        "throttle_pct_event",
    ]
    missing = [c for c in required_v2 if c not in df.columns]
    if missing:
        return df, missing

    # Treat event-level samples as canonical speed/TPS for RAWEDGES.
    df["speed_mph"] = pd.to_numeric(df["speed_mph_event"], errors="coerce")
    df["throttle_pct"] = pd.to_numeric(df["throttle_pct_event"], errors="coerce")

    return df, []


def tps_to_axis_bin(tps: float):
    if pd.isna(tps):
        return None
    t = max(0.0, min(100.0, float(tps)))
    best = TPS_AXIS[0]
    best_diff = abs(t - best)
    for ax in TPS_AXIS[1:]:
        d = abs(t - ax)
        if d < best_diff:
            best = ax
            best_diff = d
    return best


def build_table(df: pd.DataFrame, pairs):
    df = df.dropna(subset=["speed_mph", "throttle_pct", "from_gear", "to_gear"]).copy()
    df = df[df["speed_mph"] >= MIN_SPEED_MPH]
    df = df[(df["throttle_pct"] >= 0.0) & (df["throttle_pct"] <= 100.0)]

    df["from_gear"] = df["from_gear"].astype(int)
    df["to_gear"] = df["to_gear"].astype(int)
    df = df[df["from_gear"].between(1, 6) & df["to_gear"].between(1, 6)]

    buckets = {(fg, tg, ax): [] for (fg, tg) in pairs for ax in TPS_AXIS}

    for _, row in df.iterrows():
        fg = int(row["from_gear"])
        tg = int(row["to_gear"])
        pair = (fg, tg)
        if pair not in pairs:
            continue
        ax = tps_to_axis_bin(row["throttle_pct"])
        if ax is None:
            continue
        buckets[(fg, tg, ax)].append(float(row["speed_mph"]))

    rows = []
    for (fg, tg) in pairs:
        label = f"{fg} -> {tg} Shift"
        vals = []
        for ax in TPS_AXIS:
            speeds = buckets[(fg, tg, ax)]
            if not speeds:
                vals.append("")
            else:
                med = statistics.median(speeds)
                med = round(med, 1)
                vals.append(f"{med:.1f}")
        rows.append((label, vals))
    return rows


def write_tsv(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = "mph\t" + "\t".join(str(v) for v in TPS_AXIS) + "\t%\n"
    with open(path, "w", newline="") as f:
        f.write(header)
        for label, vals in rows:
            line = label + "\t" + "\t".join(vals) + "\t\n"
            f.write(line)


def main() -> None:
    root_dir = "newlogs"
    shift_dir = os.path.join(root_dir, "output", "01_tables", "shift")
    debug_dir = os.path.join(root_dir, "output", "DEBUG")

    os.makedirs(shift_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    events_glob = os.path.join(
        root_dir, "output", "00_cleaner", "__trans_focus__shift_events__*.csv"
    )
    files = sorted(glob.glob(events_glob))

    print(f"[INFO] shift_events glob: {events_glob}")
    if not files:
        raise SystemExit("[ERROR] No shift_events files found for RAWEDGES build.")

    events_file = max(files, key=os.path.getmtime)
    print(f"[INFO] Using newest shift_events file: {os.path.basename(events_file)}")

    df = pd.read_csv(events_file)

    # Normalize / verify required v2 columns.
    df, missing = ensure_canonical_columns(df)
    if missing:
        print(
            "\n[ERROR] Missing required columns in shift_events file for RAWEDGES "
            "(expected SHIFT_EVENTS v2 schema):"
        )
        for m in missing:
            print(f"   * {m}")
        raise SystemExit("[ERROR] Aborting RAWEDGES build due to missing columns.")

    # Minimal neutral gating (strict/no-fallback).
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["from_gear"] = pd.to_numeric(df["from_gear"], errors="coerce")
    df["to_gear"] = pd.to_numeric(df["to_gear"], errors="coerce")

    df = df.dropna(
        subset=["time_s", "from_gear", "to_gear", "speed_mph", "throttle_pct"]
    ).copy()

    df = df[df["speed_mph"] >= MIN_SPEED_MPH]
    df = df[(df["throttle_pct"] >= 0.0) & (df["throttle_pct"] <= 100.0)]

    df["from_gear"] = df["from_gear"].astype(int)
    df["to_gear"] = df["to_gear"].astype(int)
    df = df[df["from_gear"].between(1, 6) & df["to_gear"].between(1, 6)]

    if df.empty:
        raise SystemExit(
            "[ERROR] No usable shift events after SHIFT_EVENTS v2 gating "
            "(time_s/gear/speed/TPS)."
        )

    # Also emit a debug edges TSV for inspection (post-gating events used).
    debug_edges_path = os.path.join(
        debug_dir, "SHIFT_EDGES__RAW_scan_from_shift_events.tsv"
    )
    df.to_csv(debug_edges_path, sep="\t", index=False)
    print(f"\n[OK] Wrote raw edge list to {debug_edges_path}")

    # Build RAWEDGES tables
    print("\n[STEP] Building RAWEDGES SHIFT UP table...")
    up_rows = build_table(df, SHIFT_UP_PAIRS)
    print("[STEP] Building RAWEDGES SHIFT DOWN table...")
    dn_rows = build_table(df, SHIFT_DN_PAIRS)

    up_out = os.path.join(shift_dir, "SHIFT_TABLES__UP__Throttle17__RAWEDGES.tsv")
    down_out = os.path.join(shift_dir, "SHIFT_TABLES__DOWN__Throttle17__RAWEDGES.tsv")

    write_tsv(up_out, up_rows)
    write_tsv(down_out, dn_rows)

    print("\n[OK] RAWEDGES SHIFT tables written:")
    print("  ", up_out)
    print("  ", down_out)


if __name__ == "__main__":
    main()
