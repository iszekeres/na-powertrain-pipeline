import os, glob, sys
import argparse
import pandas as pd


def ensure_canonical_columns(df):
    """
    Accept either:
      time_s, from, to, mph, tps
    or:
      time_s, from_gear, to_gear, speed_mph, throttle_pct

    and normalize to: from_gear, to_gear, speed_mph, throttle_pct.
    """
    alias_map = {
        "from_gear": ["from_gear", "from"],
        "to_gear": ["to_gear", "to"],
        "speed_mph": ["speed_mph", "mph"],
        "throttle_pct": ["throttle_pct", "tps"],
    }

    for canon, candidates in alias_map.items():
        if canon in df.columns:
            # Already present, nothing to do
            continue
        # Try aliases in order
        for cand in candidates:
            if cand in df.columns:
                df[canon] = df[cand]
                break

    required = ["from_gear", "to_gear", "speed_mph", "throttle_pct"]
    missing = [c for c in required if c not in df.columns]
    return df, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r"C:\tuning\na-trans-data\newlogs")
    args = ap.parse_args()

    root = os.path.abspath(args.data_root)
    events_glob = os.path.join(root, "output", "00_cleaner", "__trans_focus__shift_events__*.csv")
    files = glob.glob(events_glob)
    if not files:
        print(f"[ERROR] No shift_events files found under: {events_glob}")
        sys.exit(1)

    # Pick newest shift_events file
    events_file = max(files, key=os.path.getmtime)
    print(f"[USING] {events_file}")

    df = pd.read_csv(events_file)

    print(f"[ROWS] {len(df)}")
    print("[COLUMNS]")
    for c in df.columns:
        print("  -", c)

    df, missing = ensure_canonical_columns(df)
    if missing:
        print("\n[ERROR] Missing required columns in shift_events file (after alias mapping):")
        for m in missing:
            print("   *", m)
        sys.exit(2)

    grp = df.groupby(["from_gear", "to_gear"])
    print("\n[SHIFT COUNTS + SPEED/TPS RANGES]")
    for (fg, tg), g in grp:
        n = len(g)
        spd_min = g["speed_mph"].min()
        spd_med = g["speed_mph"].median()
        spd_max = g["speed_mph"].max()
        tps_min = g["throttle_pct"].min()
        tps_med = g["throttle_pct"].median()
        tps_max = g["throttle_pct"].max()
        print(
            f"{fg} -> {tg}: n={n:3d} | "
            f"speed {spd_min:5.1f} / {spd_med:5.1f} / {spd_max:5.1f} mph | "
            f"TPS {tps_min:5.1f} / {tps_med:5.1f} / {tps_max:5.1f}"
        )


if __name__ == "__main__":
    main()
