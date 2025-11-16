import glob
import os

import numpy as np
import pandas as pd


TPS_AXIS = [0, 6, 12, 19, 25, 31, 37, 44, 50, 56, 62, 69, 75, 81, 87, 94, 100]

LOCK_ON_RPM = 50.0
UNLOCK_OFF_RPM = 120.0
MIN_GEAR = 3
MIN_SPEED = 25.0
MAX_SPEED = 120.0

ENGINE_CAND = [
    "Engine RPM",
    "Engine RPM (SAE)",
    "Engine Speed",
    "RPM",
    "Engine_RPM",
    "Engine_Speed",
]
TURB_CAND = [
    "Trans Turbine RPM",
    "Trans Turbine Speed",
    "Trans Input Shaft RPM",
    "Trans Input Shaft Speed",
    "Turbine Speed",
    "ISS",
    "Trans_Input_Shaft_RPM",
    "Trans_Input_Shaft_Speed",
]

REQ_BASE = ["time_s", "speed_mph__canon", "throttle_pct__canon", "gear_actual__canon", "brake"]


def pick_alias(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def tps_to_bin(t):
    try:
        t = float(t)
    except (TypeError, ValueError):
        return np.nan
    t = max(0.0, min(100.0, t))
    diffs = [abs(t - a) for a in TPS_AXIS]
    return TPS_AXIS[int(np.argmin(diffs))]


def build_tcc_table(events_df: pd.DataFrame | None, label_suffix: str, out_name: str, tcc_dir: str):
    if events_df is None or events_df.empty:
        print(f"[WARN] No {label_suffix} events; writing empty table {out_name}")
        header = ["mph"] + [str(a) for a in TPS_AXIS] + ["%"]
        lines = ["\t".join(header)]
        for g, label in zip([3, 4, 5, 6], ["3rd", "4th", "5th", "6th"]):
            row = [f"{label} {label_suffix}"] + [""] * len(TPS_AXIS) + [""]
            lines.append("\t".join(row))
        out_path = os.path.join(tcc_dir, out_name)
        with open(out_path, "w", newline="") as f:
            f.write("\n".join(lines))
        print(f"[OK] Wrote empty {out_name}")
        return

    df = events_df.copy()
    df["tps_bin"] = df["throttle_pct"].apply(tps_to_bin)
    df = df[df["tps_bin"].notna()]

    grp = df.groupby(["gear", "tps_bin"])["speed_mph"].median()
    print(f"[INFO] {label_suffix} groups: {len(grp)}")

    header = ["mph"] + [str(a) for a in TPS_AXIS] + ["%"]
    lines = ["\t".join(header)]

    for g, label in zip([3, 4, 5, 6], ["3rd", "4th", "5th", "6th"]):
        row = [f"{label} {label_suffix}"]
        for tps in TPS_AXIS:
            v = grp.get((g, tps), np.nan)
            if pd.isna(v):
                row.append("")
            else:
                row.append(f"{round(float(v), 1):.1f}")
        row.append("")
        lines.append("\t".join(row))

    out_path = os.path.join(tcc_dir, out_name)
    with open(out_path, "w", newline="") as f:
        f.write("\n".join(lines))
    print(f"[OK] Wrote {out_name}")


def main():
    clean_dir = os.path.join("newlogs", "cleaned")
    tcc_dir = os.path.join("newlogs", "output", "01_tables", "tcc")
    debug_dir = os.path.join("newlogs", "output", "DEBUG")

    os.makedirs(tcc_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    pattern = os.path.join(clean_dir, "__trans_focus__clean_FULL__*.csv")
    files = sorted(glob.glob(pattern))
    print(f"[INFO] Found {len(files)} CLEAN_FULL file(s) for TCC in {clean_dir}")
    if not files:
        raise SystemExit("[ERROR] No CLEAN_FULL files for TCC build.")

    apply_events: list[dict] = []
    release_events: list[dict] = []

    for path in files:
        base = os.path.basename(path)
        print(f"\n[FILE] {base}")
        try:
            df = pd.read_csv(path)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] Failed read: {e}")
            continue

        missing = [c for c in REQ_BASE if c not in df.columns]
        if missing:
            print(f"  [ERROR] Missing required columns: {missing}")
            continue

        # Use fused slip if explicitly present, else engine - turbine
        if "tcc_slip_fused" in df.columns:
            print("  [INFO] Using 'tcc_slip_fused' for slip")
            slip = pd.to_numeric(df["tcc_slip_fused"], errors="coerce")
        else:
            eng_col = pick_alias(df, ENGINE_CAND)
            tur_col = pick_alias(df, TURB_CAND)
            miss2 = []
            if eng_col is None:
                miss2.append(f"Engine RPM (any of: {ENGINE_CAND})")
            if tur_col is None:
                miss2.append(f"Turbine/Input RPM (any of: {TURB_CAND})")
            if miss2:
                print("  [ERROR] Missing for slip: " + "; ".join(miss2))
                continue
            print(f"  [INFO] Using engine '{eng_col}' and turbine '{tur_col}' for slip")
            eng = pd.to_numeric(df[eng_col], errors="coerce")
            tur = pd.to_numeric(df[tur_col], errors="coerce")
            slip = eng - tur

        time_s = pd.to_numeric(df["time_s"], errors="coerce")
        speed = pd.to_numeric(df["speed_mph__canon"], errors="coerce")
        tps = pd.to_numeric(df["throttle_pct__canon"], errors="coerce")
        gear = pd.to_numeric(df["gear_actual__canon"], errors="coerce").fillna(0).astype(int)
        brake = pd.to_numeric(df["brake"], errors="coerce").fillna(0.0)

        base_mask = (
            time_s.notna()
            & speed.notna()
            & tps.notna()
            & slip.notna()
            & (gear >= MIN_GEAR)
            & (speed >= MIN_SPEED)
            & (speed <= MAX_SPEED)
        )

        if not base_mask.any():
            print("  [WARN] No samples pass base TCC mask.")
            continue

        slip_abs = slip.abs()
        # OFF = brake <= 0.5, ON = > 0.5 (binary 0/1 from CLEAN_FULL)
        locked = (slip_abs <= LOCK_ON_RPM) & base_mask & (brake <= 0.5)
        unlocked = (slip_abs >= UNLOCK_OFF_RPM) & base_mask

        locked_prev = locked.shift(1, fill_value=False)
        unlocked_prev = unlocked.shift(1, fill_value=False)

        apply_idx = locked & (~locked_prev)
        release_idx = unlocked & (~unlocked_prev)

        idx_apply = list(df.index[apply_idx])
        idx_release = list(df.index[release_idx])

        print(f"  [INFO] APPLY events: {len(idx_apply)}")
        print(f"  [INFO] RELEASE events: {len(idx_release)}")

        for i in idx_apply:
            apply_events.append(
                {
                    "file": base,
                    "time_s": float(time_s.iloc[i]),
                    "gear": int(gear.iloc[i]),
                    "speed_mph": float(speed.iloc[i]),
                    "throttle_pct": float(tps.iloc[i]),
                }
            )

        for i in idx_release:
            release_events.append(
                {
                    "file": base,
                    "time_s": float(time_s.iloc[i]),
                    "gear": int(gear.iloc[i]),
                    "speed_mph": float(speed.iloc[i]),
                    "throttle_pct": float(tps.iloc[i]),
                }
            )

    if apply_events:
        apply_df = (
            pd.DataFrame(apply_events)
            .sort_values(["file", "gear", "time_s"])
            .reset_index(drop=True)
        )
        apply_ev_path = os.path.join(debug_dir, "TCC_APPLY_EVENTS__50_120.tsv")
        apply_df.to_csv(apply_ev_path, sep="\t", index=False)
        print(f"\n[OK] Wrote APPLY events -> {apply_ev_path}")
    else:
        apply_df = None
        print("\n[WARN] No APPLY events detected.")

    if release_events:
        release_df = (
            pd.DataFrame(release_events)
            .sort_values(["file", "gear", "time_s"])
            .reset_index(drop=True)
        )
        release_ev_path = os.path.join(debug_dir, "TCC_RELEASE_EVENTS__50_120.tsv")
        release_df.to_csv(release_ev_path, sep="\t", index=False)
        print(f"[OK] Wrote RELEASE events -> {release_ev_path}")
    else:
        release_df = None
        print("[WARN] No RELEASE events detected.")

    build_tcc_table(apply_df, "Apply", "TCC_APPLY__Throttle17.tsv", tcc_dir)
    build_tcc_table(release_df, "Release", "TCC_RELEASE__Throttle17.tsv", tcc_dir)

    print("\n[DONE] TCC APPLY/RELEASE tables built from logs (50/120 rpm thresholds).")


if __name__ == "__main__":
    main()
