import os
import glob

import numpy as np
import pandas as pd


def pick_column(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def canonize_series(raw: pd.Series) -> pd.Series:
    s = raw.astype(str).str.strip().str.upper()
    out = []
    for v in s:
        if v == "" or v == "NAN":
            out.append(np.nan)
            continue
        if any(x in v for x in ["P", "R", "N"]):
            out.append(0)
            continue
        try:
            fv = float(v)
        except ValueError:
            out.append(np.nan)
            continue
        if 1 <= fv <= 6:
            out.append(int(round(fv)))
        else:
            out.append(0 if fv == 0 else np.nan)
    ser = pd.Series(out, index=raw.index, dtype="float")
    ser = ser.ffill().fillna(0).astype(int)
    return ser


def main():
    clean_dir = os.path.join("newlogs", "cleaned")
    pattern = os.path.join(clean_dir, "__trans_focus__clean_FULL__*.csv")
    files = sorted(glob.glob(pattern))
    print(f"[INFO] Found {len(files)} CLEAN_FULL file(s) for gear canon in {clean_dir}")
    if not files:
        raise SystemExit("[ERROR] No CLEAN_FULL files for gear canon.")

    actual_cands = ["gear_actual", "Trans Current Gear", "Trans Actual Gear"]
    cmd_cands = ["gear_cmd", "Trans Commanded Gear", "Trans Cmd Gear"]

    for path in files:
        base = os.path.basename(path)
        print(f"\n[FILE] {base}")
        try:
            df = pd.read_csv(path)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] Failed read: {e}")
            continue

        actual_col = pick_column(df, actual_cands)
        cmd_col = pick_column(df, cmd_cands)
        missing = []
        if actual_col is None:
            missing.append(f"gear_actual (any of: {actual_cands})")
        if cmd_col is None:
            missing.append(f"gear_cmd (any of: {cmd_cands})")
        if missing:
            print("  [ERROR] Missing raw gear columns: " + "; ".join(missing))
            continue

        print(f"  [INFO] Using '{actual_col}' as actual, '{cmd_col}' as commanded")
        df["gear_actual__canon"] = canonize_series(df[actual_col])
        df["gear_cmd__canon"] = canonize_series(df[cmd_col])

        backup = path + ".gearcanon.bak"
        if not os.path.exists(backup):
            try:
                os.replace(path, backup)
                print(f"  [INFO] Backup -> {backup}")
            except PermissionError as e:
                print(f"  [WARN] Backup skipped (file locked): {e}")
        else:
            print(f"  [WARN] Backup exists: {backup} (overwriting main only)")
        df.to_csv(path, index=False)
        print("  [OK] Wrote canonicalized file with gear_actual__canon / gear_cmd__canon")

    print("\n[DONE] Gear canonicalization complete.")


if __name__ == "__main__":
    main()
