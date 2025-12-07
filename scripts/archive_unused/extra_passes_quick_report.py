
#!/usr/bin/env python3
# extra_passes_quick_report.py
# Scan a review dir for outputs from each weighted pass prefix and summarize what exists.

import os, glob, argparse, pandas as pd

PREFIXES = {
  "INTENT":   ".\\INTENT",
  "CORNER":   ".\\CORNER",
  "OCC":      ".\\OCC",
  "RPMFLOOR": ".\\RPMFLOOR",
  "CONSIST":  ".\\CONSIST",
  "LAT":      ".\\LAT",
  "STOPGO":   ".\\STOPGO",
  "EBRAKE":   ".\\EBRAKE",
  "TRAC":     ".\\TRAC",
  "DFCO":     ".\\DFCO",
}

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--review-dir", default=r".\06_Logs\Trans_Review")
  args = ap.parse_args()
  rd = args.review_dir

  print("Review dir:", rd)
  os.makedirs(rd, exist_ok=True)

  for name, pref in PREFIXES.items():
    pattern = os.path.join(rd, os.path.basename(pref) + "*.*")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
      print(f" - {name:<8}: no outputs found matching {os.path.basename(pref)}*")
      continue
    latest = files[0]
    size = os.path.getsize(latest)
    print(f" - {name:<8}: found {len(files):2d} files; latest: {os.path.basename(latest)} ({size} bytes)")
    # Try to peek inside if it's a text table
    try:
      if latest.endswith(".tsv"):
        df = pd.read_csv(latest, sep="\t", nrows=5)
        print(f"           preview (tsv head): {list(df.columns)[:4]} rows={len(df)}")
      elif latest.endswith(".csv"):
        df = pd.read_csv(latest, nrows=5)
        print(f"           preview (csv head): {list(df.columns)[:4]} rows={len(df)}")
    except Exception as e:
      print(f"           (preview error: {e})")

if __name__ == "__main__":
  main()
