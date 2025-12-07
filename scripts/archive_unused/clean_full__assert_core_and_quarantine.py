#!/usr/bin/env python3
import argparse, sys, csv, shutil
from pathlib import Path
CORE=["speed_mph","throttle_pct","gear_actual","time_s"]
ap=argparse.ArgumentParser(); ap.add_argument("--clean", required=True); a=ap.parse_args()
clean=Path(a.clean); quar=clean/"_quarantine"; quar.mkdir(parents=True, exist_ok=True)
bad=[]
for p in sorted(clean.glob("*.csv")):
  try:
    with p.open("r", encoding="utf-8-sig", newline="") as f:
      headers=next(csv.reader(f))
  except Exception as e:
    print(f"[ERROR] {p.name}: failed to open ({e})"); bad.append(p); continue
  headers=[h.strip() for h in headers]
  miss=[c for c in CORE if c not in headers]
  if miss:
    print(f"[ERROR] {p.name}: missing core columns {miss}")
    print("        present headers:", ", ".join(headers[:40]))
    bad.append(p)
for p in bad:
  dst=quar/p.name
  try: shutil.move(str(p), str(dst)); print(f"[QUAR] Moved bad file to {dst}")
  except Exception as e: print(f"[WARN] Could not quarantine {p.name}: {e}")
sys.exit(2 if bad else 0)
