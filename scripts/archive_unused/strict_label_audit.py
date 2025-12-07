#!/usr/bin/env python3
# strict_label_audit.py — fail on legacy labels (e.g., "3th") and bad header width.
import os, sys, argparse

def audit_dir(d):
    bad=[]
    for fn in os.listdir(d):
        if not fn.endswith("Throttle17.tsv"): continue
        p=os.path.join(d,fn)
        txt=open(p,"r",encoding="utf-8").read()
        if "3th Apply" in txt or "3th Release" in txt: bad.append(p)
        # header width check: 19 columns (mph + 17 TPS + %)
        first=txt.splitlines()[0]
        if first.count("\t") != 18: bad.append(p+" [bad header cols]")
    if bad:
        raise SystemExit("AUDIT FAIL:\n" + "\n".join(bad))
    print(f"[AUDIT OK] {d}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    a=ap.parse_args()
    for d in a.dirs: audit_dir(d)
