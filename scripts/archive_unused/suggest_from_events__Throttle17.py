import pandas as pd, numpy as np, re, pathlib as p
TPSH=["mph","0","6","12","19","25","31","37","44","50","56","62","69","75","81","87","94","100","%"]; TPS=TPSH[1:-1]
ev = pd.read_csv(r".\newlogs\output\00_validation\SHIFT_EVENTS.csv")
ev = ev.rename(columns={c:c.strip().lower() for c in ev.columns}).dropna(subset=["from","to","tps","mph"])
# 17-point TPS bins
bins=[-0.1,0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100,200]
lab =["0","0","6","12","19","25","31","37","44","50","56","62","69","75","81","87","94","100"]
ev["pair"] = ev["from"].astype(int).astype(str)+" -> "+ev["to"].astype(int).astype(str)+" Shift"
ev["tps_bin"] = pd.cut(pd.to_numeric(ev["tps"],errors="coerce"), bins=bins, labels=lab, include_lowest=True)
g = (ev.dropna(subset=["tps_bin"])
       .groupby(["pair","tps_bin"])["mph"].median()
       .reset_index())
rows = sorted(g["pair"].unique(), key=lambda s:(int(re.match(r"\s*(\d)",s).group(1)), int(re.search(r"->\s*(\d)",s).group(1))))
mat = pd.DataFrame(index=rows, columns=TPS, dtype=float)
for _,r in g.iterrows(): mat.loc[r["pair"], r["tps_bin"]] = r["mph"]
# nearest fill across TPS to avoid blanks
for idx in mat.index:
    s = pd.Series(mat.loc[idx], index=TPS).astype(float)
    s = s.interpolate("nearest", limit_direction="both")
    mat.loc[idx] = s.values
# write suggest as Throttle17 TSV
o = mat.copy(); o["%"]=""; o=o.reset_index().rename(columns={"index":"mph"})
with open(r".\newlogs\output\00_validation\SHIFT_VS_LOGS__SUGGEST.tsv","w",newline="") as f:
    f.write("\t".join(TPSH)+"\n")
    for _,r in o.iterrows():
        f.write("\t".join([str(r["mph"])] + [f"{float(r[c]):.1f}" for c in TPS] + [""]) + "\n")
print("[SUGGEST] wrote SHIFT_VS_LOGS__SUGGEST.tsv")
