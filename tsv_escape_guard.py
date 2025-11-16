import os,glob,pandas as pd, numpy as np, argparse
TPS = [0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a=ap.parse_args()
for f in glob.glob(os.path.join(a.dir,"*Throttle17*.tsv")):
    try:
        df = pd.read_csv(f, sep="\t")
        if df.shape[1]==1:
            df = pd.read_csv(f, sep=",")
        # normalize header if mph present
        cols = df.columns.tolist()
        if cols and cols[0].strip().lower()=="mph":
            rest = ["mph"] + list(map(str,TPS))
            if cols[-1].strip()=="%": rest = rest + ["%"]
            if len(rest)==df.shape[1]: df.columns = rest
        # 1dp except sentinels 317/318
        def snap(v):
            try:
                if pd.isna(v): return v
                if float(v) in (317.0,318.0): return float(v)
                return round(float(v),1)
            except: return v
        for c in df.columns[1:]:
            df[c] = df[c].map(snap)
        df.to_csv(f, sep="\t", index=False)
        print("[TSVGUARD] fixed", os.path.basename(f))
    except Exception as e:
        print("[TSVGUARD] skip", os.path.basename(f), e)
