import os,glob,math, argparse, pandas as pd, numpy as np
TPS=[0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]; PAIRS=[(1,2),(2,3),(3,4),(4,5),(5,6)]
def tps_bin(v):
    if pd.isna(v): return None
    for b in TPS:
        if v<=b+3: return b
    return 100
def load_clean(d):
    files=sorted(glob.glob(os.path.join(d,"__trans_focus__clean_FULL__*.csv"))); out=[]
    for f in files:
        try:
            df=pd.read_csv(f, low_memory=False)
            if {"time_s","speed_mph","throttle_pct","gear_actual","brake"}.issubset(df.columns):
                out.append(df.sort_values("time_s"))
        except: pass
    return out
def prep(df):
    d=df.copy()
    if "trans_temp_f" in d.columns: d=d[d["trans_temp_f"]>=100]
    d=d[(d["brake"]==0)]
    d["v"]=pd.to_numeric(d["speed_mph"], errors="coerce")
    d["t"]=pd.to_numeric(d["time_s"], errors="coerce")
    d["thr"]=pd.to_numeric(d["throttle_pct"], errors="coerce")
    d["g"]=pd.to_numeric(d["gear_actual"], errors="coerce")
    d["thr_rate"]=d["thr"].diff()/d["t"].diff()
    d=d[d["thr_rate"].abs().fillna(0)<=12.0]
    d["v_s"]=d["v"].rolling(7, min_periods=3).median()
    d["a"]=(d["v_s"].diff()/d["t"].diff()).clip(-6,6)
    return d
def accel_surface(dfs, gear):
    rec=[]
    for df in dfs:
        d=prep(df); seg=d[(d["g"]==gear) & d["v"].between(0.5,120)].copy()
        if seg.empty: continue
        seg["tbin"]=seg["thr"].apply(tps_bin); seg["vbin"]=np.round(seg["v"],1)
        rec.append(seg.groupby(["tbin","vbin"])["a"].median().reset_index())
    if not rec: return pd.DataFrame(columns=["tbin","vbin","a"])
    base=pd.concat(rec, ignore_index=True).groupby(["tbin","vbin"])["a"].median().reset_index()
    filled=[]
    for t in TPS:
        s=base[base["tbin"]==t].sort_values("vbin")
        if s.empty: continue
        v=s["vbin"].values; a=s["a"].values
        v_all=np.arange(max(0.5,v.min()), min(120.0,v.max())+0.1, 0.1)
        a_series=pd.Series(a, index=v).reindex(v_all)
        a_interp=a_series.interpolate("linear", limit_direction="both").fillna(method="ffill").fillna(method="bfill")
        filled.append(pd.DataFrame({"tbin":t,"vbin":v_all,"a":a_interp.values}))
    return pd.concat(filled, ignore_index=True) if filled else base
def pick_up(A1,A2):
    out={}
    for t in TPS:
        g=A1[A1["tbin"]==t]; n=A2[A2["tbin"]==t]
        if g.empty or n.empty: out[t]=math.nan; continue
        m=pd.merge(g,n,on="vbin",suffixes=("_g","_n"))
        if m.empty: out[t]=math.nan; continue
        cand=m[(m["a_n"]>=0.03) & (m["a_n"]>=m["a_g"]*1.02)]
        mph=float(cand.iloc[0]["vbin"]) if not cand.empty else float(m["vbin"].max())
        out[t]=round(mph,1)
    return out
def write_up(out_dir, up_rows):
    rows=[["mph"]+TPS+["%"]]
    for (frm,to), row in up_rows.items():
        rows.append([f"{frm} -> {to} Shift"]+[ (row.get(t,"") if not pd.isna(row.get(t,math.nan)) else "") for t in TPS ]+[""])
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir,"SHIFT_TABLES__UP__Throttle17.tsv"), sep="\t", header=False, index=False)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--clean", required=True); ap.add_argument("--out", required=True)
    a=ap.parse_args(); dfs=load_clean(a.clean)
    if not dfs: raise SystemExit("No CLEAN_FULL logs found in "+a.clean)
    up={}
    for (g1,g2) in [(1,2),(2,3),(3,4),(4,5),(5,6)]:
        A1=accel_surface(dfs,g1); A2=accel_surface(dfs,g2)
        up[(g1,g2)]=pick_up(A1,A2)
    write_up(a.out, up); print("[OK] Wrote UP in", a.out)
if __name__=="__main__": main()
