#!/usr/bin/env python3
import os, sys, zipfile, numpy as np, pandas as pd

TPS=[0,6,12,19,25,31,37,44,50,56,62,69,75,81,87,94,100]
COLS=["mph"]+list(map(str,TPS))+["%"]

def fix_labels(s): 
    return s.replace("3th Apply","3rd Apply").replace("3th Release","3rd Release") if isinstance(s,str) else s

def read_tsv(path):
    df=pd.read_csv(path,sep="\t",dtype=str,engine="python")
    if df.columns.tolist()!=COLS:
        # normalize to canonical header order
        cols=list(df.columns)
        if cols[0]!="mph": cols=[ "mph"]+cols
        while len(cols)<len(COLS): cols.append("%")
        df.columns=cols[:len(df.columns)]
        for c in COLS:
            if c not in df.columns: df[c]=""
        df=df[COLS]
    df["mph"]=df["mph"].apply(fix_labels)
    return df

def num(x):
    try: v=float(x); 
    except: return np.nan
    return np.nan if v>=317 else v

def round1(x):
    if x=="" or x is None: return ""
    try: v=float(x)
    except: return ""
    if v>=317: return str(int(round(v)))
    return f"{round(v,1):.1f}"

def monotone_nondec(vals):
    out=[]; m=-1e18
    for v in vals:
        if np.isfinite(v):
            m=max(m,v)
            out.append(m)
        else:
            out.append(np.nan)
    return out

def audit_shift(up,down):
    # 0.1 dp, keep sentinels
    for df in (up,down):
        for c in map(str,TPS):
            df[c]=df[c].apply(round1)
    # monotone across TPS
    def fix(df):
        for i in range(len(df)):
            arr=[num(df.loc[i,str(t)]) for t in TPS]
            arr=monotone_nondec(arr)
            for j,t in enumerate(TPS):
                v=arr[j]
                if np.isfinite(v): df.loc[i,str(t)]=f"{round(v,1):.1f}"
    fix(up); fix(down)
    # DOWN ≤ UP − 1.0
    pairs=[("1 -> 2 Shift","2 -> 1 Shift"),
           ("2 -> 3 Shift","3 -> 2 Shift"),
           ("3 -> 4 Shift","4 -> 3 Shift"),
           ("4 -> 5 Shift","5 -> 4 Shift"),
           ("5 -> 6 Shift","6 -> 5 Shift")]
    uidx={up.loc[i,"mph"]:i for i in range(len(up))}
    didx={down.loc[i,"mph"]:i for i in range(len(down))}
    for upn,dnn in pairs:
        if upn not in uidx or dnn not in didx: continue
        iu, idn = uidx[upn], didx[dnn]
        for t in TPS:
            cu=str(t); cd=cu
            u=num(up.loc[iu,cu]); d=num(down.loc[idn,cd])
            if np.isfinite(u):
                if not np.isfinite(d) or d>u-1.0:
                    down.loc[idn,cd]=f"{round(u-1.0,1):.1f}"
    return up,down

def audit_tcc(ap,rl):
    for df in (ap,rl):
        for c in map(str,TPS):
            df[c]=df[c].apply(round1)
    # RELEASE ≥ APPLY + 1.1 (ignore sentinels)
    for i in range(len(ap)):
        for t in TPS:
            c=str(t)
            a=num(ap.loc[i,c]); r=num(rl.loc[i,c])
            if np.isfinite(a):
                if not np.isfinite(r) or r<a+1.1:
                    rl.loc[i,c]=f"{round(a+1.1,1):.1f}"
    return ap,rl

def write_tsv(df,path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        f.write("mph\t"+"\t".join(map(str,TPS))+"\t%\n")
        for _,r in df.iterrows():
            f.write("\t".join([str(r["mph"])]+[str(r[str(t)]) for t in TPS]+[""])+"\n")

def main(shift_dir,tcc_dir,zip_out):
    up_p=os.path.join(shift_dir,"SHIFT_TABLES__UP__Throttle17.tsv")
    dn_p=os.path.join(shift_dir,"SHIFT_TABLES__DOWN__Throttle17.tsv")
    ap_p=os.path.join(tcc_dir,"TCC_APPLY__Throttle17.tsv")
    rl_p=os.path.join(tcc_dir,"TCC_RELEASE__Throttle17.tsv")
    for p in (up_p,dn_p,ap_p,rl_p):
        if not os.path.exists(p):
            print("[MISS]",p); sys.exit(2)
    up,down = read_tsv(up_p), read_tsv(dn_p)
    ap,rl   = read_tsv(ap_p), read_tsv(rl_p)
    up,down = audit_shift(up,down)
    ap,rl   = audit_tcc(ap,rl)
    write_tsv(up,up_p); write_tsv(down,dn_p); write_tsv(ap,ap_p); write_tsv(rl,rl_p)

    # locate slip dir next to 01_tables
    # shift_dir looks like ...\01_tables\shift\AM_CF_BASE  → base = ...\01_tables
    base_01 = os.path.dirname(os.path.dirname(os.path.abspath(shift_dir)))
    slip_dir = os.path.join(base_01,"slip")
    slip_files=[]
    if os.path.isdir(slip_dir):
        for name in os.listdir(slip_dir):
            if name.lower().endswith(".tsv"):
                slip_files.append(os.path.join(slip_dir,name))
    if not slip_files:
        print("[WARN] No slip TSVs found under", slip_dir)

    os.makedirs(os.path.dirname(zip_out),exist_ok=True)
    with zipfile.ZipFile(zip_out,"w",compression=zipfile.ZIP_DEFLATED) as z:
        z.write(up_p, arcname=os.path.join("01_tables","shift",os.path.basename(up_p)))
        z.write(dn_p, arcname=os.path.join("01_tables","shift",os.path.basename(dn_p)))
        z.write(ap_p, arcname=os.path.join("01_tables","tcc",  os.path.basename(ap_p)))
        z.write(rl_p, arcname=os.path.join("01_tables","tcc",  os.path.basename(rl_p)))
        for sf in slip_files:
            z.write(sf, arcname=os.path.join("01_tables","slip",os.path.basename(sf)))
    print("[OK] Seed pack zipped:", zip_out)
    return 0

if __name__=="__main__":
    if len(sys.argv)!=4:
        print("usage: audit_and_pack_neutral_seed.py <shift_dir> <tcc_dir> <zip_out>"); sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
