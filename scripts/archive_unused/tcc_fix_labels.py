import os, io, pandas as pd
def fix_labels(path):
    raw=open(path,"rb").read().replace(b"\\t",b"\t").replace(b"\\n",b"\n")
    df=pd.read_csv(io.StringIO(raw.decode("utf-8","replace")), sep="\t", dtype=str)
    c0=df.columns[0]
    df[c0]=df[c0].replace({"3th Apply":"3rd Apply","4th Apply":"4th Apply","5th Apply":"5th Apply","6th Apply":"6th Apply",
                           "3th Release":"3rd Release","4th Release":"4th Release","5th Release":"5th Release","6th Release":"6th Release"})
    df.to_csv(path, sep="\t", index=False)
    print("[TCC] labels fixed:", os.path.basename(path))
