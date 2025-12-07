import os, sys, io, codecs, argparse

AXIS = ["mph","0","6","12","19","25","31","37","44","50","56","62","69","75","81","87","94","100","%"]

def read_text(path):
    with open(path,"rb") as f: b=f.read()
    if b.startswith(codecs.BOM_UTF8): b=b[len(codecs.BOM_UTF8):]
    return b.decode("utf-8", errors="replace")

def unescape(txt):
    if "\\t" in txt or "\\n" in txt:
        txt = txt.replace("\\r\\n","\n").replace("\\r","\n")
        txt = txt.replace("\\n","\n").replace("\\t","\t")
    return txt

def normalize_header(hdr_raw):
    hdr = [(c or "").replace("\ufeff","").strip() for c in hdr_raw]
    if len(hdr)!=19 or hdr[0].lower() not in ("mph","\ufeffmph") or hdr[-1]!="%":
        return AXIS[:]
    mid=[c.strip() for c in hdr[1:-1]]
    if mid!=AXIS[1:-1]:
        return AXIS[:]
    hdr[0]="mph"; hdr[-1]="%"
    return hdr

def r1(s):
    s = ("" if s is None else str(s).strip())
    if s=="": return ""
    try:
        v=float(s)
        if v in (317.0,318.0): return s
        return f"{v:.1f}"
    except: return s

def fix_labels(s):
    return s.replace("3th Apply","3rd Apply").replace("3th Release","3rd Release")

def process(path, write_back=False, strict=True):
    raw = read_text(path)
    txt = unescape(raw)
    txt = txt.replace("\r\n","\n").replace("\r","\n")
    txt = fix_labels(txt)
    lines = [ln for ln in txt.split("\n") if ln!=""]
    rows  = [ln.split("\t") for ln in lines]
    if not rows: 
        raise SystemExit(f"[guard] ERROR empty TSV: {path}")
    hdr = normalize_header(rows[0])
    # map from original header to positions (best effort)
    name_to_idx = {name:i for i,name in enumerate(rows[0])}
    body=[]
    for rr in rows[1:]:
        if len(rr)<len(rows[0]): rr += [""]*(len(rows[0])-len(rr))
        out=[]
        for col in hdr:
            j = name_to_idx.get(col, None)
            val = rr[j] if (j is not None and j<len(rr)) else ""
            if col in ("mph","%"): out.append(val.strip())
            else: out.append(r1(val))
        body.append(out)
    # sanity: ensure 19 cols
    if len(hdr)!=19: 
        raise SystemExit(f"[guard] ERROR header cols={len(hdr)} expected 19: {hdr}")
    if write_back:
        with open(path,"w",encoding="utf-8",newline="") as f:
            f.write("\t".join(hdr)+"\n")
            for r in body: f.write("\t".join(r)+"\n")
    print(f"[guard] OK {os.path.basename(path)} ({len(hdr)} cols)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--write-back", action="store_true")
    args = ap.parse_args()
    any_file=False
    for root,_,files in os.walk(args.dir):
        for fn in files:
            if fn.lower().endswith(".tsv"):
                any_file=True
                process(os.path.join(root,fn), write_back=args.write_back)
    if not any_file:
        print("[guard] no TSVs found")
if __name__=="__main__":
    main()
