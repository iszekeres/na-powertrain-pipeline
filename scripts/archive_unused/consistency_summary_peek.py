import sys, pandas as pd
p = sys.argv[1]
s = pd.read_csv(p)
print("rows:", len(s))
print("\nTop 10 pairs by count:")
print(s.groupby("pair")["n"].sum().sort_values(ascending=False).head(10).to_string())
print("\nHead:")
print(s.head(12).to_string(index=False))
