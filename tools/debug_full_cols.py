import pandas as pd, glob
files = glob.glob(r'newlogs\\cleaned\\__trans_focus__clean_FULL__cold-hot1__*.csv')
files.sort()
print('USING', files[-1])
df = pd.read_csv(files[-1], nrows=0)
cols = list(df.columns)
print('TAIL COLUMNS:')
print(cols[-40:])
