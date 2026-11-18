"""Print the columns of the downloaded dataset so we can confirm the loader's column names.
Run: python scripts/inspect_data.py"""
import glob

import pandas as pd

files = sorted(glob.glob("data/raw/frn50k/**/*.parquet", recursive=True))
print("files:", files)
df = pd.read_parquet(files[0])
print(df.dtypes)
print(df.head(2).T)
print("rows:", len(df), "| cities:", df["city_id"].nunique() if "city_id" in df else "?",
      "| stores:", df["store_id"].nunique() if "store_id" in df else "?")
