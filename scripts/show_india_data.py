"""
Print one sample data point of Unwaste-India: every column of one row, then its 24 hours.

  python scripts/show_india_data.py                                   # a stockout day
  python scripts/show_india_data.py --item "Paneer 200g" --store "Saket" --date 2026-11-07
"""
import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--item", default="Toned milk 500ml")
ap.add_argument("--store", default="Gurugram Sector 45")
ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: a recent day when the shelf ran out)")
a = ap.parse_args()

df = pd.read_csv("data/raw/india_dataset/unwaste_india.csv", parse_dates=["date"], dtype={"hours_stock_status": str})
h = np.load("data/raw/india_dataset/unwaste_india_hourly.npz")

print(f"Unwaste-India: {len(df):,} rows = {df.store_name.nunique()} stores x {df.item.nunique()} items x "
      f"{df.date.nunique()} days ({df.date.min().date()} to {df.date.max().date()})")
print(f"Columns ({len(df.columns)}): {', '.join(df.columns)}")
print("Hourly arrays (same row order): " + ", ".join(f"{k} {h[k].shape}" for k in h.files))

pick = df[(df.item == a.item) & (df.store_name == a.store)]
if a.date:
    pick = pick[pick.date == pd.Timestamp(a.date)]
else:
    pick = pick[pick.stockout_hours >= 4]
if pick.empty:
    raise SystemExit("No matching row: check --item, --store and --date")
r = pick.index[-1]
row = df.loc[r]

print(f"\n--- One data point: row {r} ---")
for c in df.columns:
    v = row[c]
    if isinstance(v, float) and np.isnan(v):
        v = "(none)"
    elif isinstance(v, pd.Timestamp):
        v = v.date()
    print(f"  {c:22s} {v}")

print("\n--- The same day, hour by hour (store open 6:00-22:59) ---")
print("  hour   sold   shelf    true demand (hidden from a real store)")
for i in range(6, 23):
    shelf = "EMPTY" if h["hours_stock_status"][r, i] else "ok"
    print(f"  {i:02d}:00 {h['hours_sale'][r, i]:6d}   {shelf:6s} {h['hours_demand_true'][r, i]:8d}")
print(f"  total  {h['hours_sale'][r].sum():5d}            {h['hours_demand_true'][r].sum():8d}"
      f"   -> sales miss {h['hours_demand_true'][r].sum() - h['hours_sale'][r].sum()} units")
