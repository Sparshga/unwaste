"""
Demo 1 · Data pipeline: show what one raw row looks like and what the cleaned panel looks like.
Run: python scripts/show_data.py            (add --synthetic to use stand-in data)
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from unwaste.config import default_config          # noqa: E402
from unwaste.data import COL, build_panel, load_frn, make_synthetic  # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
syn = "--synthetic" in sys.argv
cfg = default_config(use_synthetic=syn)

print("\n=== 1. RAW ROW (FreshRetailNet-50K: one store, one item, one day) ===")
raw = make_synthetic(cfg) if syn else load_frn(cfg)
r = raw.iloc[0]
for c in [COL["city"], COL["store"], COL["product"], COL["date"], COL["sale"], COL["discount"], COL["holiday"]]:
    print(f"  {c:22s} {r[c]}")
print("  hours_sale (24 numbers)   ", np.round(np.asarray(r[COL["hours_sale"]], float), 2).tolist())
print("  hours_stock_status (24)   ", np.asarray(r[COL["hours_stock"]], int).tolist(), "  1 = shelf empty")

print("\n=== 2. CLEAN PANEL (after step 1) ===")
p = build_panel(cfg)
print(f"  {p.meta.shape[0]} series (store x item), {p.daily['dt'].nunique()} days, {len(p.daily):,} rows")
print("\n  daily  (one row per store-item-day; sales rescaled to units):")
print(p.daily[["series_id", "store_id", "product_id", "dt", "sales", "stockout_hours", "discount", "holiday"]]
      .head(5).round({"sales": 2}).to_string(index=False))
print("\n  hourly arrays, same row:  sales[24] and stockout[24]")
print("   sales   ", np.round(p.hourly_sales[0], 1).tolist())
print("   stockout", p.hourly_oos[0].astype(int).tolist(), "  (nights 22:00-06:00 forced to 0 = closed, not stockout)")
print("\n  meta  (one row per series; price & shelf life are stated assumptions):")
print(p.meta[["series_id", "store_id", "product_id", "price", "shelf_life", "min_remaining", "sellable_days"]]
      .head(5).to_string(index=False))
print("\n  sellable_days = shelf_life - ceil(30% x shelf_life)   <- FSSAI rule")
