"""Show the Indian twin dataset: python scripts/show_twin.py  (run python -m unwaste.india first)"""
import pandas as pd

df = pd.read_csv("data/raw/india_twin/twin_daily.csv", parse_dates=["date"])
print(f"{len(df):,} rows = {df.store_id.nunique()} stores x {df.product_id.nunique()} items x {df.date.nunique()} days\n")
print("Sample rows:")
print(df.head(5).to_string(index=False), "\n")
day = df.groupby("date").agg(demand_original=("demand_original", "sum"), demand_india=("demand_india", "sum"),
                             temp=("temp", "first"), rain=("rain", "first"), festival=("fest_name", "first"),
                             days_to_fest=("days_to_fest", "first"))
day["change_%"] = (100 * (day.demand_india / day.demand_original - 1)).round(1)
print("Around each festival (all stores, total units per day):")
print(day[day.days_to_fest <= 3].round(1).to_string(), "\n")
print("Hottest days:")
print(day.sort_values("temp", ascending=False).head(5).round(1).to_string())
