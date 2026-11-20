"""
Results on Unwaste-India, as numbers and three charts (run python -m unwaste.india_pipeline first).

  python scripts/india_results.py                       -> docs/img/india_fillin.png, india_forecast.png, india_policies.png
  python scripts/india_results.py --item "Paneer 200g" --store "Saket"
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--item", default="Toned milk 500ml")
ap.add_argument("--store", default="Gurugram Sector 45")
a = ap.parse_args()

E = np.load("models/india_eval.npz", allow_pickle=True)
M = json.loads(Path("web/data/india_main.json").read_text())
OUT = Path("docs/img"); OUT.mkdir(parents=True, exist_ok=True)
dates = pd.to_datetime(E["dates"]); te = int(E["te"])
s = int(np.where((E["items"] == a.item) & (E["stores"] == a.store))[0][0])
fests = [(pd.Timestamp(f["date"]), f["name"]) for f in M["chart"]["festivals"]]

# ---------------------------------------------------------------- 1. fill-in
f = M["fill_in"]
dt_ = f.get("demand_truth", {})
print("\n1 · FILL-IN (hidden demand on empty-shelf hours, same model as FreshRetailNet)")
print(f"   {f['stockout_days_pct']}% of store-item-days ran out; shelf empty {f.get('empty_hours_pct', '?')}% of open hours, "
      f"usually from {f.get('typical_start_hour', '?')}:00; sales hide {f['hidden_demand_pct']}% of demand")
if "hour_error_model" in f:
    print(f"   error on empty hours vs true demand: sales as-is {f['hour_error_sales']}% | simple average "
          f"{f['hour_error_average']}% | model {f['hour_error_model']}%")
print(f"   whole stockout days: sales as-is {f['error_sales_as_demand']}%  ->  filled in {f['error_filled']}%")
if dt_:
    print(f"   demand-truth test ({dt_['hidden_blocks']} hidden afternoon blocks): zeros {dt_['wape_naive_zero']:.0%} | "
          f"average {dt_['wape_series_average']:.0%} | model {dt_['wape_model']:.0%}")
HE = np.load("models/india_hourly_eval.npz", allow_pickle=True)
rows = np.where((HE["series"] == s) & (HE["oos"].sum(axis=1) >= 3))[0]
fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 1.4]})
if len(rows):
    r = rows[np.argmax(HE["oos"][rows].sum(axis=1))]
    hrs = np.arange(24)
    ax = axs[0]
    for h_ in hrs[HE["oos"][r].astype(bool)]:
        ax.axvspan(h_ - 0.5, h_ + 0.5, color="#FDE2E2")
    ax.bar(hrs, HE["hd"][r], color="#C9D2E6", label="True demand (unknown to a store)")
    ax.plot(hrs, HE["hs"][r], color="#D64545", marker="o", ms=3, lw=1.5, label="Recorded sales")
    ax.plot(hrs, HE["corrected"][r], color="#2B59F3", ls="--", lw=1.8, label="Filled in by the model")
    ax.set_xlim(5.5, 22.5); ax.set_xticks(range(6, 23, 2))
    ax.set_xlabel("hour of day (shaded = shelf empty)"); ax.set_ylabel("units per hour")
    ax.set_title(f"One stockout day: {HE['dt'][r][:10]}"); ax.legend(fontsize=8)
w = slice(len(dates) - 75, len(dates))
ax = axs[1]
ax.plot(dates[w], E["Y_true"][s, w], color="#14171F", lw=2, label="True demand")
ax.plot(dates[w], E["S"][s, w], color="#D64545", lw=1.5, label="Recorded sales")
ax.plot(dates[w], E["Y"][s, w], color="#2B59F3", lw=1.5, ls="--", label="Filled-in demand")
ax.set_title("Last 75 days, daily totals"); ax.set_ylabel("units per day"); ax.set_xlabel("date"); ax.legend(fontsize=8)
fig.suptitle(f"Fill-in: {a.item}, {a.store}")
fig.tight_layout(); fig.savefig(OUT / "india_fillin.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- 2. forecast
fc = M["forecast"]
print("\n2 · FORECAST, test window (next-day error = WAPE, lower is better)")
print(f"   {'':34s}{'all days':>10s}{'festival days':>15s}")
for k, lbl in [("ours", "Unwaste (weather+festivals+price)"), ("no_calendar", "Same model, no festival calendar"),
               ("avg_7d", "7-day average"), ("same_weekday", "Same weekday last week")]:
    print(f"   {lbl:34s}{fc['all'][k]:>9}%{fc['festival_days'][k]:>14}%")
print(f"   P10-P90 range contains the actual on {fc['coverage_p10_p90']}% of days (aim 80%)")
print("   per festival (Unwaste | no calendar | 7-day avg):")
for x in fc["per_festival"]:
    print(f"     {x['festival']:22s} {x['date']}  {x['ours']:5}% | {x['no_calendar']:5}% | {x['avg_7d']:5}%")
td = dates[te + 1:]
P10, P50, P90 = E["p10"][:, :-1], E["p50"][:, :-1], E["p90"][:, :-1]          # forecast made the night before
fig, axs = plt.subplots(2, 1, figsize=(12, 7.5))
for ax, (y, lo, mid, hi, title) in zip(axs, [
        (E["Y_true"][s, te + 1:], P10[s], P50[s], P90[s], f"{a.item}, {a.store}"),
        (E["Y_true"][:, te + 1:].sum(0), None, P50.sum(0), None, "All stores and items (sum of P50 forecasts)")]):
    if lo is not None:                       # a sum of P10s is not the P10 of the sum, so no band on the total
        ax.fill_between(td, lo, hi, color="#2B59F3", alpha=0.15, label="P10-P90 range")
    ax.plot(td, mid, color="#2B59F3", lw=1.8, label="P50 forecast (made the night before)")
    ax.plot(td, y, color="#14171F", lw=1.4, label="Actual (true demand)")
    for d0, n in fests:
        ax.axvline(d0, color="#D98A00", lw=1, ls=":"); ax.text(d0, ax.get_ylim()[1], n.split(" ")[0], fontsize=7, rotation=90, va="top", ha="right")
    ax.set_title(f"Forecast vs actual, test window: {title}"); ax.set_ylabel("units per day"); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "india_forecast.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- 3. policies
R = M["replay"]
keys = [k for k in ["baseline", "forecast_raw", "forecast_corrected", "unwaste"] if k in R]
names = {"baseline": "Habit\n(recent sales +15%)", "forecast_raw": "Forecast on\nraw sales", "forecast_corrected": "+ fill-in", "unwaste": "Unwaste\n(+ batch risk, actions)"}
print(f"\n3 · POLICIES on the same true demand, {M['dataset']['test_days']} test days")
print(f"   {'':26s}{'waste %':>9s}{'service %':>11s}{'net after lost-sale cost':>26s}{'moved':>8s}")
for k in keys:
    r = R[k]
    print(f"   {names[k].replace(chr(10), ' '):26s}{r['waste_pct_of_received']:>9}{r['service_level_pct']:>11}{r['net_after_lost_sales_rs']:>26,}{r.get('transfer_units', 0):>8}")
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for ax, (col, lbl, fmt) in zip(axs, [("waste_pct_of_received", "Waste % of received", "{:.2f}%"),
                                     ("service_level_pct", "Service level %", "{:.1f}%"),
                                     ("net_after_lost_sales_rs", "Net after lost-sale cost (₹ lakh)", "{:.1f}")]):
    vals = [R[k][col] / (1e5 if col.startswith("net") else 1) for k in keys]
    bars = ax.bar(range(len(keys)), vals, color=["#9AA3B5"] * (len(keys) - 1) + ["#2B59F3"])
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(keys)), [names[k] for k in keys], fontsize=8); ax.set_title(lbl)
    if col == "service_level_pct":
        ax.set_ylim(80, 100)
fig.suptitle("Same true demand, four ways of ordering (Unwaste-India test window)")
fig.tight_layout(); fig.savefig(OUT / "india_policies.png", dpi=150); plt.close(fig)
print(f"\nSaved: {OUT/'india_fillin.png'}, {OUT/'india_forecast.png'}, {OUT/'india_policies.png'}")
