"""
Demo 2 · Impute pipeline (stockout fill-in): one PNG with the three findings.
Run: python scripts/plot_impute.py           -> docs/img/impute_findings.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, ".")
from unwaste.config import default_config  # noqa: E402
from unwaste.data import build_panel  # noqa: E402
from unwaste.impute import correct_demand, demand_truth_test  # noqa: E402

syn = "--synthetic" in sys.argv
cfg = default_config(use_synthetic=syn)
p = build_panel(cfg)
corrected, daily_dem = correct_demand(p, cfg)
truth = demand_truth_test(p, cfg)

BLUE, GREY, RED, INK = "#2B59F3", "#9AA3B5", "#D64545", "#14171F"
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1.3, 1, 1.3]})
fig.suptitle(f"Stockout fill-in (impute) on {p.source}: {p.meta.shape[0]} store-items, {p.daily['dt'].nunique()} days",
             fontsize=14, fontweight="bold", x=0.01, ha="left")

# (A) one real day with a stockout: recorded sales vs corrected demand, empty hours shaded
oos_n = p.hourly_oos.sum(axis=1)
cand = np.flatnonzero((oos_n >= 3) & (oos_n <= 10))
row = int(cand[np.argmax(p.hourly_sales[cand].sum(axis=1))])
h = np.arange(24)
a = ax[0]
for hh in h[p.hourly_oos[row]]:
    a.axvspan(hh - 0.5, hh + 0.5, color=RED, alpha=0.12, lw=0)
a.bar(h, p.hourly_sales[row], color=GREY, label="Sales recorded")
a.plot(h, corrected[row], color=BLUE, lw=2.5, marker="o", ms=4, label="Demand after fill-in")
d = p.daily.iloc[row]
a.set_title(f"A · One real stockout day (store {d['store_id']}, item {d['product_id']}, {str(d['dt'])[:10]})\n"
            f"shaded = shelf empty · day total {p.hourly_sales[row].sum():.0f} recorded → {corrected[row].sum():.0f} demand",
            fontsize=10.5, loc="left")
a.set_xlabel("Hour of day"); a.set_ylabel("Units per hour"); a.set_xticks(range(0, 24, 3))
a.set_ylim(0, max(corrected[row].max(), p.hourly_sales[row].max()) * 1.2); a.legend(frameon=False)

# (B) demand-truth test: error of three methods on hidden blocks we know the answer for
b = ax[1]
names = ["Trust the zeros", "Simple average", "Our model"]
vals = [100 * truth["wape_naive_zero"], 100 * truth["wape_series_average"], 100 * truth["wape_model"]]
bars = b.bar(names, vals, color=[RED, GREY, BLUE])
for rect, v in zip(bars, vals):
    b.text(rect.get_x() + rect.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", fontweight="bold")
b.set_ylim(0, 115); b.set_ylabel("Error on hidden demand (WAPE, %)")
b.set_title(f"B · Demand-truth test: {truth['hidden_blocks']} afternoon blocks hidden\n"
            f"model recovers {truth['recovered_pct_model']:.0f}% of hidden demand (zeros: 0%)", fontsize=10.5, loc="left")

# (C) all store-items, every day: total recorded sales vs total demand after fill-in
c = ax[2]
dd = p.daily.assign(demand=daily_dem.to_numpy()).groupby("dt")[["sales", "demand"]].sum()
c.fill_between(dd.index, dd["sales"], dd["demand"], color=BLUE, alpha=0.15, label="Demand hidden by empty shelves")
c.plot(dd.index, dd["sales"], color=GREY, lw=2, label="Sales recorded")
c.plot(dd.index, dd["demand"], color=BLUE, lw=2.5, label="Demand after fill-in")
lift = 100 * (dd["demand"].sum() / dd["sales"].sum() - 1)
c.set_ylim(0, dd["demand"].max() * 1.15); c.set_ylabel("Units per day (all store-items)")
c.set_title(f"C · Every day, all store-items: demand is {lift:+.0f}% above recorded sales\n"
            f"{int(p.hourly_oos.sum()):,} empty-shelf hours corrected", fontsize=10.5, loc="left")
c.legend(frameon=False, loc="lower left"); c.tick_params(axis="x", rotation=30)

for x in ax:
    x.spines[["top", "right"]].set_visible(False)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = Path("docs/img/impute_findings.png"); out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160)
print(f"saved {out}")
