"""
Demo 3 · Forecast (step 3): one PNG showing the P10-P90 range vs what really happened.
Run: python scripts/plot_forecast.py        -> docs/img/forecast_findings.png
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
from unwaste.forecast import fit_forecast, to_matrix  # noqa: E402
from unwaste.impute import correct_demand  # noqa: E402

syn = "--synthetic" in sys.argv
cfg = default_config(use_synthetic=syn)
p = build_panel(cfg)
_, dem = correct_demand(p, cfg)
p.daily["demand"] = dem.to_numpy()
Y_cor, dates = to_matrix(p.daily, "demand")
Y_raw, _ = to_matrix(p.daily, "sales")
F = fit_forecast(Y_cor, p.daily, cfg, " corrected")
Fr = fit_forecast(Y_raw, p.daily, cfg, " raw")
te = F["train_end"]
T = Y_cor.shape[1]
test = np.arange(te + 1, T)                      # days we forecast, never trained on


def wape(pred, act):
    return 100 * np.abs(pred - act).sum() / act.sum()


# next-day forecasts for every test day
p50 = F[0.5][:, test - 1, 0]; p10 = F[0.1][:, test - 1, 0]; p90 = F[0.9][:, test - 1, 0]
act = Y_cor[:, test]
ma7 = np.stack([Y_cor[:, d - 7:d].mean(axis=1) for d in test], axis=1)
w_ours, w_raw, w_ma = wape(p50, act), wape(Fr[0.5][:, test - 1, 0], act), wape(ma7, act)
inside = 100 * ((act >= p10) & (act <= p90)).mean()

BLUE, GREY, INK, RED = "#2B59F3", "#9AA3B5", "#14171F", "#D64545"
fig, ax = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
fig.suptitle(f"Step 3 · Forecast on {p.source}: train {dates[0].date()} → {dates[te].date()}, "
             f"test {dates[test[0]].date()} → {dates[-1].date()} (never seen in training)",
             fontsize=13, fontweight="bold", x=0.01, ha="left")

# (A) one busy store-item: range vs reality
s = int(np.argmax(Y_cor[:, test].sum(axis=1)))
a = ax[0]
a.plot(dates[:te + 1], Y_cor[s, :te + 1], color=GREY, lw=1.5, label="History (training)")
a.fill_between(dates[test], p10[s], p90[s], color=BLUE, alpha=0.18, label="Forecast range P10–P90")
a.plot(dates[test], p50[s], color=BLUE, lw=2.5, label="Forecast P50 (middle)")
a.plot(dates[test], act[s], color=INK, lw=0, marker="o", ms=5, label="What really happened")
a.axvline(dates[te], color=RED, ls="--", lw=1.5); a.text(dates[te], a.get_ylim()[1] * 0.95, "  today: train | test", color=RED)
m = p.meta.sort_values("series_id").iloc[s]
a.set_title(f"A · Store {int(m['store_id'])}, item {int(m['product_id'])}: next-day forecast every night for 4 weeks", fontsize=11, loc="left")
a.set_ylabel("Units per day (demand after fill-in)"); a.set_ylim(0, None); a.legend(frameon=False, ncol=2, fontsize=9)

# (B) error comparison
b = ax[1]
names = ["7-day average", "LightGBM on\nraw sales", "LightGBM on\ncorrected demand"]
vals = [w_ma, w_raw, w_ours]
bars = b.bar(names, vals, color=[GREY, GREY, BLUE])
for r, v in zip(bars, vals):
    b.text(r.get_x() + r.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontweight="bold")
b.set_ylim(0, max(vals) * 1.3); b.set_ylabel("Next-day error (WAPE, %)  · lower is better")
b.set_title(f"B · {p.meta.shape[0]} store-items × {len(test)} test days\n"
            f"{inside:.0f}% of real days fall inside our P10–P90 range (ideal ≈ 80%)", fontsize=11, loc="left")
for x in ax:
    x.spines[["top", "right"]].set_visible(False)
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = Path("docs/img/forecast_findings.png"); out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160)
print(f"saved {out}  |  WAPE: 7-day avg {w_ma:.0f}%, raw {w_raw:.0f}%, ours {w_ours:.0f}%  |  inside range {inside:.0f}%")
