# Unwaste: evidence-based storyline

Every claim below has a number behind it and a tab that shows it. Real-data numbers are
from FreshRetailNet-50K (20 stores, 40 items, 97 days); "twin" numbers come from
`python -m unwaste.india` (fill them in from your run).

## 1. Problem (Problem & Idea, 25)
A dark store gets fresh stock in batches. Each batch has a sell-by line. Two things break
ordering in Indian cities: **sudden demand surges** (festivals, heatwaves, monsoon rain) and
**price shocks** at the mandi. Ordering by habit either runs out (lost sales) or throws stock away.

## 2. What we built (Technical Execution, 30)
| Step | What it does | Evidence |
|---|---|---|
| Fill-in | Empty-shelf hours hide true demand; we estimate it | Hidden-demand test: error 40% vs 45% (item average) vs 100% (zeros) |
| Forecast | LightGBM P10/P50/P90, next 7 days | Next-day error 22% on corrected demand |
| Batch risk | Monte Carlo replay of oldest-first selling → P(leftover) per batch | Store actions tab |
| Decide | Transfers (min-cost flow), smarter orders, early markdown | 0 shelf-life breaches |
| Replay | Same real demand, baseline vs Unwaste | **Waste 3.9% → 2.4% (−36%), service 85.9% → 89.2%** |

## 3. Indian conditions: the main objective
**Question:** can the system see a surge coming in an Indian city and order for it?

1. **Measured on real data:** heavy-rain days sell **+11%**, warm days **+7%** vs normal.
2. **Real Delhi weather replay** (Gurugram, June 2024: 34°C avg, 11 days ≥35°C, monsoon):
   waste **4.0% → 2.4%**, service **85.9% → 89.4%**.
3. **Indian twin, trained and tested:** the same real stores moved onto Delhi weather and the
   2024 festival calendar. Trained on 69 days (Eid ul-Fitr, Ram Navami, Akshaya Tritiya,
   Buddha Purnima, Delhi heatwave), tested on the next 28 (Bakrid, monsoon onset).
   - Festival-window error: without calendar ___% → with calendar ___%
   - Waste ___% → ___%, service ___% → ___%
4. **Mandi price spike** (tomato at 3.3× normal, generated in Agmarknet format):
   reading today's mandi price turns waste **+67% (worse)** into **−18% (better)** vs baseline.

## 4. Novelty: what's new here
1. **Batch-level expiry risk, not item-level stock.** We forecast whether *this batch* will be
   left on its sell-by line, with a probability, and act only when it's likely.
2. **Correcting demand hidden by stockouts** before forecasting. Raw sales understate demand by 19%.
3. **Indian-condition twin:** weather effects *measured* on real store data, transferred to
   Delhi's weather; festivals from the public calendar; train on past festivals, test on an unseen one.
4. **Reflect & correct (event radar):** the system shows the expected surge for each upcoming
   festival; the manager corrects it before ordering, and each correction becomes a labelled
   example for retraining. Human judgment where the model is weakest.
5. **Price-aware ordering:** today's mandi price, public before the night's order, adjusts the
   forecast, with a deliberately wrong elasticity guess (−0.3 vs true −0.4) so it isn't rigged.

## 5. How the vendor knows a festival is coming (Feasibility, 10)
- **Festivals:** public calendar (gazetted + regional festivals), loaded per city weeks ahead.
  The model gets "days to festival" and "festival size" as inputs.
- **Weather:** IMD / Open-Meteo forecast for the next 7 days, same inputs the model trained on.
- **Mandi prices:** Agmarknet daily modal prices, published each day.
- **Manager:** sees the event radar, corrects the surge, and the correction is logged.

## 6. Design & UX (20)
One manager screen: store actions (what to move, mark down, order), replay scoreboard,
proof tests, and the India tab with the event radar where the manager types a correction
and sees the extra units update.

## 7. Honest limits (say them before a judge does)
- Store demand is real but comes from a non-Indian city; Indian conditions are layered on top.
- Festival surge size is an **assumption** (ramp D-3..D+1). Weather effects are **measured**.
- Mandi prices in the demo are **generated**; the code reads real Agmarknet CSVs as-is.
- Weather inputs don't lower average next-day error (21.8% vs 21.6%): recent sales already carry it.
- Calendar catches only part of the festival surge; that's why the manager correction exists.

## 8. Path forward
Live Agmarknet prices → mandi price as a model input; real Indian store sales to *learn*
festival and price effects instead of assuming them; manager corrections retrain the model weekly.
