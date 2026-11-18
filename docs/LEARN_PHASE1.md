# Learn Phase 1: the code that runs at 9 AM

**Goal:** any of you can answer "show me where X happens" for any step, without help.
**How:** for each step, open the file at the line given, read only that function, then answer the
check question out loud *without looking*. If you can't, reread. Don't move on until you can.

Tonight (45 min, together, one laptop): Steps 1–5. Tomorrow 7:00–8:15: Steps 6–9 + the quiz at the end.

---

## Step 1 · Data: `unwaste/data.py` → `build_panel` (line 130)

**Idea.** One row per store × item × day, plus two 24-number arrays: sales per hour and "was the shelf empty" per hour.
- `load_frn` (line 58): picks the city with the most stores, then the top 20 stores by sales. Reads only that city (`filters=`) so it's fast.
- Line ~145: **auto-checks the meaning of the stock flag**. Hours flagged `1` must sell less, so `1` = empty shelf.
- Nights (outside 6:00–22:00) are closed, not stockouts.
- Sales are anonymised (0.2 etc.), so we **rescale** so an average item sells ~10 units/day.
- `meta`: price and shelf life aren't in the data, so they're **assumed per product**: deterministic, from a hash of the product id, so they never change between runs.
- Sell-by line = shelf life − 30% (FSSAI): `min_remaining = ceil(0.3 × shelf_life)`.

> **Check:** Why don't we treat 2 AM zeros as stockouts? Where would price come from in Blinkit's version?

---

## Step 2 · Fill-in demand: `unwaste/impute.py` → `correct_demand` (line 75)

**Idea.** Train a model **only on hours when the shelf had stock** (`fit = ~oos`), so it learns real demand. Predict the empty hours. Replace them, but never below what actually sold (`np.maximum`).
- `_hourly_frame` (line 28): turns the 24-hour arrays into one row per store-item-day-hour.
- `_add_profiles` (line 49): adds "average sales of this item at this hour", which is the strongest input.
- `_fit_predict` (line 60): `mean_model` = LightGBM with **Poisson** objective (sales are counts).
- Real run: 252,374 empty hours corrected → total demand **+19%** vs raw sales.

**Demand-truth test** `demand_truth_test` (line 93): take full-stock days, **hide every hour from a random afternoon hour to closing** (looks exactly like a real stockout), refit without them, compare.
Real result: trusting zeros = **100% error**, simple average = 45%, our model = **40%**, recovers **105%** of hidden demand.

> **Check:** Why train only on in-stock hours? Why hide *afternoon-to-closing* blocks rather than random hours?

---

## Step 3 · Forecast: `unwaste/forecast.py` → `fit_forecast` (line 72)

**Idea.** One global LightGBM per quantile (P10, P50, P90). Standing on night `o`, predict each of the next 7 days.
- `_origin_features` (line 39): only uses days `≤ o` (**no peeking at the future**): last day, 7/14/28-day means, 14-day spread.
- `_build_rows` (line 47): one row per (series, night, horizon h). Adds weekday, same-weekday history, planned discount/holiday.
- **Time split:** train nights end at `train_end = T − 28 − 1`; the last 28 days are only predicted, never trained on.
- Line ~97: forces P10 ≤ P50 ≤ P90.
- Real: next-day error (WAPE) **22%** on corrected demand vs 29% raw. 73% of actuals fall inside P10–P90 (ideal 80%, so the range is slightly too narrow: say so if asked).

> **Check:** What's a quantile model? Why is one global model better than one per store?

---

## Step 4 · Batch risk: `unwaste/risk.py` → `simulate_fefo` (line 32), `batch_risk` (line 68)

**Idea.** FEFO replayed many times.
- `sample_from_quantiles` (`backends.py` line 127): draws demand paths whose P10/P50/P90 match the forecast.
- `simulate_fefo`: for each day, sell oldest sell-by first (`order = sorted(... sellby)`); a batch can't sell after its sell-by (`b.sellby < day: continue`); on its sell-by day, whatever is left is recorded as leftover.
- `batch_risk`: 300 paths → `p_leftover` (share of paths with ≥1 left), `median_leftover`, and a histogram (the chart on the screen).

> **Check:** Walk through the test `test_fefo_sells_oldest_first_and_counts_leftover` in `tests/test_core.py`: 40 + 60 cups, 10/day. Why does the second batch end with 10?

---

## Step 5 · Transfers: `unwaste/backends.py` → `max_gain_transfers` (line 59)

**Idea.** Exactly the slide: maximise Σ x·(p − c − s) with sender surplus and receiver spare demand as limits.
- Arcs only exist when gain > 0 (unprofitable moves impossible).
- Built as **min-cost flow**: cost = −gain in paise (OR-Tools needs integers). Each sender also has a free arc to a **"dump" sink**, so moving is never forced.
- `test_slide_example_transfer_gain_is_1410`: S=30, D=32, gain 47 → moves 30 → ₹1,410.

> **Check:** Why negative costs? What does the "dump" arc mean in plain words?

---

## Step 6 · Decisions each night: `unwaste/replay.py` → `_nightly_actions` (line 198)

For every item, across all stores:
1. Risk per store (`batch_risk`). A batch is **at risk** if P(leftover) ≥ 60%, median leftover ≥ 2, and ≥ 2 days to sell-by.
2. Surplus = median leftover. Markdown value from `decide.markdown_value` (same random paths, with and without discount).
3. Receivers: same item, other stores. Capacity = `spare_demand` = 80% × (forecast next 3 days − sellable stock).
4. Guardrails: `shelf_life_ok` (arrives with ≥1 legal day) and **transfer must beat markdown per unit**.
5. `max_gain_transfers` → execute moves (stock appears at receiver next day, ₹2/unit cost).
6. Leftover surplus → early 15% markdown if it pays, else hold.

> **Check:** Name the two guardrails in code. Why surplus = *median* leftover, not the worst case?

---

## Step 7 · Ordering: `unwaste/decide.py` → `newsvendor_order` (line 25)

**Idea.** Perishable newsvendor by simulation.
- New batch is freshest → under FEFO it only sells what older stock can't (`residual = lost_by_day`).
- Try every order size Q: profit = price × sold + salvage × unsold − cost × Q − 50% price × lost sales. Pick the best.
- Guardrail: never below the 60th percentile of demand the batch can **legally** serve (window capped by sellable days).
- Baseline (`baseline_order`, line 54): 7-day average **sales** × 3 days × 1.15. Sales are censored by stockouts, so baseline under-orders.

> **Check:** Why does a short shelf life (4 days → 2 sellable) change the order?

---

## Step 8 · Replay: `unwaste/replay.py` → `run_policy` (line 87)

Day loop over the last 28 days, every store-item:
1. Sell today oldest-first (with markdown uplift if active). Count served vs demand → service level.
2. Throw away anything at its sell-by → waste (+ ₹1 salvage).
3. Nightly actions (Unwaste only).
4. Order every 3 days (staggered by series), arriving next day.

**Same demand `D` for all 4 policies**: `D` = corrected demand, rounded. Same starting stock.
Net margin = sales + salvage − purchases − transfer cost + stock left at cost.

**Real result (city 0, 20 stores, 40 items, 693 series, 4 weeks):**
| | Waste | Service | Net margin |
|---|---|---|---|
| Baseline | 3.9% | 85.9% | ₹35.55 L |
| Unwaste | **2.5%** | **90.0%** | ₹35.54 L (same) |

→ **Waste −30%, service +4.1 pts, same profit; +₹4.07 L after counting lost sales.**
Honest notes: "forecast on raw sales" has higher margin because it under-orders, and its service is lower (85.7%). Transfers alone take waste from 3.4% → 2.5%.

> **Check:** Why must all policies face identical demand? What does "time split" mean here?

---

## Step 9 · Screen: `web/index.html` + `unwaste/api.py`
`pipeline.py` writes 3 JSON files into `web/data/`; the page only reads them (no typed numbers). `api.py` serves the page + `/api/scoreboard`, `/api/actions`, `/api/demand-truth`.

---

## Morning quiz (8:00–8:15): each person answers 3, another person picks them
1. Where is the 30% FSSAI rule in code?
2. What does `p_leftover = 0.85` mean?
3. Why LightGBM Poisson for fill-in but quantile for forecast?
4. What would break if we didn't round demand in the replay?
5. Point to where a transfer is blocked if it doesn't beat a markdown.
6. Why is baseline service only 85.9%?
7. What does the sweep show when transfer cost goes ₹2 → ₹10?
8. If a judge says "your prices are made up", what's your answer?
9. Which numbers on the scoreboard are measured and which depend on assumptions?
