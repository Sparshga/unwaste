# Team guide: what to read, what to say

Each file starts with a box called **"WHAT THIS FILE DOES (say this to a judge)"**.
If a judge walks up, read that box in your own words, then point at one function.

## Who explains what

| Person | Files to own | The one function to point at |
|---|---|---|
| **Ujjwal** | `data.py`, `impute.py`, `forecast.py` | `impute.correct_demand`: model trained only on in-stock hours fills empty-shelf hours |
| **Sparsh** | `risk.py`, `decide.py`, `backends.py` | `risk.simulate_fefo` and `backends.max_gain_transfers` (the min-cost flow) |
| **Aniz** | `replay.py`, `pipeline.py`, `web/index.html`, `api.py` | `replay.run_policy`: four policies on identical demand |

## 20-minute reading order (do it before sleeping)
1. `README.md`: the diagram and the transfer formula.
2. `config.py`: every assumption. Know: ₹2 transfer, ₹1 salvage, 30% FSSAI rule, reorder every 3 days.
3. Your own files: read the top box, then the main function only.
4. `tests/test_core.py`: the slide example (+₹1,410, 30 cups left) is a passing test.

## Questions judges will ask while walking around

**"What's running right now?"**
"The nightly job on one city of real data: it fills hidden demand, forecasts ranges, replays four
weeks for four policies on the same demand, and writes the action list you see on screen."

**"Is this real data?"**
"Yes: FreshRetailNet-50K, 898 small fresh stores with hourly sales and stockout hours marked. Prices,
shelf lives and costs aren't in the data, so they're stated assumptions in config.py, and the badge on
the screen shows the data source."

**"How do you know the stockout correction works?"** (Ujjwal)
"We hide afternoon hours on full-stock days, where we know the true sales, and check the model gets them back.
Trusting the zeros is 100% wrong; our model recovers about all of the hidden demand."

**"Walk me through one decision."** (Sparsh)
Click a batch on the screen: "Monte Carlo replayed the next week 300 times; in X% of them units are
left on the sell-by line. Store Y is short of this item, so the optimiser moves N units: gain =
N × (price − ₹2 − ₹1)."

**"Why is baseline worse?"** (Aniz)
"Baseline reorders from recent sales, but sales are censored by empty shelves, and it only discounts
on the last day. Same demand for every policy, so the difference is purely the decisions."

**"What did you NOT do?"**
"No Blinkit data, so the rupees are a replay on public data with stated prices. Camera reading of
expiry labels is a stretch goal. A pilot with 2 + 2 stores would measure the real effect."

## If something breaks during the demo
- No real data → `python -m unwaste.pipeline --synthetic` (the screen shows a red SYNTHETIC badge; say so).
- LightGBM won't install → the code falls back to scikit-learn automatically (shown in the header badge).
- Screen blank → you didn't run the pipeline yet, or open via `uvicorn unwaste.api:app` / `python -m http.server -d web`.
