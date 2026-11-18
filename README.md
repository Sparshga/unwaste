# Unwaste

**Batch-level expiry risk and early action for quick-commerce dark stores.**
FEFO picks the pack. Unwaste saves the batch.

Every night Unwaste asks of every batch on every shelf: *will it sell before its sell-by line?*
If not, it prices the options in rupees (transfer to a store that's short, mark down early, or hold)
and right-sizes the next order so the problem doesn't repeat.

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # Mac: if "import lightgbm" fails ->  brew install libomp
./scripts/download_data.sh               # FreshRetailNet-50K from Hugging Face (~115 MB)
python scripts/inspect_data.py           # confirm the columns
python -m pytest -q                      # 10 tests, ~15 s
python -m unwaste.pipeline --stores 20 --products 40 --sweep   # the full run (a few minutes)
uvicorn unwaste.api:app --port 8000      # open http://localhost:8000
```

No data yet? `python -m unwaste.pipeline --synthetic` runs the identical pipeline on stand-in data
with the same columns. The screen shows a red **SYNTHETIC** badge so it can never be mistaken for real results.

---

## How it works: one nightly job

```
 parquet ─► 1 Clean & prepare ─► 2 Fill-in demand ─► 3 Forecast P10/P50/P90 ─► 4 Batch risk (Monte Carlo FEFO)
                                                                                     │
           9 Screen ◄── 8 Replay scoreboard ◄── 7 Guardrails ◄── 5 Recover (OR-Tools) + 6 Prevent (newsvendor)
```

| Step | File | In → Out |
|---|---|---|
| 0 Config | `unwaste/config.py` | every assumption in one place |
| 1 Clean & prepare | `unwaste/data.py` | parquet → daily panel + 24-hour sales and stockout flags |
| 2 Fill-in demand | `unwaste/impute.py` | in-stock hours only → estimates for empty-shelf hours; demand-truth test |
| 3 Forecast | `unwaste/forecast.py` | corrected history → P10 / P50 / P90 for the next 7 days |
| 4 Batch risk | `unwaste/risk.py` | batches + forecast range → P(leftover), units left, histogram |
| 5–7 Decide | `unwaste/decide.py` | transfer gain, markdown value, perishable newsvendor, guardrails |
| Solvers | `unwaste/backends.py` | LightGBM / OR-Tools (fallbacks: scikit-learn / NetworkX) |
| 8 Replay | `unwaste/replay.py` | 4 policies × identical demand → scoreboard + action log |
| Run all | `unwaste/pipeline.py` | writes `web/data/*.json` |
| 9 Screen | `web/index.html`, `unwaste/api.py` | store action list, scoreboard, demand-truth charts |

### The transfer problem (step 5)

```
decision   x_ij = units moved from store i to store j   (same item, same city)
maximise   Σ x_ij · (p − c_ij − s_i)          gain over dumping: price − transfer cost − salvage
subject to Σ_j x_ij ≤ S_i                     sender surplus  = median leftover from Monte Carlo
           Σ_i x_ij ≤ D_j                     receiver spare  = forecast demand − sellable stock
           x_ij = 0 if shelf life on arrival < FSSAI minimum
           x_ij = 0 if the move doesn't beat a markdown per unit
           x_ij ≥ 0, whole units
```
Solved exactly as a min-cost flow (costs = −gain in paise; unused surplus flows to a zero-cost "dump" sink).

### The ordering rule (step 6): perishable newsvendor by simulation
The new batch is the freshest, so under FEFO it only sells what older stock can't. For each candidate
order size Q we compute expected `price × sold + salvage × unsold − cost × Q − penalty × lost sales`
over Monte Carlo demand paths, and pick the best Q, never below the 60th percentile of the demand
the batch can legally serve (service-level floor).

---

## What the replay compares (same demand for everyone)

| Policy | Ordering | Actions |
|---|---|---|
| Baseline | recent 7-day average sales × 1.15 | FEFO + 30% markdown on a batch's last day |
| Forecast on raw sales | perishable newsvendor on forecasts of raw (censored) sales | same as baseline |
| + Stockout correction | same, on corrected demand | same as baseline |
| **Unwaste** | same, on corrected demand | + batch risk → transfers / early markdown |

Time split: train on the first ~62 days, replay the last 28.
Metrics: units wasted, waste % of received, service level (demand served), net margin
(sales + salvage − purchases − transfer cost + stock left at cost), false moves, shelf-life breaches.

## Data and assumptions (stated, not hidden)
- **FreshRetailNet-50K** (CC BY 4.0): hourly sales of perishables in small fresh stores, with stockout hours marked.
  Sales are anonymised, so we rescale them to ~10 units/day per series (`target_mean_daily_units`).
- Not in the data, so assumed per item (deterministic, see `config.py`): price ₹30–120, cost 70% of price,
  shelf life 4/5/7/10 days, salvage ₹1, transfer cost ₹2/unit (sweep: ₹2/5/10), lead time 1 day, reorder every 3 days.
- FSSAI rule: ≥30% of shelf life must remain at delivery → sell-by line = expiry − 30% of shelf life.
- Results prove the **mechanism** on public data; only a pilot (2 + 2 stores, 4 weeks) can measure the Blinkit effect.
