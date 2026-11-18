# Learn Phase 2: proof tests (branch `phase-2`)

**When:** 09:30–10:15, after the mid-eval. Learn first, then run it:
```bash
git checkout phase-2
python -m pytest -q                       # 11 tests
python -m unwaste.proof                   # ~3–5 min on real data → web/data/proof.json
uvicorn unwaste.api:app --port 8000       # new tab: "Proof tests"
```
If it looks right: `git checkout main && git merge phase-2 && git push`.
If anything breaks, stay on `main`: the 9 AM demo is untouched.

**Idea of Phase 2:** our slide promised four tests that try to *break* our own model. All four live in
`unwaste/proof.py`, and each result is a table/card in the Proof tab.

---

## Test 1 · Forecast baselines: `test_baselines` + `baseline_forecasts`
- Two simple forecasts anyone would try first:
  - **7-day moving average**: tomorrow ≈ the average of the last 7 days.
  - **Same weekday last week**: next Tuesday ≈ last Tuesday.
- Same error measure for all: **WAPE** = Σ|forecast − actual| ÷ Σ actual, over the 28-day test window,
  for the next day and for the whole next week (`forecast_errors`).
- **Why it matters:** if LightGBM can't beat a 7-day average, the ML is decoration. On real data last
  night, LightGBM next-day WAPE was 22%; this test shows the baselines next to it.
- Unit test: on a perfectly repeating weekly pattern, "same weekday last week" has 0% error. That proves the metric works.

> **Check:** What does WAPE 22% mean in words? Why compare against "same weekday last week" specifically?

## Test 2 · New stores: `test_new_stores`
- Pick 25% of the 20 stores at random (fixed seed) and **pretend they opened 14 days before the test**:
  their older history is set to `NaN` (`Yn[is_new, :te-13] = np.nan`), and they're **excluded from training**
  (`train_mask=~is_new`).
- Features became NaN-aware in `forecast.py` (`np.nanmean`): a new store's "28-day average" uses only the 14 days it has.
- Compare three numbers on those stores: the model with only 14 days, the same model if they had full history, and a 7-day average.
- **Why it matters:** Blinkit opens ~300 stores a quarter. A global model borrows patterns from similar stores.

> **Check:** Why must new stores be excluded from training, not just have short history?

## Test 3 · Unseen cities: `test_unseen_cities`
- Loads the **4 biggest stores from every city** (top 20 items per city) with `load_frn_multi` in `data.py`.
  It reads only those stores from disk (`filters=[("store_id","in",...)]`), so it stays fast on a laptop.
- Hold out **4 cities** (fixed seed). Train only on the other 14 (`train_mask=~in_held`), then forecast the 4 unseen ones.
- Compare: unseen-city error vs the error if the model *had* seen those cities vs a 7-day average.
- **Why it matters:** expanding to a new city shouldn't need months of data first.

> **Check:** If unseen-city error is only slightly worse than seen-city error, what does that tell a judge?

## Test 4 · Robustness: `test_robustness`
Same demand, rerun **baseline vs Unwaste** with ONE assumption changed each time:
| Case | Why we test it |
|---|---|
| Lead time 2 days | supply chains get slower |
| Shelf life ±25% | our shelf lives are assumptions from pack labels |
| Discounts lift demand 10% / 30% | we don't know Blinkit's price elasticity |
| Lost sale costs 25% of price | goodwill cost is an assumption |
| Transfer cost ₹10 | handling/transport may be pricier |
Columns: waste baseline → Unwaste, waste cut %, service, net margin change, after lost-sale cost, breaches (must be 0).

> **Check:** If one case shows Unwaste *losing* money, what do you say? (Answer: say it plainly, explain why
> (e.g. longer shelf life means less waste to save, so ordering more costs more than it saves), and that this tells Blinkit *which items* to switch Unwaste on for.)

---

## What changed in existing files (so you can explain the diff)
- `data.py`: `load_frn_multi` added; `build_panel(cfg, raw=None)` can now take pre-loaded rows; `city_id` kept in the panel.
- `forecast.py`: features use `nanmean/nanstd`; `fit_forecast(..., train_mask=...)` trains on a subset of series.
- `web/index.html`: new "Proof tests" tab (reads `web/data/proof.json`).
- `tests/test_core.py`: test for the baselines and the WAPE metric.

## 30-second judge answer for Phase 2
> "We tried to break our own model four ways. It beats simple forecasts, it works on new stores with two weeks
> of data and on cities it never saw, and Unwaste still cuts waste when we change our assumptions: slower supply,
> shorter or longer shelf life, different discount response, higher transfer cost. Every case has zero shelf-life breaches."
(Only say each claim once you've seen the real numbers in the Proof tab. If a number disagrees, say what it shows.)
