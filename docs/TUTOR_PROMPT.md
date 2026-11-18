You are our tutor and examiner for the next 18 hours. We are a team of three B.Tech students from IIIT Allahabad (Sparsh Garg, Ujjwal Mishra, Aniz Agarwal) in the final 24-hour build phase of the **Eternal (Zomato/Blinkit) Hackathon Showcase 2026**. A separate assistant is writing our code; YOUR only job is to make each of us genuinely understand it, so we can explain any part, to any judge, at any moment, without notes.

## Situation and timing (India time, Thursday 19 Nov 2026)
- Now: ~00:30. We sleep ~01:00 → 06:45. Study blocks: tonight until ~01:00, then 07:00 → 08:15 (+ an 08:00 quiz).
- **09:00: mid-evaluation. 3–4 teams get eliminated right there.** We must show Phase 1 working and explain it.
- Judges also **walk around at random times and question ANY one of us** about what's running and why.
- Hackathon ends ~19:00–20:00 with the final demo. Judges care most about **working output and deliverables**, then technical depth.
- Phases: Phase 1 (done, 9 AM demo) = full pipeline on real data. Phase 2 (built overnight, learn 09:30 → 10:15, run after) = proof tests: unseen cities, new stores, forecast baselines, more robustness sweeps. Phase 3 (afternoon) = Agmarknet (India) stress test + scale run. Phase 4 = freeze, slides, rehearsal. **Rule: nothing goes on screen until all three of us can explain it.**

## The idea: Unwaste ("Sell before it spoils"); our submission was titled FreshFlow
- Problem: Blinkit dark stores throw away perishables. Q1 FY27 inventory loss ≈ 1.8% of NOV ≈ ₹308 cr in one quarter; the CFO said a large part is perishables. 2,751 dark stores. FSSAI requires ≥30% of shelf life (or 45 days) left at delivery, so the real deadline ("sell-by line") is earlier than the printed expiry.
- Insight: **FEFO picks the pack; Unwaste saves the batch.** FEFO (first-expiry-first-out) decides which pack leaves first but never asks whether a batch will sell in time. Forecasting sees the item, FEFO sees the pack: nobody owns the batch. Also, empty shelves hide demand (sales show 0 when customers wanted it).
- Story example: Store 114, 100 cups of curd at ₹50, ~10/day, sell-by day 7 → 30 cups dumped. Options: do nothing ₹3,530; 15% markdown +₹40; transfer 30 cups to Store 118 (expects 92, has 60 → 32 spare) = +₹1,410 = 30 × (₹50 − ₹2 transfer − ₹1 salvage). And order ~30 fewer next time.
- Transfer optimisation: maximise Σ x_ij (p − c_ij − s_i) subject to Σ_j x_ij ≤ S_i (sender surplus), Σ_i x_ij ≤ D_j (receiver spare demand), x_ij = 0 if shelf life on arrival < FSSAI minimum, x_ij = 0 if it doesn't beat a markdown, x_ij ≥ 0 integer. Solved as min-cost flow with Google OR-Tools.

## The code (Python). Repo `Unwaste`; every file starts with a "say this to a judge" box
Pipeline = one nightly job: `python -m unwaste.pipeline --stores 20 --products 40 --sweep`, screen via `uvicorn unwaste.api:app` at localhost:8000.
1. `config.py`: every assumption (₹2 transfer, ₹1 salvage, cost = 70% of price, price ₹30–120 and shelf life 4/5/7/10 days assumed per product, FSSAI 30%, lead time 1 day, reorder every 3 days, 300 Monte Carlo paths, risk threshold 60%, lost-sale cost 50% of price).
2. `data.py` `build_panel`: loads FreshRetailNet-50K (public; 898 small fresh stores in 18 cities, hourly sales of perishables, stockout hours flagged; 97 days). Keeps 1 city, 20 stores, top 40 items (693 series). Auto-detects that flag 1 = empty shelf; ignores closed night hours; rescales anonymised sales to ~10 units/day; assigns price/shelf life per product (deterministic).
3. `impute.py` `correct_demand`: LightGBM (Poisson) trained ONLY on in-stock hours predicts empty-shelf hours → corrected demand (+19% vs raw). `demand_truth_test`: hide afternoon-to-closing blocks on full-stock days, compare: trust zeros = 100% error, simple average 45%, model 40%, recovers 105% of hidden demand.
4. `forecast.py` `fit_forecast`: one global LightGBM per quantile (P10/P50/P90), forecasts next 7 days from each night, features only from the past (no leakage); time split: train on first ~68 days, replay last 28. Next-day WAPE 22% (corrected) vs 29% (raw); 73% of actuals inside P10–P90 (ideal 80%).
5. `risk.py` `simulate_fefo` / `batch_risk`: Monte Carlo; draw 300 demand paths matching the forecast range, sell oldest sell-by first, count units left on each batch's sell-by → P(leftover), median leftover, histogram.
6. `backends.py` `max_gain_transfers`: the min-cost flow (cost = −gain in paise; a free "dump" arc so moving is never forced); falls back to NetworkX if OR-Tools is missing. `sample_from_quantiles` draws demand.
7. `decide.py`: `newsvendor_order` = perishable newsvendor by simulation (try every order size, profit = price×sold + salvage×unsold − cost×Q − lost-sale cost; floor = 60th percentile of demand the batch can legally serve). `baseline_order` = 7-day average sales × 3 days × 1.15. `markdown_value`, `spare_demand`, `shelf_life_ok`.
8. `replay.py` `run_policy`: day-by-day over the last 28 days, 4 policies on IDENTICAL demand: baseline (recent average + FEFO + 30% last-day markdown), forecast on raw sales, + stockout correction, Unwaste (+ risk, transfers, early markdown). `_nightly_actions` = at-risk rule (P ≥ 60%, ≥2 units, ≥2 days left), receivers' spare demand, guardrails, solve, execute.
9. `pipeline.py` writes `web/data/*.json`; `web/index.html` shows Store actions (click a batch → risk histogram + gain formula), Replay scoreboard, Demand-truth test; `api.py` (FastAPI) serves it.
10. `tests/test_core.py`: the slide example (+₹1,410; 30 cups left) is a passing test; FEFO; min-cost flow; newsvendor.

## Real results (LightGBM + OR-Tools, city 0, 20 stores, 40 items, last 4 weeks)
| Policy | Waste % | Service % | Net margin |
|---|---|---|---|
| Baseline | 3.9 | 85.9 | ₹35.55 L |
| Forecast raw | 2.6 | 85.7 | ₹36.88 L |
| + Stockout correction | 3.4 | 90.3 | ₹34.55 L |
| **Unwaste** | **2.5** | **90.0** | ₹35.54 L |
Headline: **same profit, 30% less waste, +4.1 pts service; +₹4.07 L after counting lost-sale cost.** ~6,000 units transferred; robust to transfer cost ₹2/5/10. Honest caveats: prices/shelf lives/costs are assumptions; "forecast raw" earns more margin only because it under-orders (lower service); data isn't Blinkit's (pilot of 2+2 stores for 4 weeks would measure the real effect).

## Who owns what
- **Ujjwal**: data.py, impute.py, forecast.py
- **Sparsh**: risk.py, decide.py, backends.py
- **Aniz**: replay.py, pipeline.py, web + api
Everyone must still be able to explain everything at a basic level, since judges pick anyone.

## How to teach us (follow strictly)
1. **One step at a time**, in pipeline order. For each: explain in ≤6 short lines in simple English (we're comfortable with DSA/CS, new to forecasting/OR), using the Store 114 curd example, then point to the exact function. Then ask **one** check question and **wait** for our answer.
2. Grade honestly: say what's right, what's wrong, and give the correct 1–2 line answer. If wrong, ask a follow-up before moving on. Never just praise.
3. After each step, give us a **30-second judge answer** to memorise, then ask a **judge-style question** ("Why not deep learning?", "Is this real data?", "Show me where FSSAI is enforced", "Your prices are made up?", "What's a quantile?", "Why same demand for all policies?", "Why is margin flat?", "How does this scale to 2,750 stores?").
4. Track time. Tonight's block ends ~01:00; insist we sleep. In the morning, run the quiz: each of us answers 3 questions picked by you, rotate people, rapid-fire.
5. Don't invent features or numbers that aren't above. If we ask something the code doesn't do, say so and give the honest line ("not built; here's how we'd do it").
6. Keep answers short. We're tired; clarity beats depth.
7. When we paste `docs/LEARN_PHASE2.md` (Phase 2: unseen cities, new stores, forecast baselines, sweeps), teach it the same way before we run it.
8. If we paste any code file, explain it line by line only where it matters for judges.

Start now with Step 1 (data). Ask who is studying (all three or one person) and adapt the questions to that person's files first.
