# Learn Phase 3: weather + Indian market data (branch `phase-3`)

Judges' feedback at mid-eval: *consider weather* and *use Indian market data*. Phase 3 does both.

```bash
git checkout phase-3
python -m pytest -q                         # 12 tests
python -m unwaste.pipeline --stores 20 --products 40 --sweep   # main results, now with weather in the forecast
python -m unwaste.india                     # → web/data/india.json  (needs data/raw/india/*)
uvicorn unwaste.api:app --port 8000         # tab "India & weather"
```

## 1 · Weather in the forecast (`forecast.py`, `india.weather_in_forecast`)
- The dataset already has daily **rain (mm), temperature (°C), humidity (%), wind level** for every store-day.
- They're now forecast inputs (`WEATHER = ["rain","temp","humidity","wind"]`), using the weather **of the day we predict**. In production that comes from a weather forecast (e.g. IMD), so there's no leakage.
- The tab shows: next-day error **with vs without** weather, and how demand moves on dry / light-rain / heavy-rain days and mild / warm / hot days (demand relative to each item's own average).
- **Say:** "Weather is now an input. Here's how much it reduces error, and here's how demand shifts on rainy and hot days."

## 2 · Indian weather what-if (`india.indian_weather_whatif`)
- Real **Gurugram / Delhi-NCR** weather for the **same calendar dates** as our replay (Open-Meteo archive, free).
- We feed that weather into the weather-aware model → it predicts how demand would shift (e.g. heatwave, monsoon onset) → we scale demand by that shift → replay **baseline vs Unwaste** on it.
- **Honest framing:** it's a *what-if*: "if these stores had Delhi's weather". The demand response comes from what the model learned from real store-weather data.

## 3 · Indian mandi price shock (`india.agmarknet_shock`)
- **Agmarknet** daily modal prices (Azadpur, Delhi) for tomato, onion, potato.
- Price index = price ÷ its 1-year median. We find the **worst 4-week spike** (e.g. tomato, July 2023).
- Turn it into a demand swing with a **stated price elasticity of −0.4** (+10% price → about −4% demand), then replay baseline vs Unwaste through it.
- **What we found on stand-in data:** when demand collapses **city-wide**, Unwaste keeps service higher but wastes more, because there's no store with spare demand to transfer to and the forecast takes a few days to catch up.
- **Say it plainly, then the fix:** "Our next step is to feed the mandi price itself into the forecast, so a price spike is anticipated, not discovered a week late."

## Other changes on this branch
- `config.relative_target` (off by default): predicts demand relative to the recent 14-day level, which helps forecasts follow sudden level shifts. Try `relative_target=True` in `config.py` and compare.
- Coaching notes ("say this to a judge", "Owner") removed from all code files.
- Test for the Open-Meteo and Agmarknet loaders.

## Check questions
1. Why is it fair to use the *target day's* weather as an input?
2. What does "demand index 110" mean on the what-if chart?
3. What is a price elasticity of −0.4, in one sentence?
4. Why can't transfers help when the shock hits every store at once?
