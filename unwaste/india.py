"""
Phase 3 · Weather and Indian market data.

  1. Weather in the forecast: rain, temperature, humidity and wind of the target day are
     model inputs (in production they come from a weather forecast, e.g. IMD). We measure
     forecast error with vs without weather, and how demand moves on rainy / hot days.
  2. Indian weather what-if: the same stores and decisions, but with the real Delhi-NCR
     weather of the same dates (Open-Meteo archive). The weather-aware model turns that
     weather into a demand shift; then baseline and Unwaste are replayed on it.
  3. Indian market shock: Agmarknet mandi prices (Azadpur, Delhi). We find the worst
     4-week price spike (e.g. the tomato spike), turn it into a demand swing with a
     stated price elasticity, and replay baseline vs Unwaste through it.

Run:  python -m unwaste.india
      python -m unwaste.india --synthetic      (stand-in data + generated Indian files)
Inputs: data/raw/india/weather_*.json (Open-Meteo), data/raw/india/agmarknet_*.csv
Output: web/data/india.json  (shown in the 'India & weather' tab)
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, default_config
from .data import build_panel
from .forecast import exo_matrices, WEATHER, fit_forecast, to_matrix
from .impute import correct_demand
from .replay import POLICIES, run_policy

INDIA_DIR = Path("data/raw/india")
PRICE_ELASTICITY = -0.4          # +10% price -> about -4% demand (stated assumption)
FORECAST_ELASTICITY = -0.3       # what the price-aware forecast assumes: deliberately NOT the true -0.4


# ----------------------------------------------------------------------------- loaders
def load_open_meteo(path: Path) -> pd.DataFrame:
    """Open-Meteo daily JSON -> date-indexed rain (mm), temp (°C), humidity (%), wind (Beaufort level)."""
    js = json.loads(Path(path).read_text())
    if js.get("error"):
        raise ValueError(f"Open-Meteo returned an error: {js.get('reason')}")
    d = js["daily"]
    wind_key = next(k for k in d if k.startswith("wind_speed"))
    kmh = np.asarray(d[wind_key], float)
    df = pd.DataFrame({
        "rain": np.asarray(d["precipitation_sum"], float),
        "temp": np.asarray(d["temperature_2m_mean"], float),
        "humidity": np.asarray(d.get("relative_humidity_2m_mean", [np.nan] * len(kmh)), float),
        # dataset wind is a Beaufort-like level (~1-3); convert km/h -> m/s -> Beaufort
        "wind": np.round((kmh / 3.6 / 0.836) ** (2 / 3), 2),
    }, index=pd.to_datetime(d["time"]))
    return df.ffill().bfill()


def load_agmarknet(folder: Path = INDIA_DIR) -> dict:
    """Any Agmarknet / data.gov.in CSV with a date column and a modal-price column."""
    out = {}
    for f in sorted(glob.glob(str(folder / "agmarknet_*.csv"))):
        raw = pd.read_csv(f)
        cols = {c.lower(): c for c in raw.columns}
        date_c = next((cols[c] for c in cols if "date" in c), None)
        price_c = next((cols[c] for c in cols if "modal" in c), None)
        arr_c = next((cols[c] for c in cols if "arrival" in c and "date" not in c), None)
        if not date_c or not price_c:
            print(f"[india] skip {f}: need a date column and a modal-price column, got {list(raw.columns)}")
            continue
        d = pd.DataFrame({"date": pd.to_datetime(raw[date_c], dayfirst=True, errors="coerce"),
                          "price": pd.to_numeric(raw[price_c].astype(str).str.replace(",", ""), errors="coerce")})
        if arr_c:
            d["arrivals"] = pd.to_numeric(raw[arr_c].astype(str).str.replace(",", ""), errors="coerce")
        d = d.dropna(subset=["date", "price"]).groupby("date").mean().sort_index()
        d = d.asfreq("D").interpolate(limit=7).dropna(subset=["price"])
        name = Path(f).stem.replace("agmarknet_", "").title()
        out[name] = d
        print(f"[india] Agmarknet {name}: {d.index.min().date()} → {d.index.max().date()}, {len(d)} days")
    return out


# ----------------------------------------------------------------------------- helpers
def _wape(P50, Y, te, H=7, mask=None):
    """Next-day WAPE over the test window; mask[S, T] keeps only some store-days (e.g. heavy rain)."""
    err = tot = 0.0
    for h in (1,):
        for o in range(te, Y.shape[1] - h):
            keep = np.ones(Y.shape[0], bool) if mask is None else mask[:, o + h]
            err += np.abs(P50[keep, o, h - 1] - Y[keep, o + h]).sum()
            tot += Y[keep, o + h].sum()
    return float(round(100 * err / max(tot, 1e-9), 1))


def _replay_pair(D, S_obs, F, meta, cfg) -> dict:
    b = run_policy(POLICIES["baseline"], D, S_obs, F, meta, cfg).summary()
    u = run_policy(POLICIES["unwaste"], D, S_obs, F, meta, cfg).summary()
    keep = ["waste_units", "waste_pct_of_received", "service_level_pct", "net_margin_rs",
            "net_after_lost_sales_rs", "transfer_units", "shelf_life_breaches"]
    return {"baseline": {k: b[k] for k in keep}, "unwaste": {k: u[k] for k in keep},
            "waste_reduction_pct": round(100 * (b["waste_units"] - u["waste_units"]) / max(b["waste_units"], 1), 1),
            "service_change_pts": round(u["service_level_pct"] - b["service_level_pct"], 2),
            "net_gain_rs": u["net_margin_rs"] - b["net_margin_rs"],
            "net_after_lost_gain_rs": u["net_after_lost_sales_rs"] - b["net_after_lost_sales_rs"]}


def _scale_test(Y, te, mult):
    """Multiply demand in the test window by a [S, T] or [T] multiplier."""
    out = Y.copy()
    m = np.asarray(mult)
    if m.ndim == 1:
        m = np.broadcast_to(m[None, :], Y.shape)
    out[:, te + 1:] = Y[:, te + 1:] * m[:, te + 1:]
    return out


# ----------------------------------------------------------------------------- 1. weather in forecast
def weather_in_forecast(Y, daily, cfg) -> tuple[dict, dict]:
    F_w = fit_forecast(Y, daily, cfg, " with weather")
    F_n = fit_forecast(Y, daily, replace(cfg, use_weather=False), " without weather", quiet=True)
    te = F_w["train_end"]
    res = {"wape_with_weather": _wape(F_w[0.5], Y, te), "wape_without_weather": _wape(F_n[0.5], Y, te)}
    # how demand moves with weather: demand relative to each item's own average
    d = daily.copy()
    d["rel"] = d["demand"] / d.groupby("series_id")["demand"].transform("mean").replace(0, np.nan)
    rain_b = pd.cut(d["rain"], [-0.1, 1, 10, 1e9], labels=["Dry (<1 mm)", "Light rain (1–10 mm)", "Heavy rain (>10 mm)"])
    temp_b = pd.cut(d["temp"], [-50, 25, 32, 60], labels=["Mild (<25°C)", "Warm (25–32°C)", "Hot (>32°C)"])
    res["by_rain"] = [{"bucket": str(k), "demand_vs_average_pct": round(100 * (v - 1), 1), "days": int(n)}
                      for (k, v), n in zip(d.groupby(rain_b, observed=True)["rel"].mean().items(),
                                           d.groupby(rain_b, observed=True).size())]
    res["by_temp"] = [{"bucket": str(k), "demand_vs_average_pct": round(100 * (v - 1), 1), "days": int(n)}
                      for (k, v), n in zip(d.groupby(temp_b, observed=True)["rel"].mean().items(),
                                           d.groupby(temp_b, observed=True).size())]
    # the average hides rare days: error on rainy store-days only
    rain = exo_matrices(daily)["rain"]
    for key, m in (("rainy", rain > 1), ("heavy_rain", rain > 10)):
        res[f"wape_{key}_with"] = _wape(F_w[0.5], Y, te, mask=m)
        res[f"wape_{key}_without"] = _wape(F_n[0.5], Y, te, mask=m)
        res[f"n_{key}_store_days"] = int(m[:, te + 2:].sum())
    print(f"[india] forecast error with weather {res['wape_with_weather']}% vs without {res['wape_without_weather']}%")
    print(f"[india]   rainy days (>1 mm): {res['wape_rainy_with']}% vs {res['wape_rainy_without']}%; "
          f"heavy rain (>10 mm, {res['n_heavy_rain_store_days']} store-days): "
          f"{res['wape_heavy_rain_with']}% vs {res['wape_heavy_rain_without']}%")
    return res, F_w


# ----------------------------------------------------------------------------- 2. Indian weather what-if
def align_weather(wx: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """If the weather file is for another year, shift it onto the dataset's calendar days."""
    dates = pd.DatetimeIndex(dates)
    if wx.index.isin(dates).sum() < 0.5 * len(dates):
        shift = dates[0].year - wx.index[0].year
        wx = wx.copy()
        wx.index = wx.index + pd.DateOffset(years=shift)
        wx = wx[~wx.index.duplicated()]
        print(f"[india] weather file shifted by {shift:+d} year(s) to match the dataset's calendar days")
    if wx.index.isin(dates).sum() == 0:
        raise ValueError("Weather file does not cover the dataset's calendar days; download the full year")
    return wx


def indian_weather_whatif(Y, Y_raw, daily, meta, dates, F_w, wx: pd.DataFrame, cfg) -> dict:
    te = F_w["train_end"]
    dates = pd.DatetimeIndex(dates)
    wx = align_weather(wx, dates)
    override, used = {}, {}
    for w in WEATHER:
        base = daily.pivot_table(index="series_id", columns="dt", values=w, aggfunc="mean").mean(axis=0)
        vec = base.reindex(dates).to_numpy(float)
        india = wx[w].reindex(dates).to_numpy(float)
        vec = np.where(np.isnan(india), vec, india)
        override[w] = vec
        used[w] = india[te + 1:]
    F_i = fit_forecast(Y, daily, cfg, " Indian weather", quiet=True, weather_override=override)
    # weather-driven demand shift = next-day P50 under Indian weather / under original weather
    T = Y.shape[1]
    ratio = np.ones_like(Y)
    for t in range(te + 1, T):
        a, b = F_i[0.5][:, t - 1, 0], F_w[0.5][:, t - 1, 0]
        ratio[:, t] = np.clip(np.where(b > 0.1, a / np.maximum(b, 1e-6), 1.0), 0.5, 2.0)
    Y_i = _scale_test(Y, te, ratio)
    S_i = _scale_test(Y_raw, te, ratio)
    F_i2 = fit_forecast(Y_i, daily, cfg, " Indian weather (replay)", quiet=True, weather_override=override)
    res = _replay_pair(np.round(Y_i).astype(int), S_i, F_i2, meta, cfg)
    test_dates = dates[te + 1:]
    res.update({
        "place": "Gurugram / Delhi-NCR (Open-Meteo archive)",
        "window": [str(test_dates[0].date()), str(test_dates[-1].date())],
        "avg_temp_c": round(float(np.nanmean(used["temp"])), 1),
        "total_rain_mm": round(float(np.nansum(used["rain"])), 1),
        "heavy_rain_days": int(np.nansum(used["rain"] > 10)),
        "hot_days_35c": int(np.nansum(used["temp"] >= 35)),
        "demand_shift_pct": round(100 * (Y_i[:, te + 1:].sum() / max(Y[:, te + 1:].sum(), 1e-9) - 1), 1),
        "daily": {"dates": [str(d.date()) for d in test_dates], "rain": np.round(used["rain"], 1).tolist(),
                  "temp": np.round(used["temp"], 1).tolist(),
                  "demand_index": np.round(100 * Y_i[:, te + 1:].sum(0) / np.maximum(Y[:, te + 1:].sum(0), 1e-9), 1).tolist()},
    })
    print(f"[india] Indian weather what-if: demand {res['demand_shift_pct']:+.1f}%, "
          f"waste change {-res['waste_reduction_pct']:+.1f}%, service {res['service_change_pts']:+.2f} pts")
    return res


# ----------------------------------------------------------------------------- 3. Agmarknet shock
def agmarknet_shock(Y, Y_raw, daily, meta, agm: dict, cfg, test_days: int) -> dict:
    best = None
    for name, d in agm.items():
        idx = d["price"] / d["price"].rolling(365, min_periods=60).median()
        win = idx.rolling(test_days).mean()
        if win.dropna().empty:
            continue
        end = win.idxmax()
        if best is None or win.max() > best[1]:
            best = (name, float(win.max()), end, idx)
    if best is None:
        return None
    name, peak, end, idx = best
    start = end - pd.Timedelta(days=test_days - 1)
    path = idx.loc[start:end].to_numpy(float)
    if len(path) < test_days:
        path = np.pad(path, (test_days - len(path), 0), mode="edge")
    mult_test = np.clip(path ** PRICE_ELASTICITY, 0.4, 1.6)        # price up -> demand down
    T = Y.shape[1]
    te = T - test_days - 1
    mult = np.ones(T)
    mult[te + 1:] = mult_test[-(T - te - 1):]
    Y_s, S_s = _scale_test(Y, te, mult), _scale_test(Y_raw, te, mult)
    D_s = np.round(Y_s).astype(int)
    F_s = fit_forecast(Y_s, daily, cfg, " mandi shock", quiet=True)
    res = _replay_pair(D_s, S_s, F_s, meta, cfg)                    # forecast ignores the mandi price

    # Price-aware forecast: today's mandi price is public before tonight's order.
    # Divide past demand by the price effect, forecast the "price-free" demand, then multiply
    # back by today's price effect. It uses a guessed elasticity (-0.3), not the true -0.4.
    m_hat = np.ones(T)
    m_hat[te + 1:] = np.clip(path[-(T - te - 1):] ** FORECAST_ELASTICITY, 0.4, 1.6)
    F_p = fit_forecast(Y_s / m_hat[None, :], daily, cfg, " mandi shock, price-aware", quiet=True)
    for q in cfg.quantiles:
        F_p[q] = F_p[q] * m_hat[None, :, None]                       # origin day's price level
    res["price_aware"] = _replay_pair(D_s, S_s, F_p, meta, cfg)
    res["forecast_elasticity"] = FORECAST_ELASTICITY
    weekly = agm[name]["price"].resample("W").mean().dropna()
    res.update({
        "commodity": name, "market": "Azadpur, Delhi (Agmarknet)",
        "shock_window": [str(start.date()), str(end.date())],
        "peak_price_vs_normal": round(float(idx.loc[start:end].max()), 2),
        "avg_price_vs_normal": round(peak, 2), "elasticity": PRICE_ELASTICITY,
        "demand_swing_pct": [round(100 * (float(mult_test.min()) - 1), 1), round(100 * (float(mult_test.max()) - 1), 1)],
        "series": {name2: {"weeks": [str(x.date()) for x in d["price"].resample("W").mean().dropna().index],
                           "price": d["price"].resample("W").mean().dropna().round(0).tolist()}
                   for name2, d in agm.items()},
    })
    print(f"[india] Agmarknet shock ({name} {res['shock_window'][0]}→{res['shock_window'][1]}, "
          f"avg {peak:.2f}x normal): waste change {-res['waste_reduction_pct']:+.1f}%, service {res['service_change_pts']:+.2f} pts")
    pa = res["price_aware"]
    print(f"[india]   with price-aware forecast: waste change {-pa['waste_reduction_pct']:+.1f}%, "
          f"service {pa['service_change_pts']:+.2f} pts (waste {pa['baseline']['waste_pct_of_received']}% -> "
          f"{pa['unwaste']['waste_pct_of_received']}%)")
    return res


# ----------------------------------------------------------------------------- generated Indian files
# Used only when the real files are missing. Shapes follow public patterns:
# Delhi climate by month (hot May-June, monsoon from about 28 June) and well-known
# mandi spikes (tomato Jul-Aug 2023, onion Oct-Nov 2023). Labelled "generated" everywhere.
GEN_DIR = Path("data/raw/india_generated")
_DELHI_TEMP = {1: 14, 2: 17, 3: 23, 4: 30, 5: 34, 6: 34, 7: 31, 8: 30, 9: 29, 10: 26, 11: 20, 12: 15}
_DELHI_RAIN_MM_MONTH = {1: 19, 2: 20, 3: 15, 4: 10, 5: 25, 6: 70, 7: 210, 8: 230, 9: 120, 10: 15, 11: 5, 12: 8}
_MANDI = {  # base ₹/quintal, seasonal swing, spikes (start, end, multiplier)
    "tomato": (1500, 0.25, [("2023-07-01", "2023-08-15", 3.5), ("2024-10-01", "2024-10-31", 2.0)]),
    "onion": (1800, 0.15, [("2023-10-15", "2023-11-30", 2.2), ("2024-09-01", "2024-10-15", 1.8)]),
    "potato": (1200, 0.10, [("2024-06-15", "2024-08-15", 1.5)]),
}


def generate_weather(folder: Path, year_dates: pd.DatetimeIndex, seed: int = 11) -> Path:
    rng = np.random.default_rng(seed)
    days = pd.date_range(f"{year_dates.min().year}-01-01", f"{year_dates.max().year}-12-31", freq="D")
    m = days.month.to_numpy()
    temp = np.array([_DELHI_TEMP[x] for x in m], float) + rng.normal(0, 1.8, len(days))
    p_rain = np.where((m >= 7) & (m <= 8), 0.45, np.where(m == 6, 0.15, np.where(m == 9, 0.25, 0.06)))
    monthly = np.array([_DELHI_RAIN_MM_MONTH[x] for x in m], float)
    wet = rng.random(len(days)) < p_rain
    rain = np.where(wet, rng.gamma(1.1, monthly / (30 * p_rain) / 1.1), 0.0)
    temp -= np.where(rain > 10, 3.0, 0.0)
    hum = np.clip(np.where((m >= 7) & (m <= 9), 75, np.where((m >= 4) & (m <= 6), 35, 55)) + rain * 0.5
                  + rng.normal(0, 6, len(days)), 10, 100)
    js = {"generated": True, "daily": {"time": [str(d.date()) for d in days],
          "temperature_2m_mean": temp.round(1).tolist(), "precipitation_sum": rain.round(1).tolist(),
          "relative_humidity_2m_mean": hum.round(0).tolist(),
          "wind_speed_10m_max": rng.uniform(8, 20, len(days)).round(1).tolist()}}
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "weather_generated.json"
    out.write_text(json.dumps(js))
    return out


def generate_mandi(folder: Path, seed: int = 12) -> None:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    folder.mkdir(parents=True, exist_ok=True)
    for name, (base, swing, spikes) in _MANDI.items():
        price = base * (1 + swing * np.sin(2 * np.pi * (days.day_of_year.to_numpy() - 60) / 365))
        mult = np.ones(len(days))
        for a, b, k in spikes:  # smooth ramp up and down around each spike
            mid, half = pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2, (pd.Timestamp(b) - pd.Timestamp(a)) / 2
            dist = np.abs((days - mid).days.to_numpy()) / max(half.days, 1)
            mult *= 1 + (k - 1) * np.clip(1.3 - dist, 0, 1)
        price = price * mult * np.exp(rng.normal(0, 0.06, len(days)))
        arrivals = 1000 / mult * np.exp(rng.normal(0, 0.15, len(days)))
        pd.DataFrame({"Arrival_Date": days.strftime("%d/%m/%Y"), "Modal_Price": price.round(0),
                      "Arrivals_Tonnes": arrivals.round(0)}).to_csv(folder / f"agmarknet_{name}.csv", index=False)


# ----------------------------------------------------------------------------- synthetic Indian files
def _write_fake_india(folder: Path, dates: pd.DatetimeIndex) -> None:
    rng = np.random.default_rng(3)
    folder.mkdir(parents=True, exist_ok=True)
    n = len(dates)
    temp = 30 + 8 * np.sin(np.linspace(0, 3, n)) + rng.normal(0, 1.5, n)
    rain = np.where(np.arange(n) > n * 0.75, rng.gamma(1.2, 12, n), rng.gamma(0.3, 2, n))
    js = {"daily": {"time": [str(d.date()) for d in dates], "temperature_2m_mean": temp.round(1).tolist(),
                    "precipitation_sum": rain.round(1).tolist(),
                    "relative_humidity_2m_mean": np.clip(40 + rain, 10, 100).round(0).tolist(),
                    "wind_speed_10m_mean": rng.uniform(5, 18, n).round(1).tolist()}}
    (folder / "weather_fake.json").write_text(json.dumps(js))
    days = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    base = 1500 + 300 * np.sin(np.arange(len(days)) / 58)
    spike = np.where((days >= "2023-07-01") & (days <= "2023-08-10"), 3.5, 1.0)
    pd.DataFrame({"Arrival_Date": days.strftime("%d/%m/%Y"), "Modal_Price": (base * spike).round(0)}) \
        .to_csv(folder / "agmarknet_tomato.csv", index=False)


# ----------------------------------------------------------------------------- run
def run(cfg: Config, india_dir: Path = INDIA_DIR) -> dict:
    t0 = time.time()
    panel = build_panel(cfg)
    _, demand = correct_demand(panel, cfg)
    panel.daily["demand"] = demand.to_numpy()
    Y, dates = to_matrix(panel.daily, "demand")
    Y_raw, _ = to_matrix(panel.daily, "sales")
    meta = panel.meta.sort_values("series_id").reset_index(drop=True)
    if cfg.use_synthetic:
        india_dir = Path("data/raw/_fake_india")          # stand-in files, git-ignored
        _write_fake_india(india_dir, dates)

    res = {"source": panel.source}
    res["weather_in_forecast"], F_w = weather_in_forecast(Y, panel.daily, cfg)

    # Indian inputs: real files if present, otherwise generated stand-ins (clearly labelled)
    wx_files = sorted(glob.glob(str(india_dir / "weather_*.json")))
    wx_src = "real" if wx_files and not cfg.use_synthetic else "generated"
    if not wx_files:
        wx_files = [str(generate_weather(GEN_DIR, pd.DatetimeIndex(dates)))]
        print(f"[india] no weather file in {india_dir}: using GENERATED Delhi-like weather")
    agm = load_agmarknet(india_dir)
    agm_src = "real" if agm and not cfg.use_synthetic else "generated"
    if not agm:
        generate_mandi(GEN_DIR)
        agm = load_agmarknet(GEN_DIR)
        print(f"[india] no agmarknet_*.csv in {india_dir}: using GENERATED mandi prices")
    res["inputs"] = {"weather": wx_src, "mandi": agm_src}

    res["indian_weather"] = indian_weather_whatif(Y, Y_raw, panel.daily, meta, dates, F_w,
                                                  load_open_meteo(Path(wx_files[0])), cfg)
    res["agmarknet"] = agmarknet_shock(Y, Y_raw, panel.daily, meta, agm, cfg, cfg.test_days)
    if wx_src == "generated":
        res["indian_weather"]["place"] = "Delhi-NCR (generated stand-in)"
    if agm_src == "generated":
        res["agmarknet"]["market"] = "Delhi mandi (generated stand-in)"
    res["runtime_s"] = round(time.time() - t0)
    out = Path(cfg.out_dir) / "india.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"[india] wrote {out} in {res['runtime_s']}s")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3: weather + Indian market data")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--stores", type=int, default=None)
    ap.add_argument("--products", type=int, default=None)
    ap.add_argument("--india-dir", default=str(INDIA_DIR))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = default_config(use_synthetic=a.synthetic, max_stores=a.stores, max_products=a.products, out_dir=a.out)
    run(cfg, Path(a.india_dir))


if __name__ == "__main__":
    main()
