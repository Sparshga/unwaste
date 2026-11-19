"""
Unwaste-India: the dataset we train and test on.

One row per store x item x day, Delhi-NCR, 10 dark stores, 32 perishable items,
1 Jan 2024 -> yesterday.

  REAL     weather: Open-Meteo archive for Gurugram (data/raw/india/weather_*.json);
                    days the files don't cover are filled from Delhi's monthly climate
                    and flagged in the `weather_source` column.
           festival calendar 2024-2026 (public dates, checked against two calendars).
  GENERATED demand, prices, promotions, stockouts. Patterned on FreshRetailNet:
           ~10 units a day per series on average, weekday shape, noisy counts,
           ~20% of days with an empty shelf for some hours. Weather effects follow
           what we measured on FreshRetailNet (heavy rain ~ +11%, warm days ~ +7%).
           Festival, price and promotion effects are stated assumptions (ITEMS, FEST_EFFECT).

`demand_true` is what customers wanted. A real store never sees it; we keep it only
to grade the stockout fill-in. Everything else is what a store's system would record.

Run:  python -m unwaste.india_data            -> data/raw/india_dataset/unwaste_india.csv
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/raw/india_dataset")
START = "2024-01-01"
OPEN_HOURS = 16                                   # 6:00-22:00
BASE_SCALE = 2.0                                  # a busy dark store: ~20 units per item per day
NOISE_SHAPE = 20.0                                # day-to-day randomness beyond Poisson (lower = noisier)
# share of a day's demand in each open hour (morning and evening peaks)
HOUR_PROFILE = np.array([3, 6, 8, 7, 5, 5, 5, 5, 5, 5, 6, 8, 10, 9, 7, 6], float)
HOUR_PROFILE = HOUR_PROFILE / HOUR_PROFILE.sum()

STORES = [  # name, size (x average demand)
    ("Gurugram Sector 45", 1.25), ("Gurugram DLF Phase 3", 1.10), ("Noida Sector 62", 1.00),
    ("Noida Sector 18", 0.90), ("Dwarka Sector 10", 1.05), ("Saket", 1.15),
    ("Lajpat Nagar", 0.95), ("Rohini Sector 9", 0.85), ("Indirapuram", 0.80), ("Karol Bagh", 0.75),
]

# item, category, shelf life (days, from typical pack labels), base price (₹), base units/day/store,
# season: (peak month, amplitude) or ("only", months available)
ITEMS = [
    ("Toned milk 500ml", "dairy", 3, 28, 30, (5, 0.05)), ("Full cream milk 500ml", "dairy", 3, 34, 18, (5, 0.05)),
    ("Curd 400g", "dairy", 7, 45, 12, (6, 0.25)), ("Paneer 200g", "dairy", 7, 90, 8, (11, 0.10)),
    ("Buttermilk 500ml", "dairy", 5, 20, 7, (6, 0.55)), ("Lassi 200ml", "dairy", 7, 25, 5, (6, 0.50)),
    ("White bread", "bakery", 4, 45, 12, (1, 0.05)), ("Brown bread", "bakery", 4, 55, 6, (1, 0.05)),
    ("Pav 6 pc", "bakery", 3, 35, 5, (8, 0.10)),
    ("Eggs 6 pc", "eggs_meat", 14, 48, 10, (1, 0.20)), ("Chicken curry cut 500g", "eggs_meat", 2, 180, 4, (1, 0.15)),
    ("Tomato 1kg", "vegetables", 5, 40, 14, (12, 0.10)), ("Onion 1kg", "vegetables", 10, 35, 15, (1, 0.05)),
    ("Potato 1kg", "vegetables", 14, 30, 14, (1, 0.05)), ("Spinach 250g", "vegetables", 2, 25, 6, (1, 0.45)),
    ("Coriander 100g", "vegetables", 2, 15, 11, (1, 0.15)), ("Green chilli 100g", "vegetables", 5, 10, 8, (7, 0.10)),
    ("Cucumber 500g", "vegetables", 4, 30, 7, (5, 0.45)), ("Capsicum 250g", "vegetables", 5, 30, 5, (1, 0.10)),
    ("Cauliflower 1 pc", "vegetables", 4, 40, 5, (12, 0.45)), ("Methi bunch", "vegetables", 2, 20, 4, (1, 0.55)),
    ("Carrot 500g", "vegetables", 7, 35, 5, (1, 0.55)), ("Lemon 250g", "vegetables", 10, 30, 6, (6, 0.40)),
    ("Ginger 100g", "vegetables", 14, 20, 6, (12, 0.15)),
    ("Banana 6 pc", "fruits", 4, 50, 10, (11, 0.10)), ("Apple 4 pc", "fruits", 14, 160, 5, (10, 0.20)),
    ("Papaya 1 pc", "fruits", 4, 60, 3, (3, 0.15)), ("Pomegranate 2 pc", "fruits", 10, 150, 3, (11, 0.20)),
    ("Watermelon 1 pc", "fruits", 7, 70, 4, ("only", (3, 4, 5, 6, 7))),
    ("Mango 1kg", "fruits", 5, 120, 6, ("only", (4, 5, 6, 7))),
    ("Orange 1kg", "fruits", 10, 80, 5, ("only", (11, 12, 1, 2))),
    ("Sweet corn 2 pc", "vegetables", 4, 40, 3, (8, 0.40)),
]

# festival calendar: date, name, type, size (1 = major)
FESTIVALS = [
    ("2024-01-14", "Makar Sankranti / Lohri", "harvest", 0.4), ("2024-01-26", "Republic Day", "national", 0.3),
    ("2024-03-25", "Holi", "holi", 1.0), ("2024-04-09", "Chaitra Navratri", "navratri", 0.6),
    ("2024-04-11", "Eid ul-Fitr", "eid", 1.0), ("2024-04-17", "Ram Navami", "puja", 0.6),
    ("2024-06-17", "Eid al-Adha (Bakrid)", "eid", 1.0), ("2024-08-15", "Independence Day", "national", 0.3),
    ("2024-08-19", "Raksha Bandhan", "sweets", 0.9), ("2024-08-26", "Janmashtami", "puja", 0.7),
    ("2024-10-03", "Sharad Navratri", "navratri", 0.8), ("2024-10-12", "Dussehra", "puja", 0.7),
    ("2024-10-20", "Karwa Chauth", "sweets", 0.5), ("2024-10-31", "Diwali", "diwali", 1.0),
    ("2024-11-03", "Bhai Dooj", "sweets", 0.5), ("2024-11-07", "Chhath Puja", "chhath", 0.8),
    ("2024-11-15", "Guru Nanak Jayanti", "puja", 0.4), ("2024-12-25", "Christmas", "christmas", 0.6),
    ("2024-12-31", "New Year's Eve", "christmas", 0.6),
    ("2025-01-14", "Makar Sankranti / Lohri", "harvest", 0.4), ("2025-01-26", "Republic Day", "national", 0.3),
    ("2025-03-14", "Holi", "holi", 1.0), ("2025-03-30", "Chaitra Navratri", "navratri", 0.6),
    ("2025-03-31", "Eid ul-Fitr", "eid", 1.0), ("2025-04-06", "Ram Navami", "puja", 0.6),
    ("2025-06-07", "Eid al-Adha (Bakrid)", "eid", 1.0), ("2025-08-09", "Raksha Bandhan", "sweets", 0.9),
    ("2025-08-15", "Independence Day", "national", 0.3), ("2025-08-16", "Janmashtami", "puja", 0.7),
    ("2025-09-22", "Sharad Navratri", "navratri", 0.8), ("2025-10-02", "Dussehra", "puja", 0.7),
    ("2025-10-10", "Karwa Chauth", "sweets", 0.5), ("2025-10-20", "Diwali", "diwali", 1.0),
    ("2025-10-23", "Bhai Dooj", "sweets", 0.5), ("2025-10-28", "Chhath Puja", "chhath", 0.8),
    ("2025-11-05", "Guru Nanak Jayanti", "puja", 0.4), ("2025-12-25", "Christmas", "christmas", 0.6),
    ("2025-12-31", "New Year's Eve", "christmas", 0.6),
    ("2026-01-14", "Makar Sankranti / Lohri", "harvest", 0.4), ("2026-01-26", "Republic Day", "national", 0.3),
    ("2026-03-04", "Holi", "holi", 1.0), ("2026-03-19", "Chaitra Navratri", "navratri", 0.6),
    ("2026-03-21", "Eid ul-Fitr", "eid", 1.0), ("2026-03-26", "Ram Navami", "puja", 0.6),
    ("2026-05-27", "Eid al-Adha (Bakrid)", "eid", 1.0), ("2026-08-15", "Independence Day", "national", 0.3),
    ("2026-08-28", "Raksha Bandhan", "sweets", 0.9), ("2026-09-04", "Janmashtami", "puja", 0.7),
    ("2026-10-11", "Sharad Navratri", "navratri", 0.8), ("2026-10-20", "Dussehra", "puja", 0.7),
    ("2026-10-28", "Karwa Chauth", "sweets", 0.5), ("2026-11-08", "Diwali", "diwali", 1.0),
    ("2026-11-11", "Bhai Dooj", "sweets", 0.5), ("2026-11-15", "Chhath Puja", "chhath", 0.8),
    ("2026-11-24", "Guru Nanak Jayanti", "puja", 0.4), ("2026-12-25", "Christmas", "christmas", 0.6),
    ("2026-12-31", "New Year's Eve", "christmas", 0.6),
]
FEST_TYPES = ["none", "holi", "eid", "diwali", "sweets", "navratri", "chhath", "puja", "christmas", "national", "harvest"]
# extra demand at the peak of each festival type, by category (stated assumption)
FEST_EFFECT = {
    "holi":      {"dairy": 0.45, "bakery": 0.10, "eggs_meat": 0.15, "vegetables": 0.10, "fruits": 0.10},
    "eid":       {"dairy": 0.40, "bakery": 0.15, "eggs_meat": 0.70, "vegetables": 0.20, "fruits": 0.25},
    "diwali":    {"dairy": 0.55, "bakery": 0.05, "eggs_meat": -0.15, "vegetables": 0.15, "fruits": 0.45},
    "sweets":    {"dairy": 0.35, "bakery": 0.05, "eggs_meat": 0.00, "vegetables": 0.05, "fruits": 0.25},
    "navratri":  {"dairy": 0.20, "bakery": -0.05, "eggs_meat": -0.45, "vegetables": 0.10, "fruits": 0.35},
    "chhath":    {"dairy": 0.20, "bakery": 0.00, "eggs_meat": -0.30, "vegetables": 0.15, "fruits": 0.70},
    "puja":      {"dairy": 0.25, "bakery": 0.00, "eggs_meat": -0.20, "vegetables": 0.05, "fruits": 0.30},
    "christmas": {"dairy": 0.10, "bakery": 0.45, "eggs_meat": 0.35, "vegetables": 0.05, "fruits": 0.10},
    "national":  {"dairy": 0.05, "bakery": 0.05, "eggs_meat": 0.10, "vegetables": 0.00, "fruits": 0.00},
    "harvest":   {"dairy": 0.10, "bakery": 0.00, "eggs_meat": 0.05, "vegetables": 0.15, "fruits": 0.05},
}
RAMP = {-3: 0.30, -2: 0.60, -1: 1.00, 0: 0.55, 1: -0.15}   # share of the peak effect by day relative to the festival
NAVRATRI_DAYS = 9                                          # fasting effect lasts nine days
PRICE_ELASTICITY = {"dairy": -0.2, "bakery": -0.3, "eggs_meat": -0.4, "vegetables": -0.5, "fruits": -0.5}
PROMO_UPLIFT_PER_10PCT = 0.18                              # same as the replay's markdown uplift
# produce price spikes (generated, shaped on public mandi patterns): item word, start, end, multiplier
PRICE_SPIKES = [("Tomato", "2024-07-01", "2024-08-15", 1.8), ("Tomato", "2024-10-01", "2024-11-05", 2.0),
                ("Tomato", "2025-06-20", "2025-07-31", 1.5), ("Tomato", "2026-07-01", "2026-08-10", 1.7),
                ("Onion", "2024-09-01", "2024-12-15", 1.6), ("Onion", "2025-10-01", "2025-11-20", 1.3),
                ("Potato", "2024-06-01", "2024-09-30", 1.3), ("Coriander", "2025-07-01", "2025-08-15", 1.8),
                ("Coriander", "2026-07-10", "2026-08-20", 1.7)]
MILK_STEPS = [("2024-06-03", 1.07), ("2025-05-01", 1.06)]  # milk price revisions (generated steps)
_DELHI_TEMP = {1: 14, 2: 17, 3: 23, 4: 30, 5: 34, 6: 34, 7: 31, 8: 30, 9: 29, 10: 26, 11: 20, 12: 15}
_DELHI_RAIN = {1: 19, 2: 20, 3: 15, 4: 10, 5: 25, 6: 70, 7: 210, 8: 230, 9: 120, 10: 15, 11: 5, 12: 8}


# ----------------------------------------------------------------------------- weather
def load_weather(dates: pd.DatetimeIndex, folder: Path = Path("data/raw/india"), seed: int = 5) -> pd.DataFrame:
    """Real Open-Meteo days where we have them, Delhi monthly climate for the rest (flagged)."""
    frames = []
    for f in sorted(glob.glob(str(folder / "weather_*.json"))):
        try:
            js = json.loads(Path(f).read_text())
            if js.get("generated") or "daily" not in js:
                continue
            d = js["daily"]
            wind_key = next(k for k in d if k.startswith("wind_speed"))
            frames.append(pd.DataFrame({
                "rain": d["precipitation_sum"], "temp": d["temperature_2m_mean"],
                "humidity": d.get("relative_humidity_2m_mean", [np.nan] * len(d["time"])),
                "wind_kmh": d[wind_key]}, index=pd.to_datetime(d["time"])))
        except Exception as e:                       # a broken file must not stop the run
            print(f"[india-data] skip weather file {f}: {e}")
    real = pd.concat(frames).astype(float) if frames else pd.DataFrame(columns=["rain", "temp", "humidity", "wind_kmh"])
    real = real[~real.index.duplicated(keep="last")].dropna(subset=["temp"])
    wx = real.reindex(dates)
    src = np.where(wx["temp"].notna(), "open-meteo", "climate fill")
    miss = wx["temp"].isna().to_numpy()
    if miss.any():
        rng = np.random.default_rng(seed)
        m = dates.month.to_numpy()
        clim_t = np.array([_DELHI_TEMP[x] for x in m], float) + rng.normal(0, 1.8, len(dates))
        p_rain = np.where((m >= 7) & (m <= 8), 0.45, np.where(m == 6, 0.15, np.where(m == 9, 0.25, 0.06)))
        mm = np.array([_DELHI_RAIN[x] for x in m], float)
        rain = np.where(rng.random(len(dates)) < p_rain, rng.gamma(1.1, mm / (30 * p_rain) / 1.1), 0.0)
        hum = np.clip(np.where((m >= 7) & (m <= 9), 75, np.where((m >= 4) & (m <= 6), 35, 60)) + rain * 0.5, 10, 100)
        fill = pd.DataFrame({"rain": rain, "temp": clim_t - np.where(rain > 10, 3, 0), "humidity": hum,
                             "wind_kmh": rng.uniform(8, 18, len(dates))}, index=dates)
        wx = wx.where(~pd.DataFrame(np.repeat(miss[:, None], 4, 1), index=dates, columns=wx.columns), fill)
    wx = wx[["rain", "temp", "humidity", "wind_kmh"]].astype(float)
    wx["humidity"] = wx["humidity"].fillna(55)
    wx["wind"] = np.round((wx["wind_kmh"] / 3.6 / 0.836) ** (2 / 3), 2)      # km/h -> Beaufort-like level
    wx["weather_source"] = src
    return wx


# ----------------------------------------------------------------------------- calendar
def festival_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Per day: next festival within 7 days (name, type, size, days to it) + effect ramp per type."""
    fest = [(pd.Timestamp(d), n, t, w) for d, n, t, w in FESTIVALS]
    rows = []
    for day in dates:
        nxt = [((d - day).days, n, t, w) for d, n, t, w in fest if 0 <= (d - day).days <= 7]
        k, name, typ, w = min(nxt) if nxt else (8, "", "none", 0.0)
        ramp = {}
        for d, n, t, w2 in fest:
            off = (day - d).days
            share = RAMP.get(off, 0.0)
            if t == "navratri" and 0 <= off < NAVRATRI_DAYS:
                share = 0.8                                            # fasting days
            if share:
                ramp[t] = ramp.get(t, 0.0) + share * w2
        rows.append({"date": day, "fest_name": name, "fest_type": typ, "fest_weight": w, "days_to_fest": k,
                     "holiday": float(any(d == day and w2 >= 0.5 for d, _, _, w2 in fest)), "ramp": ramp})
    return pd.DataFrame(rows).set_index("date")


def _season(month: np.ndarray, spec) -> np.ndarray:
    if spec[0] == "only":
        return np.where(np.isin(month, spec[1]), 1.0, 0.12)            # off season: a little imported stock
    peak, amp = spec
    return 1 + amp * np.cos(2 * np.pi * (month - peak) / 12)


def _weather_effect(cat: str, item: str, temp: np.ndarray, rain: np.ndarray) -> np.ndarray:
    """Rain: measured on FreshRetailNet (~ +11% on heavy-rain days, more orders when people stay in).
    Heat and cold: per category, extrapolated from the measured warm-day effect (~ +7%)."""
    eff = 1 + np.where(rain > 10, 0.11, np.where(rain > 1, 0.03, -0.01))
    hot = np.clip(temp - 30, 0, 15)
    cold = np.clip(18 - temp, 0, 15)
    cooling = any(w in item for w in ("Buttermilk", "Lassi", "Curd", "Watermelon", "Cucumber", "Lemon"))
    winter = any(w in item for w in ("Spinach", "Methi", "Carrot", "Orange", "Cauliflower", "Eggs"))
    snack = any(w in item for w in ("Onion", "Potato", "Green chilli", "Pav", "Sweet corn", "bread"))
    eff *= 1 + (0.025 * hot if cooling else (-0.006 * hot if cat in ("eggs_meat",) else 0.004 * hot))
    eff *= 1 + (0.02 * cold if winter else 0.0)
    eff *= 1 + np.where((rain > 10) & snack, 0.10, 0.0)               # rainy-day pakoras and chai
    return eff


# ----------------------------------------------------------------------------- generate
def generate(end: str | None = None, seed: int = 11, folder: Path = Path("data/raw/india")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = end or str((pd.Timestamp.now(tz="Asia/Kolkata").normalize() - pd.Timedelta(days=1)).date())
    dates = pd.date_range(START, end, freq="D")
    T = len(dates)
    wx = load_weather(dates, folder)
    cal = festival_features(dates)
    month, dow, dom = dates.month.to_numpy(), dates.dayofweek.to_numpy(), dates.day.to_numpy()
    years = (dates - dates[0]).days.to_numpy() / 365.0
    rows = []
    for si, (store, size) in enumerate(STORES):
        tilt = {c: rng.uniform(0.85, 1.2) for c in PRICE_ELASTICITY}           # neighbourhood taste
        fest_sens = rng.uniform(0.7, 1.3)
        for pi, (item, cat, shelf, base_price, base_units, season) in enumerate(ITEMS):
            # price: produce moves with the mandi, dairy has step revisions, the rest is flat
            pidx = np.ones(T)
            if cat in ("vegetables", "fruits"):
                walk = np.cumsum(rng.normal(0, 0.015, T))
                pidx = np.exp(walk - pd.Series(walk).rolling(60, min_periods=1).mean().to_numpy())
                for word, a, b, k in PRICE_SPIKES:
                    if word in item:
                        mid = pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2
                        half = max((pd.Timestamp(b) - pd.Timestamp(a)).days / 2, 1)
                        dist = np.abs((dates - mid).days.to_numpy()) / half
                        pidx *= 1 + (k - 1) * np.clip(1.25 - dist, 0, 1)
            if "milk" in item or "Curd" in item or "Paneer" in item:
                for d, k in MILK_STEPS:
                    pidx = np.where(dates >= pd.Timestamp(d), pidx * k, pidx)
            # promotions: short runs of 10-30% off; more often in the week before big festivals
            disc = np.zeros(T)
            p_start = 0.025 + 0.05 * (cal["days_to_fest"].to_numpy() <= 4) * (cal["fest_weight"].to_numpy() >= 0.8)
            starts = np.where(rng.random(T) < p_start)[0]
            for s0 in starts:
                disc[s0:s0 + rng.integers(2, 5)] = rng.choice([0.10, 0.15, 0.20, 0.25, 0.30])
            # demand
            mu = (BASE_SCALE * base_units * size * tilt[cat]
                  * (1 + 0.20 * years)                                          # quick-commerce growth
                  * _season(month, season)
                  * np.where(dow >= 5, 1.12 if cat != "dairy" else 1.05, 1.0)    # weekends
                  * np.where(dom <= 5, 1.06, np.where(dom >= 26, 0.97, 1.0))      # salary week
                  * _weather_effect(cat, item, wx["temp"].to_numpy(), wx["rain"].to_numpy())
                  * pidx ** PRICE_ELASTICITY[cat]
                  * (1 + PROMO_UPLIFT_PER_10PCT * disc / 0.10))
            fest_mult = np.ones(T)
            for t_i, ramp in enumerate(cal["ramp"].to_numpy()):
                for ftype, share in ramp.items():
                    fest_mult[t_i] *= 1 + share * fest_sens * FEST_EFFECT[ftype][cat]
            mu = mu * np.clip(fest_mult, 0.3, 3.0)
            lam = rng.gamma(NOISE_SHAPE, mu / NOISE_SHAPE)                       # day-to-day noise
            demand = rng.poisson(lam).astype(float)
            # stockouts: more likely when demand jumps above the recent level (surges empty shelves)
            recent = pd.Series(demand).rolling(14, min_periods=1).mean().shift(1).bfill().to_numpy()
            p_out = np.clip(0.14 + 0.35 * (demand > 1.3 * recent + 2), 0, 0.6)
            out_hours = np.where(rng.random(T) < p_out, rng.integers(1, 9, T), 0)
            tail = np.array([HOUR_PROFILE[OPEN_HOURS - k:].sum() if k else 0.0 for k in range(OPEN_HOURS + 1)])
            sales = np.floor(demand * (1 - tail[out_hours]) + rng.random(T) * 0.999)
            sales = np.minimum(sales, demand)
            rows.append(pd.DataFrame({
                "date": dates, "store_id": si, "store_name": store, "product_id": pi, "item": item,
                "category": cat, "shelf_life_days": shelf, "base_price": base_price,
                "price": np.round(base_price * pidx * (1 - disc), 1), "price_idx": np.round(pidx, 3),
                "discount": disc, "sales": sales, "stockout_hours": out_hours, "demand_true": demand}))
    df = pd.concat(rows, ignore_index=True)
    day = wx.join(cal.drop(columns=["ramp"]))
    df = df.merge(day[["rain", "temp", "humidity", "wind", "weather_source", "fest_name", "fest_type",
                       "fest_weight", "days_to_fest", "holiday"]], left_on="date", right_index=True)
    df["fest_type_code"] = df["fest_type"].map({t: i for i, t in enumerate(FEST_TYPES)}).astype(int)
    return df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Generate the Unwaste-India dataset")
    ap.add_argument("--end", default=None, help="last day (default: yesterday, India time)")
    a = ap.parse_args()
    df = generate(a.end)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "unwaste_india.csv", index=False)
    real = (df.drop_duplicates("date")["weather_source"] == "open-meteo").mean()
    print(f"[india-data] {len(df):,} rows = {df.store_id.nunique()} stores x {df.product_id.nunique()} items x "
          f"{df.date.nunique()} days ({df.date.min().date()} -> {df.date.max().date()})")
    print(f"[india-data] real weather on {real:.0%} of days; stockout days {(df.stockout_hours > 0).mean():.0%}; "
          f"sales hide {1 - df.sales.sum() / df.demand_true.sum():.1%} of demand")
    print(f"[india-data] wrote {OUT / 'unwaste_india.csv'}")


if __name__ == "__main__":
    main()
