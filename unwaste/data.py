"""
Step 1 · Clean & prepare.

  "We load FreshRetailNet-50K: one row per store, item and day, with 24 hourly
   sales numbers and 24 hourly stock flags. We keep one city to make the demo fast,
   mark every hour where the shelf was empty, and rescale the anonymised sales to
   realistic units. If the real files are missing, we can generate stand-in data
   with exactly the same columns, clearly labelled synthetic."

Input : parquet files in cfg.data_dir   (or synthetic)
Output: Panel(daily=DataFrame, hourly_sales=array, hourly_oos=array, meta=DataFrame)
        daily has one row per (series, day); the two arrays are [rows, 24].

"""
from __future__ import annotations

import glob
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config

# Column names in FreshRetailNet-50K. If the real file differs, change them here only.
COL = dict(
    city="city_id", store="store_id", product="product_id", cat="first_category_id",
    date="dt", sale="sale_amount", hours_sale="hours_sale", hours_stock="hours_stock_status",
    discount="discount", holiday="holiday_flag", activity="activity_flag",
    rain="precpt", temp="avg_temperature", humidity="avg_humidity", wind="avg_wind_level",
)
OPTIONAL = ["discount", "holiday", "activity", "rain", "temp", "humidity", "wind", "cat"]


@dataclass
class Panel:
    daily: pd.DataFrame          # one row per series-day, sorted by series_id, dt
    hourly_sales: np.ndarray     # [rows, 24] scaled units
    hourly_oos: np.ndarray       # [rows, 24] True = shelf empty that hour
    meta: pd.DataFrame           # one row per series: store, product, price, shelf life
    source: str                  # "FreshRetailNet-50K" or "SYNTHETIC"


# ----------------------------------------------------------------------------- helpers
def _stable_uniform(key: str) -> float:
    """Deterministic number in [0,1) from a string, so prices/shelf lives never change."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _to_matrix(col: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(x, dtype=float) for x in col.to_numpy()])


# ----------------------------------------------------------------------------- real data
def load_frn(cfg: Config) -> pd.DataFrame:
    files = sorted(glob.glob(str(cfg.data_dir / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No parquet files under {cfg.data_dir}. Download first (see README), "
            "or run with --synthetic.")
    # 1) choose the city and stores cheaply, reading only id columns
    ids = pd.concat([pd.read_parquet(f, columns=[COL["city"], COL["store"], COL["sale"]])
                     for f in files])
    if cfg.city_id is None:
        city = ids.groupby(COL["city"])[COL["store"]].nunique().idxmax()
    else:
        city = cfg.city_id
    stores = (ids[ids[COL["city"]] == city].groupby(COL["store"])[COL["sale"]].sum()
              .nlargest(cfg.max_stores).index.tolist())
    # 2) read the full rows for that city only
    parts = []
    for f in files:
        d = pd.read_parquet(f, filters=[(COL["city"], "==", city)])
        parts.append(d[d[COL["store"]].isin(stores)])
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=[COL["store"], COL["product"], COL["date"]])
    print(f"[data] FreshRetailNet-50K: city {city}, {len(stores)} stores, {len(df):,} rows")
    return df


def load_frn_multi(cfg: Config, stores_per_city: int = 4, products_per_city: int = 20) -> pd.DataFrame:
    """Phase 2: a few top stores from EVERY city (for the unseen-cities test).
    Reads only the chosen stores from disk (pyarrow filter), so memory stays small."""
    files = sorted(glob.glob(str(cfg.data_dir / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files under {cfg.data_dir}.")
    ids = pd.concat([pd.read_parquet(f, columns=[COL["city"], COL["store"], COL["product"], COL["sale"]])
                     for f in files])
    by_store = ids.groupby([COL["city"], COL["store"]], as_index=False)[COL["sale"]].sum()
    keep = by_store.sort_values(COL["sale"], ascending=False).groupby(COL["city"]).head(stores_per_city)
    stores = keep[COL["store"]].astype(int).tolist()
    sub = ids.merge(keep[[COL["city"], COL["store"]]], on=[COL["city"], COL["store"]])
    prods = (sub.groupby([COL["city"], COL["product"]], as_index=False)[COL["sale"]].sum()
             .sort_values(COL["sale"], ascending=False).groupby(COL["city"]).head(products_per_city))
    parts = []
    for f in files:
        d = pd.read_parquet(f, filters=[(COL["store"], "in", stores)])
        d = d.merge(prods[[COL["city"], COL["product"]]], on=[COL["city"], COL["product"]])
        parts.append(d)
    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=[COL["store"], COL["product"], COL["date"]])
    print(f"[data] FreshRetailNet-50K multi-city: {df[COL['city']].nunique()} cities, "
          f"{len(stores)} stores, {len(df):,} rows")
    return df


# ----------------------------------------------------------------------------- synthetic
def make_synthetic(cfg: Config, n_days: int = 90, n_cities: int = 1) -> pd.DataFrame:
    """Stand-in data with the SAME columns as FreshRetailNet-50K (for dev/testing only)."""
    rng = np.random.default_rng(cfg.seed)
    S, P = cfg.max_stores, cfg.max_products
    dates = pd.date_range("2024-03-28", periods=n_days, freq="D")
    hour_profile = np.array([0, 0, 0, 0, 0, 0, .02, .04, .07, .07, .06, .06, .07, .06,
                             .05, .05, .06, .08, .1, .09, .06, .04, .02, 0.0])
    hour_profile /= hour_profile.sum()
    dow = np.array([0.95, 0.92, 0.95, 1.0, 1.08, 1.2, 1.15])
    rows = []
    store_eff = rng.lognormal(0, 0.35, S)
    prod_eff = rng.lognormal(0, 0.6, P)
    cat = rng.integers(1, 6, P)
    for s in range(S):
        for p in range(P):
            lam_base = 0.9 * store_eff[s] * prod_eff[p]
            for di, d in enumerate(dates):
                disc = 1.0 if rng.random() > 0.12 else rng.choice([0.85, 0.9])
                hol = int(d.day_of_week >= 5)
                rain = float(rng.gamma(0.5, 2)) * (3 if rng.random() < 0.15 else 1)
                temp = float(rng.normal(27, 5))
                # weather moves demand: heavy rain keeps shoppers in (more orders), heat lifts dairy/drinks
                wx = (1 + 0.25 * (rain > 5)) * (1 + 0.02 * max(temp - 30, 0))
                lam = lam_base * dow[d.day_of_week] * (1 + 1.8 * (1 - disc)) * wx
                hourly_lam = lam * hour_profile * 10
                demand = rng.poisson(hourly_lam)
                stock = rng.poisson(max(lam * 10 * rng.uniform(0.7, 1.6), 1))
                sold = np.zeros(24)
                oos = np.zeros(24, dtype=int)
                left = stock
                for h in range(24):
                    take = min(left, demand[h])
                    sold[h] = take
                    left -= take
                    if left <= 0 and 6 <= h <= 22:
                        oos[h] = 1
                scale = 0.1  # mimic anonymised, scaled sales
                rows.append({
                    COL["city"]: s % n_cities, COL["store"]: s, COL["product"]: p, COL["cat"]: int(cat[p]),
                    COL["date"]: d.strftime("%Y-%m-%d"), COL["sale"]: float(sold.sum() * scale),
                    COL["hours_sale"]: (sold * scale).tolist(), COL["hours_stock"]: oos.tolist(),
                    COL["discount"]: float(disc), COL["holiday"]: hol, COL["activity"]: int(disc < 1),
                    COL["rain"]: rain, COL["temp"]: temp,
                    COL["humidity"]: float(np.clip(60 + 3 * rain + rng.normal(0, 8), 20, 100)),
                    COL["wind"]: float(abs(rng.normal(2, 1))),
                })
    print(f"[data] SYNTHETIC stand-in: {S} stores x {P} products x {n_days} days")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- build panel
def build_panel(cfg: Config, raw: pd.DataFrame | None = None) -> Panel:
    if raw is None:
        raw = make_synthetic(cfg) if cfg.use_synthetic else load_frn(cfg)
        # keep the top-selling products across the chosen stores
        top = raw.groupby(COL["product"])[COL["sale"]].sum().nlargest(cfg.max_products).index
        raw = raw[raw[COL["product"]].isin(top)].copy()
    source = "SYNTHETIC" if cfg.use_synthetic else "FreshRetailNet-50K"
    raw = raw.copy()
    raw["dt"] = pd.to_datetime(raw[COL["date"]])
    raw = raw.sort_values([COL["store"], COL["product"], "dt"]).reset_index(drop=True)

    hs = _to_matrix(raw[COL["hours_sale"]])
    st = _to_matrix(raw[COL["hours_stock"]])
    # Is "1" = out of stock? Check: flagged hours must sell less than unflagged hours.
    flag1 = st == 1
    oos = flag1 if hs[flag1].mean() <= hs[~flag1].mean() else ~flag1
    lo, hi = cfg.open_hours
    closed = np.ones(24, dtype=bool)
    closed[lo:hi + 1] = False
    oos[:, closed] = False                       # nights are closed, not stockouts

    # rescale anonymised sales so an average series sells ~target units a day
    series_mean = raw.groupby([COL["store"], COL["product"]])[COL["sale"]].mean()
    k = cfg.target_mean_daily_units / max(series_mean.mean(), 1e-9)
    hs = hs * k

    daily = pd.DataFrame({
        "city_id": raw[COL["city"]].to_numpy(),
        "store_id": raw[COL["store"]].to_numpy(),
        "product_id": raw[COL["product"]].to_numpy(),
        "dt": raw["dt"].to_numpy(),
        "sales": hs.sum(axis=1),
        "stockout_hours": oos.sum(axis=1),
    })
    for key in OPTIONAL:
        c = COL[key]
        daily[key] = raw[c].to_numpy() if c in raw.columns else 0.0
    daily["series_id"] = daily.groupby(["store_id", "product_id"]).ngroup()

    # per-series facts the data doesn't carry: price and shelf life (deterministic)
    meta = daily.groupby("series_id").agg(city_id=("city_id", "first"),
                                          store_id=("store_id", "first"),
                                          product_id=("product_id", "first"),
                                          cat=("cat", "first"),
                                          mean_sales=("sales", "mean")).reset_index()
    u_price = meta["product_id"].map(lambda p: _stable_uniform(f"price-{p}"))
    meta["price"] = (cfg.price_min + u_price * (cfg.price_max - cfg.price_min)).round(0)
    u_sl = meta["product_id"].map(lambda p: _stable_uniform(f"shelf-{p}"))
    choices = np.array(cfg.shelf_life_choices)
    meta["shelf_life"] = choices[(u_sl * len(choices)).astype(int).clip(0, len(choices) - 1)]
    meta["min_remaining"] = np.ceil(cfg.fssai_min_remaining_frac * meta["shelf_life"]).astype(int)
    meta["sellable_days"] = meta["shelf_life"] - meta["min_remaining"]

    print(f"[data] {meta.shape[0]} series, {daily['dt'].nunique()} days, "
          f"stockout share of open hours = {oos.sum() / max((~closed).sum() * len(daily), 1):.1%}")
    return Panel(daily=daily, hourly_sales=hs, hourly_oos=oos, meta=meta, source=source)
