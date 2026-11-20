"""
Step 3 · Forecast demand as a range (P10 / P50 / P90).

  "One global LightGBM model for every store and item. Standing on any night, it
   predicts each of the next 7 days, and instead of one number it gives a low,
   middle and high case. Inputs are things we'd know that night: recent demand,
   same-weekday history, day of week, planned discounts and holidays. We train only
   on the first weeks and forecast the last 4 weeks, so the model never sees the
   future (time split)."

Input : demand matrix [series, days] (corrected, or raw for the ablation)
Output: dict {0.1: F10, 0.5: F50, 0.9: F90}, each [series, origin_day, horizon]
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .backends import quantile_model
from .config import Config


def to_matrix(daily: pd.DataFrame, value: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    m = daily.pivot_table(index="series_id", columns="dt", values=value, aggfunc="sum")
    m = m.sort_index().sort_index(axis=1).fillna(0.0)
    return m.to_numpy(), pd.DatetimeIndex(m.columns)


WEATHER = ["rain", "temp", "humidity", "wind"]
FEST = ["days_to_fest", "fest_weight"]   # from the public festival calendar, known weeks ahead
INDIA = ["fest_type_code", "price_idx"]  # festival kind + the item's own price vs normal (Unwaste-India)
CAL = ["month", "dom"]                   # month and day of month of the target day (yearly data only)
WX_CHANGE = ["temp_delta", "cold_delta", "heat_delta"]   # the target day vs the 14 days before it
WX_ITEM = ["cold_x", "heat_x"]           # the same change times how strongly this item reacted in training


def temp_change(temps) -> dict:
    """Per day: temperature, heating degrees (below 18 C) and cooling degrees (above 24 C), each minus
    its average over the previous 14 days. Demand is forecast relative to the last 14 days of sales,
    so the model needs to know how the weather differs from those days, not just today's weather."""
    t = np.asarray(temps, float)
    out = {}
    for name, v in (("temp_delta", t), ("cold_delta", np.clip(18 - t, 0, None)), ("heat_delta", np.clip(t - 24, 0, None))):
        prev = pd.Series(v).shift(1).rolling(14, min_periods=1).mean().to_numpy()
        out[name] = np.nan_to_num(v - prev)
    return out


def exo_matrices(daily: pd.DataFrame) -> dict:
    out = {}
    for c in ["discount", "holiday", "activity"] + [w for w in WEATHER + FEST + INDIA + WX_CHANGE + WX_ITEM if w in daily.columns]:
        out[c] = daily.pivot_table(index="series_id", columns="dt", values=c, aggfunc="mean") \
            .sort_index().sort_index(axis=1).fillna(0).to_numpy()
    return out


def _origin_features(Y: np.ndarray, o: int) -> dict:
    """Features known on the night of day o (uses days <= o only)."""
    # nan-aware: a "new store" has NaN before it opened, so averages use only the days it has
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        def mean_last(k):
            return np.nanmean(Y[:, max(0, o - k + 1):o + 1], axis=1)
        return {"last": Y[:, o], "m7": mean_last(7), "m14": mean_last(14), "m28": mean_last(28),
                "sd14": np.nanstd(Y[:, max(0, o - 13):o + 1], axis=1)}


def _build_rows(Y, exo, dates, origins, H, series_codes, exo_future=None, calendar=False):
    """exo_future (optional): {column: [S, H]} values for the days after the data (live prediction)."""
    S, T = Y.shape
    blocks = []
    for o in origins:
        base = _origin_features(Y, o)
        for h in range(1, H + 1):
            t = o + h
            tt = min(t, T - 1)                      # beyond the data: reuse last known exo
            dow = (dates[o].day_of_week + h) % 7
            same_dow = [Y[:, t - 7 * k] for k in (1, 2, 3, 4) if 0 <= t - 7 * k <= o]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sd = np.nanmean(same_dow, axis=0) if same_dow else base["m7"]
            sd = np.where(np.isnan(sd), base["m7"], sd)
            ex = (lambda c: exo_future[c][:, t - T]) if (exo_future is not None and t >= T) \
                else (lambda c: exo[c][:, tt])
            blk = {**base, "h": np.full(S, h), "dow": np.full(S, dow),
                   "same_dow": sd,
                   "discount": ex("discount"), "holiday": ex("holiday"),
                   "activity": ex("activity"),
                   **{w: ex(w) for w in WEATHER + FEST + INDIA + WX_CHANGE + WX_ITEM if w in exo},
                   "store": series_codes[0], "product": series_codes[1],
                   "series": np.arange(S), "origin": np.full(S, o),
                   "target": Y[:, t] if t < T else np.full(S, np.nan)}
            if calendar:
                day = dates[o] + pd.Timedelta(days=h)
                blk["month"], blk["dom"] = np.full(S, day.month), np.full(S, day.day)
            blocks.append(pd.DataFrame(blk))
    return pd.concat(blocks, ignore_index=True)


BASE_FEATS = ["last", "m7", "m14", "m28", "sd14", "h", "dow", "same_dow", "discount", "holiday",
              "activity", "store", "product"]
FEATS = BASE_FEATS + WEATHER          # weather of the target day (from a weather forecast in production)


def fit_forecast(Y: np.ndarray, daily: pd.DataFrame, cfg: Config, label: str = "",
                 train_mask: np.ndarray | None = None, quiet: bool = False,
                 weather_override: dict | None = None) -> dict:
    """Train on the first (T - test_days) days, return forecasts for every origin in the test.
    train_mask (bool per series): Phase 2 tests train on some series and forecast the others."""
    S, T = Y.shape
    H = cfg.horizon_days
    _, dates = to_matrix(daily, "sales")
    exo = exo_matrices(daily)
    meta = daily.groupby("series_id")[["store_id", "product_id"]].first().sort_index()
    codes = (meta["store_id"].astype("category").cat.codes.to_numpy(),
             meta["product_id"].astype("category").cat.codes.to_numpy())
    train_end = T - cfg.test_days - 1                     # last training day index
    step = getattr(cfg, "train_origin_step", 1)
    train_origins = [o for o in range(14, train_end, step) if o + 1 <= train_end]
    cal = getattr(cfg, "calendar_feats", False)
    train = _build_rows(Y, exo, dates, train_origins, H, codes, calendar=cal)
    train = train[train["origin"] + train["h"] <= train_end]
    train = train.dropna(subset=["target"])
    if train_mask is not None:
        train = train[train_mask[train["series"].to_numpy()]]
    test_origins = list(range(train_end, T))
    exo_test = exo
    if weather_override:                       # what-if: same stores, different weather in the test weeks
        exo_test = {k: v.copy() for k, v in exo.items()}
        for k, vec in weather_override.items():
            exo_test[k][:, train_end + 1:] = np.asarray(vec)[None, train_end + 1:]
    test = _build_rows(Y, exo_test, dates, test_origins, H, codes, calendar=cal)
    feats = FEATS if (cfg.use_weather and all(w in train.columns for w in WEATHER)) else BASE_FEATS
    if cfg.use_festivals and all(f in train.columns for f in FEST):
        feats = feats + FEST
        if "fest_type_code" in train.columns:
            feats = feats + ["fest_type_code"]
    if "price_idx" in train.columns:
        feats = feats + ["price_idx"]
    if cfg.use_weather and all(c in train.columns for c in WX_CHANGE):
        feats = feats + WX_CHANGE
    if cfg.use_weather and all(c in train.columns for c in WX_ITEM):
        feats = feats + WX_ITEM
    if cal:
        feats = feats + CAL
    if cfg.relative_target:
        # predict demand RELATIVE to the item's recent level (last 14 days), then scale back.
        # Trees can't extrapolate, so this lets forecasts follow sudden level shifts
        # (price shocks, new stores) instead of anchoring to the levels seen in training.
        for df in (train, test):
            df["level"] = df["m14"].fillna(df["m7"]).fillna(0) + 1.0
            for c in ["last", "m7", "m14", "m28", "sd14", "same_dow"]:
                df[c] = df[c] / df["level"]
        train["target"] = train["target"] / train["level"]
        feats = feats + ["level"]

    out, models = {}, {}
    for q in cfg.quantiles:
        m = quantile_model(q, seed=cfg.seed, n_estimators=cfg.n_estimators)
        m.fit(train[feats].to_numpy(), train["target"].to_numpy())
        models[q] = m
        pred = np.clip(m.predict(test[feats].to_numpy()), 0, None)
        if cfg.relative_target:
            pred = pred * test["level"].to_numpy()
        F = np.full((S, T, H), np.nan)
        F[test["series"].to_numpy(), test["origin"].to_numpy(), test["h"].to_numpy() - 1] = pred
        out[q] = F
    # keep quantiles ordered: P10 <= P50 <= P90
    out[0.1] = np.minimum(out[0.1], out[0.5])
    out[0.9] = np.maximum(out[0.9], out[0.5])

    # quick accuracy report on the test window (horizon 1); target column is still in units
    tgt = test.dropna(subset=["target"])
    sel = tgt[tgt["h"] == 1]
    p50 = out[0.5][sel["series"], sel["origin"], 0]
    wape = np.abs(p50 - sel["target"]).sum() / max(sel["target"].sum(), 1e-9)
    inside = ((sel["target"] >= out[0.1][sel["series"], sel["origin"], 0]) &
              (sel["target"] <= out[0.9][sel["series"], sel["origin"], 0])).mean()
    if not quiet:
        print(f"[forecast{label}] train rows {len(train):,}; next-day WAPE {wape:.0%}; "
              f"actuals inside P10-P90: {inside:.0%} (ideal ~80%)")
    out["train_end"] = train_end
    out["features"] = feats
    out["models"], out["codes"], out["calendar"] = models, codes, cal
    out["metrics"] = {"wape_h1": float(wape), "coverage_p10_p90": float(inside)}
    return out
