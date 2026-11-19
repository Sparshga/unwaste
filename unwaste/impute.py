"""
Step 2 · Fill in demand hidden by empty shelves (stockout correction).

  "When a shelf is empty, sales show zero even though customers wanted the item.
   We train a gradient-boosting model ONLY on hours when the shelf had stock, so it
   learns real demand. Then we ask it what the empty hours would have sold, and
   replace those fake-low numbers. We prove it works with the demand-truth test:
   hide hours where we know the answer, and check the model gets them back."

Input : Panel (hourly sales + stockout flags)
Output: corrected hourly demand [rows, 24] and corrected daily demand
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backends import mean_model
from .config import Config
from .data import Panel

FEATURES = ["series_hour_mean", "series_mean", "hour", "dow", "discount",
            "holiday", "activity", "rain", "temp", "store_code", "product_code"]


def _hourly_frame(panel: Panel, cfg: Config) -> pd.DataFrame:
    d = panel.daily
    n = len(d)
    lo, hi = cfg.open_hours
    hours = np.arange(lo, hi + 1)
    rows = np.repeat(np.arange(n), len(hours))
    hr = np.tile(hours, n)
    f = pd.DataFrame({
        "row": rows, "hour": hr,
        "sales": panel.hourly_sales[rows, hr],
        "oos": panel.hourly_oos[rows, hr],
        "series_id": d["series_id"].to_numpy()[rows],
        "dow": pd.DatetimeIndex(d["dt"]).day_of_week.to_numpy()[rows],
    })
    for c in ["discount", "holiday", "activity", "rain", "temp"]:
        f[c] = d[c].to_numpy()[rows]
    f["store_code"] = d["store_id"].astype("category").cat.codes.to_numpy()[rows]
    f["product_code"] = d["product_id"].astype("category").cat.codes.to_numpy()[rows]
    return f


def _add_profiles(f: pd.DataFrame, fit_mask: np.ndarray) -> pd.DataFrame:
    """Average in-stock sales per series-hour and per series (learned from fit rows only)."""
    base = f[fit_mask]
    sh = base.groupby(["series_id", "hour"])["sales"].mean().rename("series_hour_mean")
    sm = base.groupby("series_id")["sales"].mean().rename("series_mean")
    f = f.join(sh, on=["series_id", "hour"]).join(sm, on="series_id")
    f["series_hour_mean"] = f["series_hour_mean"].fillna(f["series_mean"]).fillna(0)
    f["series_mean"] = f["series_mean"].fillna(0)
    return f


def _fit_predict(f: pd.DataFrame, fit_mask: np.ndarray, pred_mask: np.ndarray,
                 cfg: Config, max_rows: int = 400_000) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    idx = np.flatnonzero(fit_mask)
    if len(idx) > max_rows:
        idx = rng.choice(idx, max_rows, replace=False)
    model = mean_model(seed=cfg.seed, n_estimators=cfg.n_estimators)
    model.fit(f.loc[idx, FEATURES].to_numpy(), f.loc[idx, "sales"].to_numpy())
    pred = np.zeros(len(f))
    pidx = np.flatnonzero(pred_mask)
    if len(pidx):
        pred[pidx] = np.clip(model.predict(f.loc[pidx, FEATURES].to_numpy()), 0, None)
    return pred


def correct_demand(panel: Panel, cfg: Config) -> tuple[np.ndarray, pd.Series]:
    """Return corrected hourly demand [rows,24] and corrected daily demand (per daily row)."""
    f = _hourly_frame(panel, cfg)
    fit = ~f["oos"].to_numpy()
    f = _add_profiles(f, fit)
    pred = _fit_predict(f, fit, f["oos"].to_numpy(), cfg)
    corrected = panel.hourly_sales.copy()
    oos_rows = f["oos"].to_numpy()
    r, h = f.loc[oos_rows, "row"].to_numpy(), f.loc[oos_rows, "hour"].to_numpy()
    # never go below what was actually sold in that hour
    corrected[r, h] = np.maximum(pred[oos_rows], panel.hourly_sales[r, h])
    daily = pd.Series(corrected.sum(axis=1), index=panel.daily.index, name="demand")
    lift = daily.sum() / max(panel.daily["sales"].sum(), 1e-9) - 1
    print(f"[fill-in] corrected {oos_rows.sum():,} empty-shelf hours; "
          f"total demand is {lift:+.1%} vs raw sales")
    return corrected, daily


def demand_truth_test(panel: Panel, cfg: Config, frac_days: float = 0.08) -> dict:
    """
    Fake stockouts where we KNOW the answer, then check who recovers the hidden demand.
    We take full-stock days, hide every hour from a random afternoon hour to closing
    (exactly how a real stockout looks), refit the fill-in model without those hours,
    and compare its estimate of the hidden block with the truth.
    """
    rng = np.random.default_rng(cfg.seed + 1)
    f = _hourly_frame(panel, cfg)
    lo, hi = cfg.open_hours
    full_days = np.flatnonzero((panel.hourly_oos.sum(axis=1) == 0) &
                               (panel.hourly_sales.sum(axis=1) > 0))
    pick = rng.choice(full_days, max(1, int(frac_days * len(full_days))), replace=False)
    cut = dict(zip(pick, rng.integers(14, hi - 1, len(pick))))      # stockout starts here
    rows, hours = f["row"].to_numpy(), f["hour"].to_numpy()
    cut_for_row = np.array([cut.get(r, 99) for r in rows])
    hide = hours >= cut_for_row
    fit = ~f["oos"].to_numpy() & ~hide
    f = _add_profiles(f, fit)
    pred = _fit_predict(f, fit, hide, cfg)
    blocks = pd.DataFrame({"row": rows[hide], "truth": f.loc[hide, "sales"].to_numpy(),
                           "model": pred[hide],
                           "average": f.loc[hide, "series_hour_mean"].to_numpy()}).groupby("row").sum()
    truth = blocks["truth"].to_numpy()

    def wape(est):
        return float(np.abs(est - truth).sum() / max(truth.sum(), 1e-9))

    res = {
        "hidden_blocks": int(len(blocks)), "hidden_hours": int(hide.sum()),
        "wape_naive_zero": wape(np.zeros_like(truth)),
        "wape_series_average": wape(blocks["average"].to_numpy()),
        "wape_model": wape(blocks["model"].to_numpy()),
        "recovered_pct_model": float(100 * blocks["model"].sum() / max(truth.sum(), 1e-9)),
        "recovered_pct_naive": 0.0,
    }
    s = blocks.sample(min(30, len(blocks)), random_state=cfg.seed).sort_values("truth")
    res["sample"] = {"truth": s["truth"].round(2).tolist(), "model": s["model"].round(2).tolist()}
    print(f"[demand-truth] {len(blocks)} hidden afternoon blocks. Error (WAPE): naive zeros "
          f"{res['wape_naive_zero']:.0%}, series average {res['wape_series_average']:.0%}, "
          f"model {res['wape_model']:.0%}; model recovers {res['recovered_pct_model']:.0f}% of hidden demand")
    return res


def example_day(panel: Panel, corrected: np.ndarray) -> dict:
    """Pick one real day with a stockout to show observed vs corrected hourly demand."""
    oos_counts = panel.hourly_oos.sum(axis=1)
    sales = panel.hourly_sales.sum(axis=1)
    cand = np.flatnonzero((oos_counts >= 3) & (oos_counts <= 10))
    if len(cand) == 0:
        cand = np.flatnonzero(oos_counts > 0)
    if len(cand) == 0:
        return {}
    r = int(cand[np.argmax(sales[cand])])
    d = panel.daily.iloc[r]
    return {"store_id": str(d["store_id"]), "product_id": str(d["product_id"]),
            "date": str(pd.Timestamp(d["dt"]).date()),
            "observed": panel.hourly_sales[r].round(2).tolist(),
            "corrected": corrected[r].round(2).tolist(),
            "stockout": panel.hourly_oos[r].astype(int).tolist()}
