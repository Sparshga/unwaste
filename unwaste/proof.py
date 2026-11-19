"""
Phase 2 · Proof tests: the checks we promised on the engineering slide.

  "Four tests that try to break our own model.
   1. Forecast baselines: is LightGBM actually better than a 7-day average or
      'same as last week'?
   2. New stores: a store that opened 14 days ago, never seen in training. Does the
      global model still forecast it?
   3. Unseen cities: train on 14 cities, forecast 4 cities the model has never seen.
   4. Robustness: change our assumptions (lead time, shelf life, discount response,
      lost-sale cost) and rerun the replay. Does Unwaste still beat the baseline?"

Run:  python -m unwaste.proof              (real data, ~3-5 min)
      python -m unwaste.proof --synthetic  (stand-in data, for testing)
Output: web/data/proof.json  (shown in the 'Proof tests' tab)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import Config, default_config
from .data import build_panel, load_frn_multi, make_synthetic
from .forecast import fit_forecast, to_matrix
from .impute import correct_demand
from .replay import POLICIES, run_policy


# ----------------------------------------------------------------------------- metrics
def forecast_errors(P50: np.ndarray, Y: np.ndarray, train_end: int, H: int,
                    series: np.ndarray | None = None) -> dict:
    """WAPE of a P50 forecast [S, T, H] over the test window, next-day and full week."""
    S, T = Y.shape
    idx = np.arange(S) if series is None else np.asarray(series)
    out = {}
    for name, hs in (("next_day", [1]), ("week", list(range(1, H + 1)))):
        err = tot = 0.0
        for h in hs:
            for o in range(train_end, T - h):
                p = P50[idx, o, h - 1]
                a = Y[idx, o + h]
                ok = ~np.isnan(p) & ~np.isnan(a)
                err += np.abs(p[ok] - a[ok]).sum()
                tot += a[ok].sum()
        out[name] = float(round(100 * err / max(tot, 1e-9), 1))
    return out


def baseline_forecasts(Y: np.ndarray, train_end: int, H: int) -> dict:
    """7-day moving average and 'same weekday last week' as P50 forecasts."""
    S, T = Y.shape
    ma = np.full((S, T, H), np.nan)
    sn = np.full((S, T, H), np.nan)
    for o in range(train_end, T):
        m7 = Y[:, max(0, o - 6):o + 1].mean(axis=1)
        for h in range(1, H + 1):
            ma[:, o, h - 1] = m7
            src = o + h - 7
            while src > o:
                src -= 7
            sn[:, o, h - 1] = Y[:, src]
    return {"Moving average (7 days)": ma, "Same weekday last week": sn}


# ----------------------------------------------------------------------------- tests
def test_baselines(Y, F, cfg) -> dict:
    te, H = F["train_end"], cfg.horizon_days
    rows = [{"model": k, **forecast_errors(v, Y, te, H)} for k, v in baseline_forecasts(Y, te, H).items()]
    rows.append({"model": "LightGBM P50 (ours)", **forecast_errors(F[0.5], Y, te, H)})
    print("[proof] forecast baselines:", rows)
    return {"rows": rows}


def test_new_stores(Y, daily, meta, F_full, cfg, frac: float = 0.25) -> dict:
    """Pretend some stores opened 14 days before the replay: no older history, not in training."""
    rng = np.random.default_rng(cfg.seed + 2)
    stores = np.array(sorted(meta["store_id"].unique()))
    k = max(1, int(round(frac * len(stores))))
    new = set(rng.choice(stores, k, replace=False).tolist())
    is_new = meta["store_id"].isin(new).to_numpy()
    te = F_full["train_end"]
    Yn = Y.copy()
    Yn[is_new, :te - 13] = np.nan                       # only 14 days of history
    Fn = fit_forecast(Yn, daily, cfg, " new-stores", train_mask=~is_new, quiet=True)
    H = cfg.horizon_days
    res = {
        "new_stores": int(k), "history_days": 14,
        "new_store_wape": forecast_errors(Fn[0.5], Y, te, H, np.flatnonzero(is_new)),
        "same_stores_with_full_history": forecast_errors(F_full[0.5], Y, te, H, np.flatnonzero(is_new)),
        "established_stores": forecast_errors(Fn[0.5], Y, te, H, np.flatnonzero(~is_new)),
        "moving_average_on_new": forecast_errors(baseline_forecasts(Yn, te, H)["Moving average (7 days)"],
                                                 Y, te, H, np.flatnonzero(is_new)),
    }
    print("[proof] new stores:", res)
    return res


def test_unseen_cities(cfg, holdout: int = 4) -> dict:
    """Train on most cities, forecast cities the model has never seen."""
    raw = make_synthetic(replace(cfg, max_stores=24), n_cities=8) if cfg.use_synthetic else load_frn_multi(cfg)
    panel = build_panel(cfg, raw=raw)
    _, demand = correct_demand(panel, cfg)
    panel.daily["demand"] = demand.to_numpy()
    Y, _ = to_matrix(panel.daily, "demand")
    meta = panel.meta.sort_values("series_id").reset_index(drop=True)
    cities = np.array(sorted(meta["city_id"].unique()))
    rng = np.random.default_rng(cfg.seed + 3)
    held = set(rng.choice(cities, min(holdout, len(cities) - 1), replace=False).tolist())
    in_held = meta["city_id"].isin(held).to_numpy()
    F_out = fit_forecast(Y, panel.daily, cfg, " unseen-cities", train_mask=~in_held, quiet=True)
    F_all = fit_forecast(Y, panel.daily, cfg, " all-cities", quiet=True)
    te, H = F_all["train_end"], cfg.horizon_days
    res = {
        "train_cities": int(len(cities) - len(held)), "test_cities": sorted(int(c) for c in held),
        "series": int(len(meta)),
        "unseen_city_wape": forecast_errors(F_out[0.5], Y, te, H, np.flatnonzero(in_held)),
        "same_cities_if_seen": forecast_errors(F_all[0.5], Y, te, H, np.flatnonzero(in_held)),
        "moving_average": forecast_errors(baseline_forecasts(Y, te, H)["Moving average (7 days)"],
                                          Y, te, H, np.flatnonzero(in_held)),
    }
    print("[proof] unseen cities:", res)
    return res


def test_robustness(D, Y_raw, F, meta, cfg) -> list:
    """Change one assumption at a time; rerun baseline and Unwaste on the same demand."""
    def meta_shelf(scale):
        m = meta.copy()
        m["shelf_life"] = np.maximum(2, np.round(m["shelf_life"] * scale)).astype(int)
        m["min_remaining"] = np.ceil(cfg.fssai_min_remaining_frac * m["shelf_life"]).astype(int)
        m["sellable_days"] = np.maximum(1, m["shelf_life"] - m["min_remaining"])
        return m

    cases = [("Base case", {}, None),
             ("Lead time 2 days", {"lead_time_days": 2}, None),
             ("Shelf life 25% shorter", {}, 0.75),
             ("Shelf life 25% longer", {}, 1.25),
             ("Discounts lift demand less (10%)", {"markdown_uplift_per_10pct": 0.10}, None),
             ("Discounts lift demand more (30%)", {"markdown_uplift_per_10pct": 0.30}, None),
             ("Lost sale costs 25% of price", {"lost_sale_penalty_frac": 0.25}, None),
             ("Transfer cost ₹10", {"transfer_cost_per_unit": 10.0}, None)]
    out = []
    for name, over, shelf in cases:
        c = default_config(**{**cfg.to_dict(), **over})
        m = meta_shelf(shelf) if shelf else meta
        b = run_policy(POLICIES["baseline"], D, Y_raw, F, m, c).summary()
        u = run_policy(POLICIES["unwaste"], D, Y_raw, F, m, c).summary()
        row = {"case": name,
               "waste_baseline_pct": b["waste_pct_of_received"], "waste_unwaste_pct": u["waste_pct_of_received"],
               "waste_reduction_pct": round(100 * (b["waste_units"] - u["waste_units"]) / max(b["waste_units"], 1), 1),
               "service_baseline_pct": b["service_level_pct"], "service_unwaste_pct": u["service_level_pct"],
               "net_gain_rs": u["net_margin_rs"] - b["net_margin_rs"],
               "net_after_lost_gain_rs": u["net_after_lost_sales_rs"] - b["net_after_lost_sales_rs"],
               "breaches": u["shelf_life_breaches"]}
        out.append(row)
        print(f"[proof] robustness {name:34s} waste −{row['waste_reduction_pct']:5.1f}%  "
              f"service {row['service_baseline_pct']}→{row['service_unwaste_pct']}  net {row['net_gain_rs']:+,}")
    return out


# ----------------------------------------------------------------------------- run
def run(cfg: Config, skip_cities: bool = False) -> dict:
    t0 = time.time()
    panel = build_panel(cfg)
    _, demand = correct_demand(panel, cfg)
    panel.daily["demand"] = demand.to_numpy()
    Y_raw, _ = to_matrix(panel.daily, "sales")
    Y, _ = to_matrix(panel.daily, "demand")
    meta = panel.meta.sort_values("series_id").reset_index(drop=True)
    F = fit_forecast(Y, panel.daily, cfg, " corrected")
    res = {"source": panel.source,
           "baselines": test_baselines(Y, F, cfg),
           "new_stores": test_new_stores(Y, panel.daily, meta, F, cfg),
           "robustness": test_robustness(np.round(Y).astype(int), Y_raw, F, meta, cfg)}
    res["unseen_cities"] = None if skip_cities else test_unseen_cities(cfg)
    res["runtime_s"] = round(time.time() - t0)
    out = Path(cfg.out_dir) / "proof.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"[proof] wrote {out} in {res['runtime_s']}s")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 proof tests")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--stores", type=int, default=None)
    ap.add_argument("--products", type=int, default=None)
    ap.add_argument("--skip-cities", action="store_true", help="skip the slower unseen-cities test")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = default_config(use_synthetic=a.synthetic, max_stores=a.stores, max_products=a.products, out_dir=a.out)
    run(cfg, skip_cities=a.skip_cities)


if __name__ == "__main__":
    main()
