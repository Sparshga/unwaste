"""
The main pipeline on Unwaste-India: fill in hidden demand, train, test, replay, and save
everything the live "Predict tomorrow" screen needs.

  train : 1 Jan 2024 -> 31 Jul 2026  (two Diwalis, two Holis, two monsoons, two winters)
  test  : 1 Aug 2026 -> last day      (Raksha Bandhan, Janmashtami, Navratri, Dussehra, Diwali, Chhath)

Run:  python -m unwaste.india_data        (once: makes the dataset)
      python -m unwaste.india_pipeline    -> web/data/india_main.json, india_actions.json, models/live_state.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .backends import backend_names
from .config import default_config
from .forecast import fit_forecast, to_matrix
from .data import Panel
from .impute import correct_demand, demand_truth_test
from .india_data import FESTIVALS, OPEN, OUT as DATA_DIR
from .replay import POLICIES, run_policy

TEST_START = "2026-08-01"
STATE = Path("models/live_state.pkl")
TAIL = 42                                         # days of history the live screen keeps


def load_dataset(path: Path = DATA_DIR / "unwaste_india.csv") -> pd.DataFrame:
    hp = path.with_name("unwaste_india_hourly.npz")
    if not path.exists() or not hp.exists():
        raise SystemExit(f"{path} or {hp.name} not found: run  python -m unwaste.india_data  first")
    df = pd.read_csv(path, parse_dates=["date"], dtype={"hours_stock_status": str})
    h = np.load(hp)
    df.attrs["hourly"] = {"hs": h["hours_sale"], "oos": h["hours_stock_status"], "hd": h["hours_demand_true"]}
    return df


DAILY_MAX_SHELF_LIFE = 4


def panel(df: pd.DataFrame, cfg, delivery_days: int = 3):
    """Daily table + 24-hour sales and stockout flags -> the same Panel FreshRetailNet uses."""
    d = df.rename(columns={"date": "dt"}).copy()
    d["store_id"], d["product_id"] = d["store_name"], d["item"]       # readable names everywhere
    d["series_id"] = d.groupby(["store_id", "product_id"], sort=True).ngroup()
    d["activity"] = 0.0
    hr = df.attrs["hourly"]
    meta = d.groupby("series_id").agg(store_id=("store_id", "first"), product_id=("product_id", "first"),
                                      cat=("category", "first"), price=("base_price", "first"),
                                      shelf_life=("shelf_life_days", "first")).reset_index()
    meta["city_id"] = 0
    meta["min_remaining"] = np.ceil(0.30 * meta["shelf_life"]).astype(int)       # FSSAI: 30% left at sale
    meta["sellable_days"] = (meta["shelf_life"] - meta["min_remaining"]).clip(lower=1)
    meta["review_days"] = np.where(meta["shelf_life"] <= DAILY_MAX_SHELF_LIFE, 1, delivery_days)
    pnl = Panel(daily=d, hourly_sales=hr["hs"].astype(float), hourly_oos=hr["oos"].astype(bool), meta=meta,
                source="Unwaste-India")
    # step 1, exactly as on FreshRetailNet: a model trained only on in-stock hours fills the empty ones
    corrected, daily = correct_demand(pnl, cfg)
    d["demand"] = daily.to_numpy()
    return d, meta, pnl, corrected


def _wape(P, Y, te, mask=None):
    err = tot = 0.0
    for o in range(te, Y.shape[1] - 1):
        if mask is not None and not mask[o + 1]:
            continue
        err += np.abs(P[:, o] - Y[:, o + 1]).sum()
        tot += Y[:, o + 1].sum()
    return round(100 * err / max(tot, 1e-9), 1)


def run(fast: bool = False, delivery_days: int = 3) -> dict:
    t0 = time.time()
    df = load_dataset()
    cfg0 = default_config(open_hours=OPEN, n_estimators=120 if fast else 200)
    d, meta, pnl, corrected = panel(df, cfg0, delivery_days)
    Y, dates = to_matrix(d, "demand")
    S_obs, _ = to_matrix(d, "sales")
    Y_true, _ = to_matrix(d, "demand_true")
    T = len(dates)
    test_days = int((dates > pd.Timestamp(TEST_START) - pd.Timedelta(days=1)).sum())
    cfg = default_config(test_days=test_days, calendar_feats=True, review_period_days=1, open_hours=OPEN, late_markdown_units_only=True, newsvendor_review_window=True, train_origin_step=3 if fast else 2,
                         n_estimators=120 if fast else 200, out_dir="web/data")
    print(f"[india] {Y.shape[0]} series x {T} days ({dates[0].date()} -> {dates[-1].date()}); "
          f"train to {dates[T - test_days - 1].date()}, test {test_days} days; backends {backend_names()}")

    # 1. fill-in test: we know the true hourly demand here, a real store does not
    hr = df.attrs["hourly"]
    oos = hr["oos"].astype(bool)
    hs, hd = hr["hs"].astype(float), hr["hd"].astype(float)
    sid = d["series_id"].to_numpy()
    instock = ~oos
    sum_sh = np.zeros((sid.max() + 1, 24))
    cnt_sh = np.zeros((sid.max() + 1, 24))
    np.add.at(sum_sh, sid, np.where(instock, hs, 0))
    np.add.at(cnt_sh, sid, instock)
    avg_sh = (sum_sh / np.maximum(cnt_sh, 1))[sid]                     # simple average of in-stock hours
    truth = hd[oos]
    err = lambda est: round(100 * float(np.abs(est - truth).sum() / max(truth.sum(), 1e-9)), 1)
    so_days = df["stockout_hours"].to_numpy() > 0
    t_day = df["demand_true"].to_numpy()[so_days]
    fill = {"stockout_days_pct": round(100 * so_days.mean(), 1),
            "empty_hours_pct": round(100 * oos.sum() / (len(df) * (OPEN[1] - OPEN[0] + 1)), 1),
            "typical_start_hour": int(np.median(df.loc[so_days, "first_stockout_hour"])),
            "hidden_demand_pct": round(100 * (1 - df["sales"].sum() / df["demand_true"].sum()), 1),
            "hour_error_sales": err(hs[oos]), "hour_error_average": err(np.maximum(avg_sh[oos], hs[oos])),
            "hour_error_model": err(corrected[oos]),
            "error_sales_as_demand": round(100 * np.abs(df["sales"].to_numpy()[so_days] - t_day).sum() / t_day.sum(), 1),
            "error_filled": round(100 * np.abs(d["demand"].to_numpy()[so_days] - t_day).sum() / t_day.sum(), 1)}
    fill["demand_truth"] = demand_truth_test(pnl, cfg0)
    print(f"[fill-in] empty hours {fill['empty_hours_pct']}% (stockouts usually from {fill['typical_start_hour']}:00). "
          f"Error on empty hours vs true demand: sales as-is {fill['hour_error_sales']}% | simple average "
          f"{fill['hour_error_average']}% | model {fill['hour_error_model']}%")
    print(f"[fill-in] whole stockout days: sales as-is {fill['error_sales_as_demand']}% -> filled in {fill['error_filled']}%")

    # 2. forecast: ours vs no-calendar vs simple baselines, on the test window
    F = fit_forecast(Y, d, cfg, " Unwaste (weather + festivals + price)")
    F_nc = fit_forecast(Y, d, replace(cfg, use_festivals=False, calendar_feats=False), " without calendar", quiet=True)
    F_raw = fit_forecast(S_obs, d, cfg, " on raw sales (no fill-in)", quiet=True)
    te = F["train_end"]
    P_ours, P_nc = F[0.5][:, :, 0], F_nc[0.5][:, :, 0]
    P_ma = np.full_like(P_ours, np.nan)
    P_lw = np.full_like(P_ours, np.nan)
    for o in range(te, T):
        P_ma[:, o] = Y[:, max(0, o - 6):o + 1].mean(axis=1)
        P_lw[:, o] = Y[:, o - 6]
    fest_days = np.zeros(T, bool)
    for dd, name, typ, w in FESTIVALS:
        if w >= 0.5:
            k = dates.searchsorted(pd.Timestamp(dd))
            fest_days[max(0, k - 3):min(T, k + 1)] = True                 # 3 days before + the day
    fc = {"all": {"ours": _wape(P_ours, Y_true, te), "no_calendar": _wape(P_nc, Y_true, te),
                  "avg_7d": _wape(P_ma, Y_true, te), "same_weekday": _wape(P_lw, Y_true, te)},
          "festival_days": {"ours": _wape(P_ours, Y_true, te, fest_days), "no_calendar": _wape(P_nc, Y_true, te, fest_days),
                            "avg_7d": _wape(P_ma, Y_true, te, fest_days), "same_weekday": _wape(P_lw, Y_true, te, fest_days)},
          "coverage_p10_p90": round(100 * F["metrics"]["coverage_p10_p90"], 1)}
    per_fest = []
    for dd, name, typ, w in FESTIVALS:
        k = dates.searchsorted(pd.Timestamp(dd))
        if w >= 0.5 and te + 1 <= k < T:
            m = np.zeros(T, bool)
            m[max(te + 1, k - 3):k + 1] = True
            per_fest.append({"festival": name, "date": dd, "ours": _wape(P_ours, Y_true, te, m),
                             "no_calendar": _wape(P_nc, Y_true, te, m), "avg_7d": _wape(P_ma, Y_true, te, m)})
    fc["per_festival"] = per_fest
    print(f"[forecast] test error: ours {fc['all']['ours']}% | no calendar {fc['all']['no_calendar']}% | "
          f"7-day avg {fc['all']['avg_7d']}% | same weekday {fc['all']['same_weekday']}%")
    print(f"[forecast] festival days: ours {fc['festival_days']['ours']}% | no calendar "
          f"{fc['festival_days']['no_calendar']}% | 7-day avg {fc['festival_days']['avg_7d']}%")

    # 3. replay: reality = TRUE demand; policies only see sales and forecasts
    D = np.round(Y_true).astype(int)
    rep, ledgers = {}, {}
    for key in ["baseline", "forecast_raw", "forecast_corrected", "unwaste"]:
        led = run_policy(POLICIES[key], D, S_obs, F_raw if key == "forecast_raw" else F, meta, cfg,
                         log_actions=(key == "unwaste"))
        ledgers[key] = led
        rep[key] = {"label": POLICIES[key].name, **led.summary(), "transfer_units": led.transfer_units,
                    "markdowns": led.markdowns}
        print(f"[replay] {key:20s} waste {rep[key]['waste_pct_of_received']:5.2f}%  service "
              f"{rep[key]['service_level_pct']:5.2f}%  net after lost sales ₹{rep[key]['net_after_lost_sales_rs']:,}")
    # the service dial: a manager can trade a little waste for fewer empty shelves
    dial = []
    for floor in (0.6, 0.75, 0.85):
        sm = ledgers["unwaste"].summary() if floor == cfg.service_floor else \
            run_policy(POLICIES["unwaste"], D, S_obs, F, meta, replace(cfg, service_floor=floor)).summary()
        dial.append({"service_floor": floor, "waste_pct": sm["waste_pct_of_received"],
                     "service_pct": sm["service_level_pct"], "net_after_lost_rs": sm["net_after_lost_sales_rs"]})
    print("[dial] " + " | ".join(f"floor {x['service_floor']}: waste {x['waste_pct']}% service {x['service_pct']}%" for x in dial))
    b, u = rep["baseline"], rep["unwaste"]
    headline = {"waste_cut_pct": round(100 * (b["waste_units"] - u["waste_units"]) / max(b["waste_units"], 1), 1),
                "service_gain_pts": round(u["service_level_pct"] - b["service_level_pct"], 2),
                "net_after_lost_gain_rs": u["net_after_lost_sales_rs"] - b["net_after_lost_sales_rs"]}
    print(f"[headline] waste down {headline['waste_cut_pct']}%, service {headline['service_gain_pts']:+} pts, "
          f"net after lost-sale cost {headline['net_after_lost_gain_rs']:+,} ₹ over {test_days} days")

    # 4. daily totals for the chart: actual vs next-day forecast in the test window
    days = list(range(te + 1, T))
    chart = {"dates": [str(dates[t].date()) for t in days],
             "actual": [round(float(Y_true[:, t].sum())) for t in days],
             "p50": [round(float(F[0.5][:, t - 1, 0].sum())) for t in days],
             "p10": [round(float(F[0.1][:, t - 1, 0].sum())) for t in days],
             "p90": [round(float(F[0.9][:, t - 1, 0].sum())) for t in days],
             "avg_7d": [round(float(P_ma[:, t - 1].sum())) for t in days],
             "festivals": [{"date": dd, "name": n} for dd, n, _, w in FESTIVALS
                           if w >= 0.5 and str(dates[te + 1].date()) <= dd <= str(dates[-1].date())]}

    wsrc = df.drop_duplicates("date")["weather_source"].value_counts().to_dict()
    out = {"dataset": {"rows": int(len(df)), "stores": int(df.store_id.nunique()), "items": int(df.product_id.nunique()),
                       "days": int(T), "start": str(dates[0].date()), "end": str(dates[-1].date()),
                       "train_end": str(dates[te].date()), "test_days": test_days,
                       "weather_days": {k: int(v) for k, v in wsrc.items()},
                       "categories": df.groupby("category").item.nunique().to_dict()},
           "fill_in": fill, "forecast": fc, "replay": rep, "headline": headline, "chart": chart, "dial": dial,
           "features": F["features"], "backends": backend_names(), "runtime_s": 0,
           "replenishment": {"daily_max_shelf_life": DAILY_MAX_SHELF_LIFE, "every_days": delivery_days,
                             "daily_items": int((meta.drop_duplicates("product_id")["review_days"] == 1).sum()),
                             "periodic_items": int((meta.drop_duplicates("product_id")["review_days"] > 1).sum())}}
    # store actions for the last 7 nights, readable
    acts = []
    for a in ledgers["unwaste"].actions:
        if a["night"] >= T - 8:
            a = dict(a)
            a["date"] = str(dates[a["night"]].date())
            a["sellby_date"] = str((dates[0] + pd.Timedelta(days=int(a["sellby"]))).date())
            acts.append(a)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # arrays for the results script (scripts/india_results.py)
    so_m, _ = to_matrix(d, "stockout_hours")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    last = (d["dt"] >= dates[-75]).to_numpy()
    np.savez_compressed(STATE.parent / "india_hourly_eval.npz", series=sid[last], dt=d.loc[last, "dt"].astype(str).to_numpy(),
                        hs=hs[last], hd=hd[last], oos=oos[last], corrected=corrected[last])
    np.savez_compressed(STATE.parent / "india_eval.npz", dates=dates.astype(str).to_numpy(), Y=Y, S=S_obs, Y_true=Y_true,
                        stockout=so_m, te=te, p10=F[0.1][:, te:, 0], p50=F[0.5][:, te:, 0], p90=F[0.9][:, te:, 0],
                        p50_nc=P_nc[:, te:], p50_raw=F_raw[0.5][:, te:, 0], stores=meta["store_id"].to_numpy(),
                        items=meta["product_id"].to_numpy(), cats=meta["cat"].to_numpy())

    # 5. live state: production model on ALL days + the stock on hand tonight
    cfg_live = replace(cfg, test_days=1)
    F_live = fit_forecast(Y, d, cfg_live, " production model (all days)", quiet=True)
    stock = ledgers["unwaste"].final_stock
    off = T - TAIL
    from .risk import Batch
    tail_stock = [[Batch(b.qty, b.sellby - off, b.arrival - off, b.uid) for b in st if b.qty > 0] for st in stock]
    last = d[d["dt"] == dates[-1]].sort_values("series_id")
    clim = d.drop_duplicates("dt").assign(doy=lambda x: x["dt"].dt.dayofyear)[["doy", "rain", "temp", "humidity", "wind"]]
    state = {"models": F_live["models"], "features": F_live["features"], "codes": F_live["codes"],
             "Y_tail": Y[:, off:], "S_tail": S_obs[:, off:], "dates_tail": dates[off:], "meta": meta,
             "stock": tail_stock, "price_idx": last["price_idx"].to_numpy(), "cfg": cfg_live,
             "climate": clim.groupby("doy").mean().to_dict("index"), "last_weather":
                 last[["rain", "temp", "humidity", "wind"]].iloc[0].to_dict(), "built": time.strftime("%Y-%m-%d %H:%M")}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE, "wb") as f:
        pickle.dump(state, f)
    out["runtime_s"] = round(time.time() - t0)
    for name, obj in (("india_main.json", out), ("india_actions.json", {"actions": acts})):
        (Path(cfg.out_dir) / name).write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"[india] wrote web/data/india_main.json, india_actions.json and {STATE} in {out['runtime_s']}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and test on Unwaste-India; save the live model")
    ap.add_argument("--fast", action="store_true", help="fewer trees, fewer training days (quick check)")
    ap.add_argument("--delivery-days", type=int, default=3,
                    help=f"delivery cycle for items with shelf life above {DAILY_MAX_SHELF_LIFE} days (shorter-life items: daily)")
    a = ap.parse_args()
    run(a.fast, a.delivery_days)


if __name__ == "__main__":
    main()
