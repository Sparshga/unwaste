"""
Live "Predict tomorrow": the saved production model + tonight's stock + the inputs a
judge types (date, weather, festival, price change) -> next 7 days of demand and
tonight's plan (orders, risky batches, transfers, markdowns).

  weather : live Open-Meteo forecast for Gurugram; if the internet is down, the average
            weather of the same calendar days in our history. Either can be overridden.
  festival: from the public calendar; can be overridden (none / a festival type and size).
  price   : % change per category from tomorrow (a promo or a price rise).
"""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import pickle
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .decide import baseline_order, newsvendor_order
from .forecast import WEATHER, _build_rows
from .india_data import FEST_TYPES, FESTIVALS
from .replay import POLICIES, Ledger, _nightly_actions, run_policy
from .risk import Batch, batch_risk, simulate_fefo
from .backends import sample_from_quantiles

STATE = Path("models/live_state.pkl")
H = 7
LAT, LON = 28.46, 77.03                      # Gurugram


@lru_cache(maxsize=1)
def load_state(mtime: float = 0.0) -> dict:
    with open(STATE, "rb") as f:
        return pickle.load(f)


def state() -> dict:
    if not STATE.exists():
        raise FileNotFoundError("models/live_state.pkl not found: run python -m unwaste.india_pipeline")
    return load_state(STATE.stat().st_mtime)


@lru_cache(maxsize=1)
def _sample_stock(mtime: float = 0.0) -> list:
    """Sample starting stock: a store that orders every 3 days by habit (recent sales + 15%),
    replayed over the last 3 weeks of history. This is the stock Unwaste inspects tonight."""
    st = state()
    cfg = replace(st["cfg"], review_period_days=3, test_days=21)
    led = run_policy(POLICIES["baseline"], np.round(st["Y_tail"]).astype(int), st["S_tail"], None, st["meta"], cfg)
    return led.final_stock


def _target_dates(st) -> pd.DatetimeIndex:
    last = st["dates_tail"][-1]
    return pd.date_range(last + pd.Timedelta(days=1), periods=H, freq="D")


def fetch_weather(days: pd.DatetimeIndex) -> tuple[list[dict], str]:
    """Live 7-day forecast from Open-Meteo; falls back to our own climate history."""
    url = ("https://api.open-meteo.com/v1/forecast?latitude=%s&longitude=%s&timezone=Asia%%2FKolkata"
           "&daily=precipitation_sum,temperature_2m_mean,temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,wind_speed_10m_max"
           "&start_date=%s&end_date=%s" % (LAT, LON, days[0].date(), days[-1].date()))
    try:
        with urllib.request.urlopen(url, timeout=6) as r:
            d = json.loads(r.read())["daily"]
        out = []
        for i in range(len(d["time"])):
            kmh = float(d["wind_speed_10m_max"][i] or 10)
            out.append({"date": d["time"][i], "rain": float(d["precipitation_sum"][i] or 0),
                        # whole-day mean, the same field the model was trained on (archive daily mean)
                        "temp": round(float(d["temperature_2m_mean"][i]) if d.get("temperature_2m_mean") and d["temperature_2m_mean"][i] is not None
                                      else (float(d["temperature_2m_max"][i]) + float(d["temperature_2m_min"][i])) / 2, 1),
                        "humidity": float(d["relative_humidity_2m_mean"][i] or 55),
                        "wind": round((kmh / 3.6 / 0.836) ** (2 / 3), 2)})
        if len(out) == len(days):
            return out, "Open-Meteo live forecast (Gurugram)"
    except Exception as e:
        print(f"[live] weather forecast unavailable ({e}); using climate history")
    clim = state()["climate"]
    out = []
    for day in days:
        c = clim.get(day.dayofyear) or clim.get(min(clim, key=lambda k: abs(k - day.dayofyear)))
        out.append({"date": str(day.date()), "rain": round(c["rain"], 1), "temp": round(c["temp"], 1),
                    "humidity": round(c["humidity"]), "wind": round(c["wind"], 2)})
    return out, "average of the same calendar days in our history (offline)"


def calendar_for(days: pd.DatetimeIndex, override: dict | None = None) -> list[dict]:
    """Festival features per day; override = {"date": "...", "type": "diwali", "size": 1.0} or {"type": "none"}."""
    fest = [(pd.Timestamp(d), n, t, w) for d, n, t, w in FESTIVALS]
    if override and override.get("date"):
        od = pd.Timestamp(override["date"])
        fest = [f for f in fest if f[0] != od]
        if override.get("type", "none") != "none":
            fest.append((od, override.get("name") or override["type"].title(), override["type"],
                         float(override.get("size", 1.0))))
    out = []
    for day in days:
        nxt = [((d - day).days, n, t, w) for d, n, t, w in fest if 0 <= (d - day).days <= 7]
        k, name, typ, w = min(nxt) if nxt else (8, "", "none", 0.0)
        out.append({"date": str(day.date()), "days_to_fest": k, "fest_name": name, "fest_type": typ,
                    "fest_weight": w, "holiday": float(any(d == day and w2 >= 0.5 for d, _, _, w2 in fest))})
    return out


def context() -> dict:
    st = state()
    days = _target_dates(st)
    wx, src = fetch_weather(days)
    cats = sorted(st["meta"]["cat"].unique())
    return {"last_day": str(st["dates_tail"][-1].date()), "days": [str(d.date()) for d in days],
            "weather": wx, "weather_source": src, "calendar": calendar_for(days),
            "fest_types": [t for t in FEST_TYPES], "categories": cats, "built": st["built"],
            "items": sorted(st["meta"]["product_id"].unique())}


def _forecast(st, wx: list[dict], cal: list[dict], price_idx: np.ndarray, discount: np.ndarray) -> dict:
    """Quantile forecasts [S, 7] from the saved models, features built exactly as in training."""
    Y, dates = st["Y_tail"], st["dates_tail"]
    S, T = Y.shape
    meta = st["meta"]
    fut = {c: np.zeros((S, H)) for c in ["discount", "holiday", "activity", "days_to_fest", "fest_weight",
                                         "fest_type_code", "price_idx"] + WEATHER}
    for h in range(H):
        for w in WEATHER:
            fut[w][:, h] = wx[h][w]
        fut["holiday"][:, h] = cal[h]["holiday"]
        fut["days_to_fest"][:, h] = cal[h]["days_to_fest"]
        fut["fest_weight"][:, h] = cal[h]["fest_weight"]
        fut["fest_type_code"][:, h] = FEST_TYPES.index(cal[h]["fest_type"])
    fut["price_idx"][:], fut["discount"][:] = price_idx[:, None], discount
    exo = {c: np.zeros((S, T)) for c in fut}                 # history exo is not used for horizon > 0
    rows = _build_rows(Y, exo, dates, [T - 1], H, st["codes"], exo_future=fut, calendar=True)
    X = rows[st["features"]].to_numpy()
    out = {}
    for q, m in st["models"].items():
        p = np.clip(m.predict(X), 0, None)
        F = np.zeros((S, H))
        F[rows["series"].to_numpy(), rows["h"].to_numpy() - 1] = p
        out[q] = F
    out[0.1] = np.minimum(out[0.1], out[0.5])
    out[0.9] = np.maximum(out[0.9], out[0.5])
    return out


def predict(inputs: dict) -> dict:
    """inputs: {"target": "YYYY-MM-DD", "weather": {"temp":..,"rain":..,"humidity":..} (optional override for the
    target day), "festival": {"type": "auto"|"none"|<type>, "size": 1.0, "name": ".."}, "price_change": {cat: pct},
    "discount": {cat: pct}}"""
    st = state()
    meta = st["meta"]
    cfg = replace(st["cfg"], mc_paths=150)                 # fast enough to answer on stage
    days = _target_dates(st)
    target = pd.Timestamp(inputs.get("target") or days[0])
    k = int(max(0, min(H - 1, (target - days[0]).days)))
    wx, wsrc = fetch_weather(days)
    if inputs.get("weather"):
        for key, v in inputs["weather"].items():
            if v not in (None, "") and key in ("temp", "rain", "humidity", "wind"):
                wx[k][key] = float(v)
    fest = inputs.get("festival") or {"type": "auto"}
    override = None if fest.get("type", "auto") == "auto" else {"date": str(target.date()), **fest}
    cal = calendar_for(days, override)
    cats = meta["cat"].to_numpy()
    price_idx = st["price_idx"].copy()
    disc = np.zeros((len(meta), H))
    for c, pct in (inputs.get("price_change") or {}).items():
        if pct not in (None, ""):
            price_idx = np.where(cats == c, price_idx * (1 + float(pct) / 100), price_idx)
    for c, pct in (inputs.get("discount") or {}).items():
        if pct not in (None, ""):
            disc[cats == c, k] = float(pct) / 100

    F = _forecast(st, wx, cal, price_idx, disc)
    # "normal day": same stores and dates, average weather, no festival, today's prices
    clim, _ = [], None
    for day in days:
        c = st["climate"].get(day.dayofyear) or list(st["climate"].values())[0]
        clim.append({"rain": c["rain"], "temp": c["temp"], "humidity": c["humidity"], "wind": c["wind"]})
    none_cal = [{**c, "days_to_fest": 8, "fest_name": "", "fest_type": "none", "fest_weight": 0.0, "holiday": 0.0}
                for c in cal]
    F0 = _forecast(st, clim, none_cal, st["price_idx"], np.zeros_like(disc))

    # tonight's plan from tonight's stock
    sample = inputs.get("stock", "sample") == "sample"
    stock = copy.deepcopy(_sample_stock(STATE.stat().st_mtime) if sample else st["stock"])
    T = st["Y_tail"].shape[1]
    t = T - 1
    Ffull = {q: np.zeros((len(meta), T, H)) for q in F}
    for q in F:
        Ffull[q][:, t, :] = F[q]
    # every batch on hand tonight: chance it is still there at its sell-by line (before any action)
    rng_r = np.random.default_rng(cfg.seed + 1)
    d0 = st["dates_tail"][0]
    day = lambda i: str((d0 + pd.Timedelta(days=int(i))).date())
    batches = []
    for s in range(len(meta)):
        bs = [b for b in stock[s] if b.qty > 0]
        if not bs:
            continue
        res = batch_risk(bs, F[0.1][s], F[0.5][s], F[0.9][s], t + 1, cfg.mc_paths, rng_r)
        for b, r in zip(bs, res["batches"]):
            batches.append({"series": s, "uid": b.uid, "store": meta["store_id"].iloc[s], "item": meta["product_id"].iloc[s],
                            "qty": int(b.qty), "arrived": day(min(b.arrival, t)), "sellby": day(b.sellby),
                            "days_left": int(b.sellby - t), "p_leftover": round(r["p_leftover"], 2),
                            "exp_leftover": round(r["exp_leftover"], 1), "hist": r["hist"], "price": float(meta["price"].iloc[s]),
                            "demand_next": round(float(F[0.5][s, 0]), 1), "action": "ok"})
    led = Ledger()
    by_product = {}
    for s, p in enumerate(meta["product_id"].to_numpy()):
        by_product.setdefault(p, []).append(s)
    rng = np.random.default_rng(cfg.seed)
    _nightly_actions(t, stock, Ffull, meta, by_product, np.full(len(meta), -1), [], led, cfg, rng, True)
    sell_days, price = meta["sellable_days"].to_numpy(), meta["price"].to_numpy()
    orders, habit = np.zeros(len(meta), int), np.zeros(len(meta), int)
    for s in range(len(meta)):
        cur = [b for b in stock[s] if b.qty > 0]
        cs = replace(cfg, review_period_days=int(meta["review_days"].iloc[s])) if "review_days" in meta.columns else cfg
        orders[s] = newsvendor_order(cur, F[0.1][s], F[0.5][s], F[0.9][s], t + 1, int(sell_days[s]),
                                     float(price[s]), cs, rng)
        pos = sum(b.qty for b in cur if b.sellby >= t + 1)
        habit[s] = baseline_order(st["S_tail"][s, -7:], pos, cs)            # habit: recent sales + 15%, to the next delivery
    on_hand = np.array([sum(b.qty for b in stock[s] if b.qty > 0) for s in range(len(meta))])

    # KPI: expected waste from tonight's order and lost sales tomorrow, Unwaste vs habit (same demand paths)
    rng_k = np.random.default_rng(cfg.seed + 2)
    waste_u, waste_h, lost_u, lost_h = (np.zeros(len(meta)) for _ in range(4))
    for s in range(len(meta)):
        paths = np.round(sample_from_quantiles(F[0.1][s], F[0.5][s], F[0.9][s], cfg.mc_paths, rng_k))
        cur = [Batch(b.qty, b.sellby, b.arrival) for b in stock[s] if b.qty > 0]
        for q, wv, lv in ((orders[s], waste_u, lost_u), (habit[s], waste_h, lost_h)):
            nb = Batch(int(q), t + int(sell_days[s]), t + 1)
            sim = simulate_fefo(cur + [nb], paths, t + 1)
            wv[s] = float(sim["leftover"][:, -1].mean()) if q > 0 else 0.0
            lv[s] = float(sim["lost_by_day"][:, 0].mean())

    # per item, all stores
    df = pd.DataFrame({"item": meta["product_id"], "category": meta["cat"], "store": meta["store_id"],
                       "normal": F0[0.5][:, k], "p10": F[0.1][:, k], "p50": F[0.5][:, k], "p90": F[0.9][:, k],
                       "order": orders, "habit_order": habit, "on_hand": on_hand, "shelf_life": meta["shelf_life"]})
    items = df.groupby(["item", "category"], as_index=False).agg(
        normal=("normal", "sum"), p10=("p10", "sum"), p50=("p50", "sum"), p90=("p90", "sum"), order=("order", "sum"),
        habit_order=("habit_order", "sum"), on_hand=("on_hand", "sum"), shelf_life=("shelf_life", "first"))
    items["change_pct"] = (100 * (items["p50"] / items["normal"].clip(lower=1e-9) - 1)).round(1)
    items = items.sort_values("change_pct", ascending=False)
    acts = []
    for a in led.actions:
        if a["action"] == "hold":
            continue
        acts.append({"action": a["action"], "item": a["product_id"], "store": a["store_id"], "to_store": a["to_store"],
                     "units": a["units"], "p_leftover": a["p_leftover"], "value_rs": a["value_rs"],
                     "sellby": str((st["dates_tail"][0] + pd.Timedelta(days=int(a["sellby"]))).date())})
    acts.sort(key=lambda a: -a["value_rs"])
    by_uid = {(a["series"], a["batch"]): a for a in led.actions}
    for b in batches:
        a = by_uid.get((b["series"], b["uid"]))
        if a:
            b.update({"action": a["action"], "units": a["units"], "to_store": a["to_store"], "value_rs": a["value_rs"]})
        elif b["p_leftover"] >= 0.3 and b["days_left"] <= 1:
            b["action"] = "clearance"          # sells its last day tomorrow: too late to move, 30% off on the day
        elif b["p_leftover"] >= 0.3:
            b["action"] = "watch"
        b["order_tonight"] = int(orders[b["series"]])
    rank = {"transfer": 0, "markdown": 1, "hold": 2, "clearance": 3, "watch": 4, "ok": 5}
    batches.sort(key=lambda b: (rank[b["action"]], -b["p_leftover"], b["days_left"]))
    risky = sum(1 for a in led.actions)
    tot = lambda F_: [round(float(F_[:, h].sum())) for h in range(H)]
    return {
        "target": str(days[k].date()), "k": k, "weather_used": wx[k], "weather_source": wsrc,
        "festival_used": cal[k], "days": [str(d.date()) for d in days],
        "totals": {"p10": tot(F[0.1]), "p50": tot(F[0.5]), "p90": tot(F[0.9]), "normal": tot(F0[0.5])},
        "kpi": {"demand": round(float(items["p50"].sum())), "normal": round(float(items["normal"].sum())),
                "change_pct": float(round(100 * (items["p50"].sum() / max(items["normal"].sum(), 1e-9) - 1), 1)),
                "order_units": int(orders.sum()), "habit_units": int(habit.sum()), "risky_batches": risky,
                "transfers": sum(a["units"] for a in acts if a["action"] == "transfer"),
                "markdowns": sum(1 for a in acts if a["action"] == "markdown")},
        "items": json.loads(items.round(1).to_json(orient="records")), "actions": acts,
        "batches": batches[:400], "n_batches": len(batches), "stores": sorted(meta["store_id"].unique()),
        "series": [{"store": meta["store_id"].iloc[s], "item": meta["product_id"].iloc[s], "cat": meta["cat"].iloc[s],
                    "shelf": int(meta["shelf_life"].iloc[s]), "order": int(orders[s]), "on_hand": int(on_hand[s]),
                    "habit": int(habit[s]), "price": float(price[s]),
                    "cycle": int(meta["review_days"].iloc[s]) if "review_days" in meta.columns else int(cfg.review_period_days),
                    "cost": round(float(cfg.cost_ratio * price[s]), 2),
                    "waste_u": round(waste_u[s], 1), "waste_h": round(waste_h[s], 1),
                    "lost_u": round(lost_u[s], 1), "lost_h": round(lost_h[s], 1),
                    "normal": np.round(F0[0.5][s], 1).tolist(), "p10": np.round(F[0.1][s], 1).tolist(),
                    "p50": np.round(F[0.5][s], 1).tolist(), "p90": np.round(F[0.9][s], 1).tolist()}
                   for s in range(len(meta))],
        "service_floor": cfg.service_floor, "mc_paths": cfg.mc_paths,
        "calendar": cal, "weather": wx, "price_change": inputs.get("price_change") or {},
        "stock_source": "sample: a store that orders every 3 days by habit (simulated, last 3 weeks)" if sample
                        else "Unwaste's own daily-ordering stock from the replay",
    }
