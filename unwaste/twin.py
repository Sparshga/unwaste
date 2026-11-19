"""
Indian twin: the same real stores and sales patterns, moved onto an Indian city.

  What stays real : store-item demand patterns, stockouts, discounts (FreshRetailNet-50K).
  What changes    : weather -> Delhi-NCR (real Open-Meteo file if present), holidays ->
                    the Indian festival calendar of the same dates.
  How demand moves:
    weather  : MEASURED on the real data (demand by rain and temperature bucket), then
               applied to Delhi's weather. Above the hottest day in the data we extrapolate
               the measured slope (capped), and say so.
    festivals: ASSUMED ramp before each festival (D-3 .. D+1), scaled by the festival's size
               and each item's sensitivity. Stated in FEST_RAMP.

  Then we TRAIN and TEST on the twin:
    A  forecast with Delhi weather, no festival calendar
    B  forecast with Delhi weather + festival calendar (days to festival, festival size)
  Training days contain Eid ul-Fitr, Ram Navami, Akshaya Tritiya, Buddha Purnima...;
  the test weeks contain Bakrid and the monsoon onset, so B is tested on a festival it
  never saw.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .forecast import exo_matrices, fit_forecast
from .replay import POLICIES, run_policy

# date, name, size (1 = major). Public 2024 calendar (Delhi).
FESTIVALS = [
    ("2024-03-25", "Holi", 1.0), ("2024-03-29", "Good Friday", 0.3),
    ("2024-04-09", "Navratri begins / Ugadi", 0.5), ("2024-04-11", "Eid ul-Fitr", 1.0),
    ("2024-04-17", "Ram Navami", 0.8), ("2024-04-21", "Mahavir Jayanti", 0.3),
    ("2024-05-10", "Akshaya Tritiya", 0.5), ("2024-05-23", "Buddha Purnima", 0.4),
    ("2024-06-17", "Eid al-Adha (Bakrid)", 1.0), ("2024-07-07", "Rath Yatra", 0.4),
    ("2024-07-17", "Muharram", 0.3), ("2024-08-19", "Raksha Bandhan", 0.9),
    ("2024-08-26", "Janmashtami", 0.8), ("2024-10-12", "Dussehra", 0.9),
    ("2024-10-31", "Diwali", 1.0),
]
FEST_RAMP = {-3: 0.15, -2: 0.30, -1: 0.50, 0: 0.20, 1: -0.10}   # stated assumption, x size x item sensitivity
NO_FEST = 8                                                   # "days to festival" when none within a week


def festival_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Per day: days to the next festival (0..7, 8 = none), its size, and the assumed demand ramp."""
    fest = [(pd.Timestamp(d), n, w) for d, n, w in FESTIVALS]
    shift = 0
    if not any(dates.min() <= d <= dates.max() for d, _, _ in fest):     # data from another year
        shift = dates[0].year - 2024
        fest = [(d + pd.DateOffset(years=shift), n, w) for d, n, w in fest]
    rows = []
    for day in dates:
        nxt = [(int((d - day).days), n, w) for d, n, w in fest if 0 <= (d - day).days <= 7]
        k, name, w = min(nxt) if nxt else (NO_FEST, "", 0.0)
        ramp = sum(w2 * FEST_RAMP.get(int((day - d).days), 0.0) for d, _, w2 in fest)
        rows.append({"dt": day, "days_to_fest": k, "fest_weight": w, "fest_name": name,
                     "fest_ramp": ramp, "is_fest": float(any(d == day for d, _, _ in fest))})
    return pd.DataFrame(rows).set_index("dt")


def _effect_curve(x_obs: np.ndarray, rel: np.ndarray, bins: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Measured demand-vs-normal by bucket -> (bucket centre, relative demand)."""
    cut = pd.cut(x_obs, bins)
    g = pd.DataFrame({"x": x_obs, "r": rel, "b": cut}).dropna().groupby("b", observed=True)
    g = g.filter(lambda d: len(d) >= 200).groupby("b", observed=True)
    return g["x"].mean().to_numpy(), g["r"].mean().to_numpy()


def _apply_curve(x: np.ndarray, cx: np.ndarray, cy: np.ndarray, cap: tuple[float, float]) -> np.ndarray:
    """Interpolate inside the measured range; extrapolate the last measured slope outside, capped."""
    y = np.interp(x, cx, cy)
    if len(cx) >= 2:
        hi = x > cx[-1]
        y[hi] = cy[-1] + (x[hi] - cx[-1]) * (cy[-1] - cy[-2]) / max(cx[-1] - cx[-2], 1e-9)
        lo = x < cx[0]
        y[lo] = cy[0] + (x[lo] - cx[0]) * (cy[1] - cy[0]) / max(cx[1] - cx[0], 1e-9)
    return np.clip(y, *cap)


def build_twin(Y, S, daily, dates, wx: pd.DataFrame, cfg, seed: int = 21) -> dict:
    """Indian twin of the panel: demand Y_tw, sales S_tw, and a daily table with Indian inputs."""
    rng = np.random.default_rng(seed)
    S_n, T = Y.shape
    exo = exo_matrices(daily)
    rel = Y / np.maximum(Y.mean(axis=1, keepdims=True), 1e-9)
    rain_c = _effect_curve(exo["rain"].ravel(), rel.ravel(), [-0.1, 0.5, 2, 5, 10, 25, 1e9])
    temp_c = _effect_curve(exo["temp"].ravel(), rel.ravel(), [-50, 15, 20, 24, 27, 30, 33, 60])
    cap = (0.75, 1.30)

    wx = wx.reindex(dates).ffill().bfill()
    ind_rain = np.broadcast_to(wx["rain"].to_numpy()[None, :], (S_n, T))
    ind_temp = np.broadcast_to(wx["temp"].to_numpy()[None, :], (S_n, T))
    w_adj = (_apply_curve(ind_rain.ravel(), *rain_c, cap) / _apply_curve(exo["rain"].ravel(), *rain_c, cap)) \
        * (_apply_curve(ind_temp.ravel(), *temp_c, cap) / _apply_curve(exo["temp"].ravel(), *temp_c, cap))
    w_adj = np.clip(w_adj.reshape(S_n, T), 0.6, 1.6)

    cal = festival_calendar(dates)
    sens = rng.beta(2, 2, S_n)                                   # item sensitivity to festivals, 0..1
    f_adj = 1 + sens[:, None] * cal["fest_ramp"].to_numpy()[None, :]
    noise = np.exp(rng.normal(0, 0.04, (S_n, T)))
    mult = w_adj * f_adj * noise
    Y_tw, S_tw = Y * mult, S * mult

    d = daily.copy()
    for c in ["rain", "temp", "humidity", "wind"]:
        if c in wx.columns:
            d[c] = d["dt"].map(wx[c])
    for c in ["days_to_fest", "fest_weight"]:
        d[c] = d["dt"].map(cal[c]).astype(float)
    d["holiday"] = d["dt"].map(cal["is_fest"]).astype(float)       # Indian festival days replace holidays
    return {"Y": Y_tw, "S": S_tw, "daily": d, "cal": cal, "sens": sens,
            "weather_effect": {"rain_mm": np.round(rain_c[0], 1).tolist(), "rain_rel": np.round(rain_c[1], 3).tolist(),
                               "temp_c": np.round(temp_c[0], 1).tolist(), "temp_rel": np.round(temp_c[1], 3).tolist()}}


def _wape(P50, Y, te, keep_day=None):
    err = tot = 0.0
    for o in range(te, Y.shape[1] - 1):
        if keep_day is not None and not keep_day[o + 1]:
            continue
        err += np.abs(P50[:, o, 0] - Y[:, o + 1]).sum()
        tot += Y[:, o + 1].sum()
    return float(round(100 * err / max(tot, 1e-9), 1))


def train_and_test(twin: dict, meta: pd.DataFrame, cfg) -> dict:
    Y, S, d, cal = twin["Y"], twin["S"], twin["daily"], twin["cal"]
    F_a = fit_forecast(Y, d, replace(cfg, use_festivals=False), " twin A: weather only", quiet=True)
    F_b = fit_forecast(Y, d, replace(cfg, use_festivals=True), " twin B: weather + festivals")
    te = F_b["train_end"]
    dates = cal.index
    fest_win = cal["fest_ramp"].abs().to_numpy() > 0                    # days inside a festival ramp
    hot = d.groupby("dt")["temp"].first().reindex(dates).to_numpy() >= 35
    D = np.round(Y).astype(int)
    base = run_policy(POLICIES["baseline"], D, S, F_b, meta, cfg).summary()
    ua = run_policy(POLICIES["unwaste"], D, S, F_a, meta, cfg).summary()
    ub = run_policy(POLICIES["unwaste"], D, S, F_b, meta, cfg).summary()
    keep = ["waste_pct_of_received", "service_level_pct", "net_after_lost_sales_rs", "transfer_units", "shelf_life_breaches"]

    # event radar: festivals in the test weeks -> what the calendar adds to tomorrow's forecast
    radar = []
    for day in dates[te + 1:]:
        c = cal.loc[day]
        o = dates.get_loc(day) - 1
        if c["days_to_fest"] > 3 or o < te:
            continue
        with_cal, without = float(F_b[0.5][:, o, 0].sum()), float(F_a[0.5][:, o, 0].sum())
        top = np.argsort(-(F_b[0.5][:, o, 0] - F_a[0.5][:, o, 0]))[:3]
        radar.append({"date": str(day.date()), "event": c["fest_name"], "days_before": int(c["days_to_fest"]),
                      "base_units": round(without), "calendar_units": round(with_cal),
                      "uplift_pct": round(100 * (with_cal / max(without, 1e-9) - 1), 1),
                      "actual_units": round(float(Y[:, o + 1].sum())),
                      "top_items": [f"{meta.loc[i, 'store_id']}·{meta.loc[i, 'product_id']}" for i in top]})
    wx_days = d.groupby("dt")[["temp", "rain"]].first().reindex(dates)
    for day in dates[te + 1:]:
        t_, r_ = float(wx_days.loc[day, "temp"]), float(wx_days.loc[day, "rain"])
        if t_ >= 38 or r_ >= 20:
            radar.append({"date": str(day.date()), "event": f"Heatwave {t_:.0f}°C" if t_ >= 38 else f"Heavy rain {r_:.0f} mm",
                          "days_before": 0, "weather": True})
    radar.sort(key=lambda r: r["date"])

    res = {
        "train_days": int(te + 1), "test_days": int(len(dates) - te - 1),
        "train_festivals": [n for dd, n, _ in FESTIVALS if dates[0] <= pd.Timestamp(dd) <= dates[te]],
        "test_festivals": [n for dd, n, _ in FESTIVALS if dates[te + 1] <= pd.Timestamp(dd) <= dates[-1]],
        "hot_days_train": int(hot[:te + 1].sum()), "hot_days_test": int(hot[te + 1:].sum()),
        "wape": {"all_a": _wape(F_a[0.5], Y, te), "all_b": _wape(F_b[0.5], Y, te),
                 "fest_a": _wape(F_a[0.5], Y, te, fest_win), "fest_b": _wape(F_b[0.5], Y, te, fest_win),
                 "hot_b": _wape(F_b[0.5], Y, te, hot)},
        "replay": {"baseline": {k: base[k] for k in keep}, "unwaste_a": {k: ua[k] for k in keep},
                   "unwaste_b": {k: ub[k] for k in keep}},
        "radar": radar, "weather_effect": twin["weather_effect"],
        "assumptions": {"festival_ramp": FEST_RAMP, "item_sensitivity": "Beta(2,2), mean 0.5"},
    }
    w = res["wape"]
    print(f"[twin] train {res['train_days']} days ({', '.join(res['train_festivals'])}), "
          f"test {res['test_days']} days ({', '.join(res['test_festivals']) or 'no festival'})")
    print(f"[twin] festival-window error: A (weather only) {w['fest_a']}% vs B (+ calendar) {w['fest_b']}%; "
          f"all days {w['all_a']}% vs {w['all_b']}%")
    r = res["replay"]
    print(f"[twin] waste: baseline {r['baseline']['waste_pct_of_received']}%, Unwaste A {r['unwaste_a']['waste_pct_of_received']}%, "
          f"Unwaste B {r['unwaste_b']['waste_pct_of_received']}%; service {r['baseline']['service_level_pct']}% / "
          f"{r['unwaste_a']['service_level_pct']}% / {r['unwaste_b']['service_level_pct']}%")
    return res
