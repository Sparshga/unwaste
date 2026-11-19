"""
Step 8 · Decision replay: baseline vs Unwaste on the SAME demand.

  "We replay the last 4 weeks day by day for every store and item. Both policies face
   exactly the same customer demand, so the comparison is fair. The baseline is a
   common simple approach: reorder from the recent average, sell oldest-first, and
   discount 30% only on a batch's last day. Unwaste uses the corrected forecast, the
   Monte Carlo risk, transfers, early markdowns and newsvendor ordering. Every night
   we log what each policy did, and at the end we count waste, rupees, service level,
   false moves and shelf-life breaches."

"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backends import max_gain_transfers
from .config import Config
from .decide import (baseline_order, markdown_value, newsvendor_order, shelf_life_ok,
                     spare_demand, transfer_gain_per_unit)
from .risk import Batch, batch_risk


@dataclass
class Policy:
    name: str
    use_forecast: bool          # newsvendor on forecasts (else baseline ordering)
    use_actions: bool           # risk -> transfers / early markdown
    late_markdown: bool         # baseline last-day clearance


POLICIES = {
    "baseline": Policy("Baseline: recent average + FEFO + last-day 30% markdown", False, False, True),
    "forecast_raw": Policy("Forecast on raw sales (no stockout correction)", True, False, True),
    "forecast_corrected": Policy("+ stockout correction", True, False, True),
    "unwaste": Policy("Unwaste: + batch risk, transfers, early markdown", True, True, True),
}


@dataclass
class Ledger:
    demand: float = 0.0
    served: float = 0.0
    received: float = 0.0
    waste: float = 0.0
    revenue: float = 0.0
    salvage: float = 0.0
    purchase: float = 0.0
    transfer_cost: float = 0.0
    transfer_units: int = 0
    markdowns: int = 0
    late_markdowns: int = 0
    false_move_units: float = 0.0
    breaches: int = 0
    ending_stock_value: float = 0.0
    lost_units: float = 0.0
    lost_value: float = 0.0          # lost units x price
    penalty_frac: float = 0.5
    actions: list = field(default_factory=list)

    def summary(self) -> dict:
        # stock still on the shelf at the end is valued at cost, so ordering more isn't penalised unfairly
        net = self.revenue + self.salvage - self.purchase - self.transfer_cost + self.ending_stock_value
        return {
            "waste_units": round(self.waste),
            "waste_pct_of_received": round(100 * self.waste / max(self.received, 1), 2),
            "service_level_pct": round(100 * self.served / max(self.demand, 1), 2),
            "net_margin_rs": round(net),
            "lost_units": round(self.lost_units),
            "net_after_lost_sales_rs": round(net - self.penalty_frac * self.lost_value),
            "revenue_rs": round(self.revenue),
            "purchase_rs": round(self.purchase),
            "transfer_units": int(self.transfer_units),
            "early_markdowns": int(self.markdowns),
            "late_markdowns": int(self.late_markdowns),
            "false_move_units": round(self.false_move_units),
            "shelf_life_breaches": int(self.breaches),
        }


def run_policy(policy: Policy, D: np.ndarray, S_obs: np.ndarray, F: dict | None,
               meta: pd.DataFrame, cfg: Config, log_actions: bool = False) -> Ledger:
    """
    D     : true demand [series, days] (stockout-corrected, rounded)
    S_obs : observed sales [series, days] (what a simple system would see)
    F     : forecasts {q: [series, origin, horizon]} or None
    """
    rng = np.random.default_rng(cfg.seed)
    nS, T = D.shape
    t0 = F["train_end"] + 1 if F is not None else T - cfg.test_days
    price = meta["price"].to_numpy()
    unit_cost = cfg.cost_ratio * price
    sell_days = meta["sellable_days"].to_numpy()
    R, L = cfg.review_period_days, cfg.lead_time_days
    by_product = defaultdict(list)
    for s, p in enumerate(meta["product_id"].to_numpy()):
        by_product[p].append(s)

    led = Ledger(penalty_frac=cfg.lost_sale_penalty_frac)
    stock: list[list[Batch]] = [[] for _ in range(nS)]
    markdown_until = np.full(nS, -1)
    open_moves = []                                     # (sender, until_day, units)
    lost_today = np.zeros(nS)

    # identical starting stock for every policy: ~R days of average demand, fresh
    for s in range(nS):
        q = int(round(D[s, max(0, t0 - 14):t0].mean() * R))
        stock[s].append(Batch(q, t0 + sell_days[s] - 1, t0, f"{s}-init"))
        led.received += q
        led.purchase += q * unit_cost[s]

    for t in range(t0, T):
        lost_today[:] = 0
        # ---------------- sell today, oldest first ----------------
        for s in range(nS):
            batches = [b for b in stock[s] if b.arrival <= t and b.qty > 0]
            md = 0.0
            if t <= markdown_until[s]:
                md = cfg.markdown_frac
            elif policy.late_markdown and any(b.sellby == t for b in batches):
                md = cfg.late_markdown_frac
                led.late_markdowns += 1
            uplift = 1 + cfg.markdown_uplift_per_10pct * (md / 0.10)
            want = int(round(D[s, t] * uplift))
            base_want = D[s, t]
            sold = 0
            for b in sorted(batches, key=lambda b: b.sellby):
                if b.sellby < t:
                    continue
                take = min(b.qty, want - sold)
                b.qty -= take
                sold += take
                if sold >= want:
                    break
            led.demand += base_want
            led.served += min(sold, base_want)
            led.revenue += sold * price[s] * (1 - md)
            lost_today[s] = max(0, base_want - sold)
            led.lost_units += lost_today[s]
            led.lost_value += lost_today[s] * price[s]
            # expire anything at its sell-by line tonight
            keep = []
            for b in stock[s]:
                if b.sellby <= t and b.arrival <= t:
                    led.waste += b.qty
                    led.salvage += b.qty * cfg.salvage_per_unit
                else:
                    keep.append(b)
            stock[s] = keep

        # false moves: a sender that ran out after giving stock away
        still = []
        for (s, until, units) in open_moves:
            if t <= until:
                hit = min(units, lost_today[s])
                led.false_move_units += hit
                units -= hit
                if units > 0 and t < until:
                    still.append((s, until, units))
        open_moves = still

        if t == T - 1:
            for s in range(nS):
                led.ending_stock_value += sum(b.qty for b in stock[s]) * unit_cost[s]
            break

        # ---------------- nightly decisions ----------------
        if policy.use_actions and F is not None:
            _nightly_actions(t, stock, F, meta, by_product, markdown_until, open_moves,
                             led, cfg, rng, log_actions)

        # ---------------- ordering ----------------
        for s in range(nS):
            if (t - s) % R != 0:
                continue
            arrival = t + L
            position = sum(b.qty for b in stock[s] if b.sellby >= arrival)
            if policy.use_forecast and F is not None:
                h0 = arrival - t - 1
                cur = [Batch(b.qty, b.sellby, b.arrival) for b in stock[s] if b.qty > 0]
                q = newsvendor_order(cur, F[0.1][s, t, h0:], F[0.5][s, t, h0:], F[0.9][s, t, h0:],
                                     arrival, int(sell_days[s]), float(price[s]), cfg, rng)
            else:
                q = baseline_order(S_obs[s, max(0, t - 6):t + 1], position, cfg)
            if q > 0:
                stock[s].append(Batch(q, arrival + sell_days[s] - 1, arrival, f"{s}-d{arrival}"))
                led.received += q
                led.purchase += q * unit_cost[s]
    return led


def _nightly_actions(t, stock, F, meta, by_product, markdown_until, open_moves, led, cfg,
                     rng, log_actions):
    H = cfg.horizon_days
    price = meta["price"].to_numpy()
    start = t + 1
    for product, series in by_product.items():
        senders, info = {}, {}
        for s in series:
            bs = [b for b in stock[s] if b.qty > 0]
            if not bs:
                continue
            q10, q50, q90 = F[0.1][s, t], F[0.5][s, t], F[0.9][s, t]
            res = batch_risk(bs, q10, q50, q90, start, cfg.mc_paths, rng)
            for b, r in zip(bs, res["batches"]):
                if (r["p_leftover"] >= cfg.risk_threshold and r["median_leftover"] >= cfg.min_units_to_act
                        and b.sellby >= t + 2):
                    surplus = int(min(b.qty, round(r["median_leftover"])))
                    md_gain = markdown_value(bs, q10, q50, q90, start, price[s], cfg.markdown_frac, cfg, rng)
                    senders[s] = surplus
                    info[s] = {"batch": b, "risk": r, "md_gain": md_gain}
                    break                                   # one risky batch per series per night
        if not senders:
            continue
        # receivers: same product elsewhere, spare demand over the next 3 days
        capacity = {}
        for j in series:
            if j in senders:
                continue
            w0, w1 = cfg.transfer_days - 1, min(cfg.transfer_days + 2, H)
            sellable = sum(b.qty for b in stock[j] if b.sellby >= t + cfg.transfer_days)
            cap = spare_demand(F[0.5][j, t, w0:w1], sellable)
            if cap > 0:
                capacity[j] = cap
        gain = {}
        for i in senders:
            b = info[i]["batch"]
            if not shelf_life_ok(b.sellby, t, cfg):                 # guardrail: legal shelf life
                continue
            g = transfer_gain_per_unit(price[i], cfg)
            md_per_unit = info[i]["md_gain"] / max(senders[i], 1)
            if g <= md_per_unit:                                    # guardrail: must beat markdown
                continue
            for j in capacity:
                gain[(i, j)] = g
        moves = max_gain_transfers(senders, capacity, gain) if gain else {}
        moved = defaultdict(int)
        for (i, j), x in moves.items():
            b = info[i]["batch"]
            x = min(x, b.qty)
            if x <= 0:
                continue
            b.qty -= x
            moved[i] += x
            stock[j].append(Batch(x, b.sellby, t + cfg.transfer_days, f"{b.uid}->{j}"))
            led.transfer_units += x
            led.transfer_cost += x * cfg.transfer_cost_per_unit
            open_moves.append((i, b.sellby, x))
            if log_actions:
                led.actions.append(_log(t, i, meta, info[i], "transfer", x, j,
                                        x * transfer_gain_per_unit(price[i], cfg)))
        for i in senders:
            left = senders[i] - moved[i]
            if left >= cfg.min_units_to_act and info[i]["md_gain"] > 0 and markdown_until[i] < t + 1:
                markdown_until[i] = info[i]["batch"].sellby
                led.markdowns += 1
                if log_actions:
                    led.actions.append(_log(t, i, meta, info[i], "markdown", left, None,
                                            info[i]["md_gain"]))
            elif left >= cfg.min_units_to_act and log_actions:
                led.actions.append(_log(t, i, meta, info[i], "hold", left, None, 0.0))


def _log(t, s, meta, inf, kind, units, to, value):
    r = inf["risk"]
    m = meta.iloc[s]
    sid = lambda k: str(meta["store_id"].iloc[k])
    return {"night": int(t), "series": int(s), "store_id": sid(s),
            "product_id": str(meta["product_id"].iloc[s]), "batch": inf["batch"].uid,
            "batch_qty": int(inf["batch"].qty + (units if kind == "transfer" else 0)),
            "sellby": int(inf["batch"].sellby), "p_leftover": round(r["p_leftover"], 2),
            "median_leftover": round(r["median_leftover"], 1), "hist": r["hist"],
            "action": kind, "units": int(units),
            "to_store": None if to is None else sid(to),
            "value_rs": round(float(value)), "price": float(m["price"])}
