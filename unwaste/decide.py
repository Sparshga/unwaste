"""
Steps 5-7 · Decide: recover this batch, prevent recurrence, guardrails.

  "For every risky batch we price the options in rupees. A transfer earns
   price - transfer cost - salvage per unit, but only into a store with spare demand,
   only if the cups still have legal shelf life on arrival, and only if it beats a
   markdown. Across a whole city that is a min-cost-flow problem, solved exactly by
   OR-Tools. Whatever surplus is left gets an early markdown if it pays. Separately,
   the next order is set with the newsvendor rule: order up to the 85th percentile
   of demand, so we cut waste without emptying shelves."

"""
from __future__ import annotations

import numpy as np

from .backends import sample_from_quantiles
from .config import Config
from .risk import Batch, simulate_fefo


# ----------------------------------------------------------------------------- prevent
def newsvendor_order(batches: list[Batch], q10, q50, q90, arrival: int, sellable_days: int,
                     price: float, cfg: Config, rng: np.random.Generator) -> int:
    """
    Perishable newsvendor, solved by simulation.
    The new batch is the freshest, so with FEFO it only sells what existing stock can't.
    For each candidate order size Q we compute expected profit:
        price x sold  +  salvage x unsold-at-sell-by  -  cost x Q  -  penalty x lost sales
    and pick the best Q, never below the service-floor quantile of uncovered demand.
    q10/q50/q90 cover days arrival .. arrival+H-1.
    """
    H = len(q50)
    paths = np.round(sample_from_quantiles(q10, q50, q90, cfg.mc_paths, rng))
    sim = simulate_fefo(batches, paths, arrival) if batches else {"lost_by_day": paths}
    residual = sim["lost_by_day"]                       # demand existing stock can't serve
    sell_window = residual[:, :min(sellable_days, H)].sum(axis=1)
    cover = residual[:, :min(cfg.review_period_days, sellable_days, H)].sum(axis=1)  # demand this batch can legally serve
    cost = cfg.cost_ratio * price
    penalty = cfg.lost_sale_penalty_frac * price
    q_max = int(np.ceil(np.quantile(sell_window, 0.98))) + 1
    grid = np.arange(0, max(q_max, 1) + 1)
    sold = np.minimum(grid[:, None], sell_window[None, :])
    lost = np.maximum(cover[None, :] - grid[:, None], 0)
    profit = (price * sold + cfg.salvage_per_unit * (grid[:, None] - sold)
              - cost * grid[:, None] - penalty * lost).mean(axis=1)
    q_best = int(grid[np.argmax(profit)])
    q_floor = int(np.ceil(np.quantile(cover, cfg.service_floor)))   # guardrail: service floor
    return max(q_best, q_floor)


def baseline_order(recent_sales: np.ndarray, position: float, cfg: Config) -> int:
    """Simple common approach: recent average x days to cover x safety factor."""
    target = cfg.baseline_safety_factor * recent_sales.mean() * cfg.review_period_days
    return int(max(0, round(target - position)))


# ----------------------------------------------------------------------------- recover
def markdown_value(batches: list[Batch], q10, q50, q90, start_day: int, price: float,
                   frac: float, cfg: Config, rng: np.random.Generator) -> float:
    """Expected ₹ gain of marking the item down now vs not (same random demand paths)."""
    base = np.round(sample_from_quantiles(q10, q50, q90, cfg.mc_paths, rng))
    uplift = 1 + cfg.markdown_uplift_per_10pct * (frac / 0.10)
    a = simulate_fefo(batches, base, start_day)
    b = simulate_fefo(batches, np.round(base * uplift), start_day)
    rev_a = a["sold"].sum(axis=1) * price + a["leftover"].sum(axis=1) * cfg.salvage_per_unit
    rev_b = b["sold"].sum(axis=1) * price * (1 - frac) + b["leftover"].sum(axis=1) * cfg.salvage_per_unit
    return float((rev_b - rev_a).mean())


def transfer_gain_per_unit(price: float, cfg: Config) -> float:
    """Gain over dumping: p - c - s."""
    return price - cfg.transfer_cost_per_unit - cfg.salvage_per_unit


def spare_demand(q50_window: np.ndarray, sellable_stock: float, safety: float = 0.8) -> int:
    """Receiver capacity = forecast demand in the window - stock it can already sell."""
    return int(max(0, np.floor(safety * (q50_window.sum() - sellable_stock))))


# ----------------------------------------------------------------------------- guardrails
def shelf_life_ok(batch_sellby: int, today: int, cfg: Config) -> bool:
    """Moved stock must arrive with at least one legal selling day left."""
    return batch_sellby >= today + cfg.transfer_days + 1
