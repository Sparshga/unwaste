"""
Step 4 · Batch risk: Monte Carlo replay of FEFO.

WHAT THIS FILE DOES (say this to a judge):
  "FEFO is a picking rule: the oldest pack leaves first. It can't tell you whether
   a batch will sell in time. So we replay the next week many times. Each replay
   draws a realistic demand path from the forecast range and sells oldest-first,
   exactly like FEFO. Counting what's left of each batch on its sell-by line, across
   all replays, gives a probability, e.g. '85% chance of 20+ cups left'."

Input : current batches + incoming orders, forecast quantiles for the next days
Output: per-batch leftover samples -> P(leftover), expected and median leftover units
Owner : Sparsh.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backends import sample_from_quantiles


@dataclass
class Batch:
    qty: int          # units on hand (or on order)
    sellby: int       # last day index this batch may be sold (expiry minus FSSAI margin)
    arrival: int      # day index it is available from
    uid: str = ""     # e.g. "S12-P305-d61"


def simulate_fefo(batches: list[Batch], demand: np.ndarray, start_day: int) -> dict:
    """
    Sell demand oldest-first across many paths.
    demand: [paths, H] units for days start_day .. start_day+H-1
    Returns leftover[paths, n_batches] at each batch's sell-by (or at horizon end),
            sold[paths, n_batches], lost[paths] (unmet demand).
    """
    P, H = demand.shape
    order = sorted(range(len(batches)), key=lambda k: (batches[k].sellby, batches[k].arrival))
    qty = np.tile(np.array([b.qty for b in batches], float), (P, 1))
    sold = np.zeros_like(qty)
    leftover = np.full_like(qty, np.nan)
    lost = np.zeros(P)
    lost_by_day = np.zeros((P, H))
    for d in range(H):
        day = start_day + d
        need = demand[:, d].astype(float).copy()
        for k in order:
            b = batches[k]
            if b.arrival > day or b.sellby < day:
                continue
            take = np.minimum(qty[:, k], need)
            qty[:, k] -= take
            sold[:, k] += take
            need -= take
        lost += need
        lost_by_day[:, d] = need
        for k in order:                       # batches reaching their sell-by line today
            if batches[k].sellby == day:
                leftover[:, k] = qty[:, k]
    for k in range(len(batches)):             # not expired within the horizon: unknown, use 0
        if np.isnan(leftover[0, k]):
            leftover[:, k] = 0.0 if batches[k].sellby > start_day + H - 1 else qty[:, k]
    return {"leftover": leftover, "sold": sold, "lost": lost, "lost_by_day": lost_by_day, "order": order}


def batch_risk(batches: list[Batch], q10, q50, q90, start_day: int, n_paths: int,
               rng: np.random.Generator, uplift: float = 1.0) -> dict:
    """Monte Carlo: draw demand paths from the forecast range, replay FEFO, summarise."""
    demand = np.round(sample_from_quantiles(q10, q50, q90, n_paths, rng) * uplift)
    sim = simulate_fefo(batches, demand, start_day)
    L = sim["leftover"]
    summary = []
    for k, b in enumerate(batches):
        summary.append({
            "uid": b.uid, "qty": b.qty, "sellby": b.sellby,
            "p_leftover": float((L[:, k] >= 1).mean()),
            "exp_leftover": float(L[:, k].mean()),
            "median_leftover": float(np.median(L[:, k])),
            "hist": np.histogram(L[:, k], bins=12, range=(0, max(b.qty, 1)))[0].tolist(),
        })
    return {"batches": summary, "sim": sim, "demand": demand}
