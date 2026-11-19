"""
Library adapters: LightGBM + OR-Tools when installed, safe fallbacks otherwise.

  "Our two heavy libraries are LightGBM for forecasting and Google OR-Tools for the
   transfer optimisation. This file is the only place that talks to them, so if a
   laptop can't install one, the same code runs on scikit-learn's gradient boosting
   or NetworkX's min-cost flow, and the results are the same kind of answer."

"""
from __future__ import annotations

import numpy as np

try:
    import lightgbm as lgb  # type: ignore
    HAS_LGBM = True
except Exception:  # pragma: no cover - depends on machine
    HAS_LGBM = False

try:
    from ortools.graph.python import min_cost_flow as _ortools_mcf  # type: ignore
    HAS_ORTOOLS = True
except Exception:  # pragma: no cover
    HAS_ORTOOLS = False


def backend_names() -> dict:
    return {"forecast": "LightGBM" if HAS_LGBM else "scikit-learn HistGradientBoosting",
            "optimiser": "OR-Tools min-cost flow" if HAS_ORTOOLS else "NetworkX min-cost flow"}


# ----------------------------------------------------------------------------- quantile model
def quantile_model(alpha: float, seed: int = 7, n_estimators: int = 300):
    """Gradient-boosted trees that predict the alpha-quantile of demand."""
    if HAS_LGBM:
        return lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=n_estimators,
                                 learning_rate=0.05, num_leaves=31, min_child_samples=20,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                                 random_state=seed, verbose=-1)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="quantile", quantile=alpha, max_iter=n_estimators,
                                         learning_rate=0.05, max_leaf_nodes=31,
                                         random_state=seed)


def mean_model(seed: int = 7, n_estimators: int = 300):
    """Gradient-boosted trees for the expected value (used by the fill-in step)."""
    if HAS_LGBM:
        return lgb.LGBMRegressor(objective="poisson", n_estimators=n_estimators,
                                 learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="poisson", max_iter=n_estimators,
                                         learning_rate=0.05, random_state=seed)


# ----------------------------------------------------------------------------- min-cost flow
def max_gain_transfers(supply: dict, capacity: dict, gain: dict) -> dict:
    """
    Solve the city-wide transfer problem exactly.

      maximise   sum_ij x_ij * gain_ij
      subject to sum_j x_ij <= supply_i      (sender surplus)
                 sum_i x_ij <= capacity_j    (receiver spare demand)
                 x_ij >= 0, whole units      (only arcs with gain_ij > 0 exist)

    As a min-cost flow: costs are -gain (in paise, integers). Unused surplus flows to a
    'dump' sink at zero cost, so moving stock is never forced.
    supply: {sender: units}, capacity: {receiver: units}, gain: {(sender, receiver): ₹/unit}
    Returns {(sender, receiver): units}.
    """
    arcs = [(i, j, g) for (i, j), g in gain.items()
            if g > 0 and supply.get(i, 0) > 0 and capacity.get(j, 0) > 0]
    if not arcs:
        return {}
    senders = sorted({i for i, _, _ in arcs}, key=str)
    receivers = sorted({j for _, j, _ in arcs}, key=str)
    total = int(sum(supply[i] for i in senders))

    if HAS_ORTOOLS:
        smcf = _ortools_mcf.SimpleMinCostFlow()
        nid = {}
        for n in [("s", i) for i in senders] + [("r", j) for j in receivers] + [("sink", 0)]:
            nid[n] = len(nid)
        sink = nid[("sink", 0)]
        arc_index = {}
        for i, j, g in arcs:
            a = smcf.add_arc_with_capacity_and_unit_cost(nid[("s", i)], nid[("r", j)],
                                                         int(supply[i]), -int(round(g * 100)))
            arc_index[a] = (i, j)
        for j in receivers:
            smcf.add_arc_with_capacity_and_unit_cost(nid[("r", j)], sink, int(capacity[j]), 0)
        for i in senders:  # 'dump' arc: leaving surplus where it is costs nothing
            smcf.add_arc_with_capacity_and_unit_cost(nid[("s", i)], sink, int(supply[i]), 0)
            smcf.set_node_supply(nid[("s", i)], int(supply[i]))
        smcf.set_node_supply(sink, -total)
        status = smcf.solve()
        if status != smcf.OPTIMAL:
            raise RuntimeError(f"OR-Tools min-cost flow failed with status {status}")
        out = {}
        for a, (i, j) in arc_index.items():
            f = smcf.flow(a)
            if f > 0:
                out[(i, j)] = int(f)
        return out

    import networkx as nx
    G = nx.DiGraph()
    G.add_node("sink", demand=total)
    for i in senders:
        G.add_node(("s", i), demand=-int(supply[i]))
        G.add_edge(("s", i), "sink", capacity=int(supply[i]), weight=0)
    for j in receivers:
        G.add_edge(("r", j), "sink", capacity=int(capacity[j]), weight=0)
    for i, j, g in arcs:
        G.add_edge(("s", i), ("r", j), capacity=int(supply[i]), weight=-int(round(g * 100)))
    flow = nx.min_cost_flow(G)
    out = {}
    for i, j, _ in arcs:
        f = flow[("s", i)].get(("r", j), 0)
        if f > 0:
            out[(i, j)] = int(f)
    return out


def sample_from_quantiles(q10, q50, q90, size, rng) -> np.ndarray:
    """
    Draw demand samples whose 10th/50th/90th percentiles match the forecast.
    Piecewise-linear inverse CDF through (0.1,q10), (0.5,q50), (0.9,q90), tails extended.
    q10/q50/q90: arrays of shape [H]; returns [size, H], clipped at 0.
    """
    q10, q50, q90 = (np.asarray(x, float) for x in (q10, q50, q90))
    q10 = np.minimum(q10, q50)
    q90 = np.maximum(q90, q50)
    u = rng.random((size, q50.shape[0]))
    lo_slope = (q50 - q10) / 0.4
    hi_slope = (q90 - q50) / 0.4
    x = np.where(u < 0.5, q50 - (0.5 - u) * lo_slope, q50 + (u - 0.5) * hi_slope)
    return np.clip(x, 0, None)
