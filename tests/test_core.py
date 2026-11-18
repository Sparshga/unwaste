"""Fast checks that the core maths does what the slides say. Run: pytest -q"""
import numpy as np

from unwaste.backends import max_gain_transfers, sample_from_quantiles
from unwaste.config import default_config
from unwaste.decide import newsvendor_order, transfer_gain_per_unit
from unwaste.risk import Batch, simulate_fefo


def test_slide_example_transfer_gain_is_1410():
    cfg = default_config()                              # ₹2 transfer, ₹1 salvage
    g = transfer_gain_per_unit(50.0, cfg)               # 50 - 2 - 1 = 47 per cup
    moves = max_gain_transfers({"store114": 30}, {"store118": 32}, {("store114", "store118"): g})
    assert moves == {("store114", "store118"): 30}      # x = min(30, 32)
    assert moves[("store114", "store118")] * g == 1410


def test_receiver_capacity_is_respected():
    moves = max_gain_transfers({"a": 30}, {"b": 10, "c": 5}, {("a", "b"): 40, ("a", "c"): 30})
    assert moves[("a", "b")] == 10 and moves[("a", "c")] == 5


def test_best_receiver_gets_stock_first():
    moves = max_gain_transfers({"a": 10}, {"b": 10, "c": 10}, {("a", "b"): 10, ("a", "c"): 40})
    assert moves == {("a", "c"): 10}


def test_unprofitable_moves_never_happen():
    assert max_gain_transfers({"a": 10}, {"b": 10}, {("a", "b"): -3}) == {}


def test_fefo_sells_oldest_first_and_counts_leftover():
    batches = [Batch(40, sellby=3, arrival=0), Batch(60, sellby=8, arrival=0)]
    demand = np.full((1, 9), 10.0)                       # 10 a day, days 0..8
    sim = simulate_fefo(batches, demand, start_day=0)
    assert sim["leftover"][0, 0] == 0                    # old batch: 40 sold in 4 days (0..3)
    assert sim["leftover"][0, 1] == 10                   # new batch: 50 sold days 4..8, 10 left


def test_slide_example_curd_batch():
    # 100 cups, 10 a day, sell-by on day 7 (days 1..7 = 7 selling days) -> 30 left
    sim = simulate_fefo([Batch(100, sellby=7, arrival=1)], np.full((1, 7), 10.0), start_day=1)
    assert sim["leftover"][0, 0] == 30


def test_quantile_sampler_matches_forecast():
    rng = np.random.default_rng(0)
    x = sample_from_quantiles([8.0], [10.0], [14.0], 20000, rng)[:, 0]
    assert abs(np.quantile(x, 0.5) - 10) < 0.3
    assert abs(np.quantile(x, 0.9) - 14) < 0.4


def test_newsvendor_orders_nothing_when_shelf_is_full():
    cfg = default_config(mc_paths=500)
    rng = np.random.default_rng(1)
    full = [Batch(200, sellby=10, arrival=0)]
    q = newsvendor_order(full, *[np.full(7, v) for v in (8.0, 10.0, 12.0)], arrival=1,
                         sellable_days=5, price=50.0, cfg=cfg, rng=rng)
    assert q == 0


def test_newsvendor_orders_roughly_window_demand_when_empty():
    cfg = default_config(mc_paths=500)
    rng = np.random.default_rng(1)
    q = newsvendor_order([], *[np.full(7, v) for v in (8.0, 10.0, 12.0)], arrival=1,
                         sellable_days=5, price=50.0, cfg=cfg, rng=rng)
    assert 25 <= q <= 55                                 # ~3 days of cover, not the whole week
