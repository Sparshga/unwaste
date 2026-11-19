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


def test_forecast_baselines_and_errors():
    from unwaste.proof import baseline_forecasts, forecast_errors
    Y = np.tile(np.array([10.0, 12, 14, 10, 12, 14, 20] * 4), (2, 1))   # weekly pattern, 28 days
    B = baseline_forecasts(Y, train_end=20, H=7)
    sn = forecast_errors(B["Same weekday last week"], Y, 20, 7)
    assert sn["next_day"] == 0.0 and sn["week"] == 0.0                   # perfect on a pure weekly pattern
    assert forecast_errors(B["Moving average (7 days)"], Y, 20, 7)["week"] > 0


def test_open_meteo_and_agmarknet_loaders(tmp_path):
    import json
    import pandas as pd
    from unwaste.india import load_agmarknet, load_open_meteo
    (tmp_path / "weather_x.json").write_text(json.dumps({"daily": {
        "time": ["2024-06-01", "2024-06-02"], "temperature_2m_mean": [35.0, 30.0],
        "precipitation_sum": [0.0, 25.0], "relative_humidity_2m_mean": [30, 80],
        "wind_speed_10m_mean": [10.0, 20.0]}}))
    wx = load_open_meteo(tmp_path / "weather_x.json")
    assert list(wx.columns) == ["rain", "temp", "humidity", "wind"] and wx["rain"].iloc[1] == 25.0
    pd.DataFrame({"Arrival_Date": ["01/07/2023", "02/07/2023"], "Modal_Price": ["1,500", "4,500"]}) \
        .to_csv(tmp_path / "agmarknet_tomato.csv", index=False)
    agm = load_agmarknet(tmp_path)
    assert agm["Tomato"]["price"].tolist() == [1500.0, 4500.0]


def test_align_weather_shifts_year():
    import pandas as pd
    from unwaste.india import align_weather
    wx = pd.DataFrame({"rain": range(366)}, index=pd.date_range("2023-01-01", periods=366))
    dates = pd.date_range("2024-04-01", periods=97)
    out = align_weather(wx, dates)
    assert out.index.isin(dates).all() or out.index.isin(dates).sum() == 97


def test_india_dataset_small():
    from unwaste.india_data import festival_features, generate
    df = generate(end="2024-01-31", folder=__import__("pathlib").Path("does-not-exist"))
    assert len(df) == 10 * 32 * 31
    assert (df["sales"] <= df["demand_true"]).all()
    assert set(df["weather_source"]) == {"climate fill"}
    cal = festival_features(__import__("pandas").date_range("2024-01-10", "2024-01-14"))
    assert cal.loc["2024-01-13", "days_to_fest"] == 1 and cal.loc["2024-01-14", "holiday"] == 0.0


def test_newsvendor_review_window_orders_less():
    import numpy as np
    from dataclasses import replace
    from unwaste.config import default_config
    from unwaste.decide import newsvendor_order
    cfg = default_config(review_period_days=1)
    q = np.full(7, 20.0)
    rng = np.random.default_rng(0)
    whole = newsvendor_order([], q * 0.8, q, q * 1.2, 1, 3, 50.0, cfg, rng)
    short = newsvendor_order([], q * 0.8, q, q * 1.2, 1, 3, 50.0, replace(cfg, newsvendor_review_window=True), np.random.default_rng(0))
    assert short < whole
