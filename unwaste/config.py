"""
Step 0 · Config: every knob in one place.

  "All assumptions live here: prices, transfer cost, salvage value, lead time,
   service floor, which city we run on. Nothing is hidden inside the code, so we
   can change one number and rerun the whole replay."

"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Config:
    # ---------- data ----------
    data_dir: Path = Path("data/raw/frn50k")      # where the parquet files live
    use_synthetic: bool = False                  # True = stand-in data with the same shape
    city_id: int | None = None                   # None = pick the city with most stores
    max_stores: int = 20                         # keep the demo fast: one city, 20 stores
    max_products: int = 40                       # top-selling perishables in that city
    open_hours: tuple[int, int] = (6, 22)        # store hours used for demand (6:00-22:00)
    target_mean_daily_units: float = 10.0        # rescale anonymised sales to "cups a day"

    # ---------- money (₹) ----------
    price_min: float = 30.0                      # item prices are assumed per product
    price_max: float = 120.0
    cost_ratio: float = 0.70                     # purchase cost = 70% of price
    salvage_per_unit: float = 1.0                # what a dumped unit still fetches (₹)
    transfer_cost_per_unit: float = 2.0          # handling + transport per unit moved (₹)

    # ---------- shelf life ----------
    shelf_life_choices: tuple[int, ...] = (4, 5, 7, 10)   # days, from Indian pack labels
    fssai_min_remaining_frac: float = 0.30       # >=30% shelf life must remain at delivery

    # ---------- operations ----------
    lead_time_days: int = 1                      # order today, arrives tomorrow
    review_period_days: int = 3                  # stores reorder every 3 days
    transfer_days: int = 1                       # moved stock arrives next day
    test_days: int = 28                          # last 4 weeks are the replay (time split)

    # ---------- models ----------
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    horizon_days: int = 7                        # forecast the next 7 days
    mc_paths: int = 300                          # Monte Carlo replays per decision
    n_estimators: int = 200                      # boosting rounds per model
    use_weather: bool = True                     # rain, temperature, humidity, wind as forecast inputs
    relative_target: bool = False                # predict demand relative to the recent 14-day level
    seed: int = 7

    # ---------- policy ----------
    service_floor: float = 0.60                  # never order below the 60th pct of uncovered demand
    lost_sale_penalty_frac: float = 0.50         # a lost sale costs 50% of price in goodwill
    risk_threshold: float = 0.6                  # act only if P(leftover) >= 60%
    min_units_to_act: int = 2                    # ignore tiny leftovers
    markdown_frac: float = 0.15                  # early markdown depth
    late_markdown_frac: float = 0.30             # baseline: last-day clearance
    markdown_uplift_per_10pct: float = 0.18      # +18% demand for every 10% discount
    baseline_safety_factor: float = 1.15         # baseline orders 15% above recent average

    # ---------- outputs ----------
    out_dir: Path = Path("web/data")

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


def default_config(**overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            raise KeyError(f"Unknown config key: {k}")
        cur = getattr(cfg, k)
        setattr(cfg, k, Path(v) if isinstance(cur, Path) else v)
    return cfg
