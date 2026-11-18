"""
The nightly job: runs steps 1-8 end to end and writes everything the screen shows.

WHAT THIS FILE DOES (say this to a judge):
  "One command runs the whole pipeline: clean the data, fill in hidden demand,
   forecast ranges, replay the last four weeks for four policies on identical demand,
   and write the scoreboard, the demand-truth test and every store's action list as
   JSON files. The web screen just reads those files."

Run:  python -m unwaste.pipeline            (real data)
      python -m unwaste.pipeline --synthetic (stand-in data, for testing)
Owner: Aniz (outputs), everyone (run it).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .backends import backend_names
from .config import Config, default_config
from .data import build_panel
from .forecast import fit_forecast, to_matrix
from .impute import correct_demand, demand_truth_test, example_day
from .replay import POLICIES, run_policy


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def run(cfg: Config, sweep: bool = False) -> dict:
    t_start = time.time()
    out = Path(cfg.out_dir)
    print(f"[run] backends: {backend_names()}")

    panel = build_panel(cfg)                                            # step 1
    corrected_hourly, demand = correct_demand(panel, cfg)               # step 2
    panel.daily["demand"] = demand.to_numpy()
    truth = demand_truth_test(panel, cfg)
    truth["example_day"] = example_day(panel, corrected_hourly)
    truth["source"] = panel.source
    _write(out / "demand_truth.json", truth)

    Y_raw, dates = to_matrix(panel.daily, "sales")
    Y_cor, _ = to_matrix(panel.daily, "demand")
    meta = panel.meta.sort_values("series_id").reset_index(drop=True)

    F_cor = fit_forecast(Y_cor, panel.daily, cfg, " corrected")         # step 3
    F_raw = fit_forecast(Y_raw, panel.daily, cfg, " raw")

    D = np.round(Y_cor).astype(int)                                     # truth for the replay
    results, actions = {}, []
    for key, pol in POLICIES.items():                                   # steps 4-8
        F = F_raw if key == "forecast_raw" else F_cor
        t0 = time.time()
        led = run_policy(pol, D, Y_raw, F, meta, cfg, log_actions=(key == "unwaste"))
        results[key] = {"label": pol.name, **led.summary()}
        print(f"[replay] {key:20s} waste {results[key]['waste_pct_of_received']:5.1f}%  "
              f"service {results[key]['service_level_pct']:5.1f}%  "
              f"net ₹{results[key]['net_margin_rs']:>10,}  ({time.time() - t0:.0f}s)")
        if key == "unwaste":
            actions = led.actions

    sweep_res = []
    if sweep:
        for c in (2.0, 5.0, 10.0):
            cfg_c = default_config(**{**cfg.to_dict(), "transfer_cost_per_unit": c})
            led = run_policy(POLICIES["unwaste"], D, Y_raw, F_cor, meta, cfg_c)
            sweep_res.append({"transfer_cost_rs": c, **led.summary()})
            print(f"[sweep] transfer ₹{c:>4}: net ₹{sweep_res[-1]['net_margin_rs']:,}, "
                  f"transfers {sweep_res[-1]['transfer_units']}")

    b, u = results["baseline"], results["unwaste"]
    headline = {
        "waste_reduction_pct": round(100 * (b["waste_units"] - u["waste_units"]) / max(b["waste_units"], 1), 1),
        "net_margin_gain_rs": u["net_margin_rs"] - b["net_margin_rs"],
        "service_change_pts": round(u["service_level_pct"] - b["service_level_pct"], 2),
    }
    t0 = F_cor["train_end"] + 1
    scoreboard = {
        "source": panel.source, "backends": backend_names(),
        "stores": int(meta["store_id"].nunique()), "items": int(meta["product_id"].nunique()),
        "series": int(len(meta)),
        "replay_window": [str(dates[t0].date()), str(dates[-1].date())],
        "train_window": [str(dates[0].date()), str(dates[t0 - 1].date())],
        "forecast": {"corrected": F_cor["metrics"], "raw": F_raw["metrics"]},
        "policies": results, "headline": headline, "sweep": sweep_res,
        "config": cfg.to_dict(), "runtime_s": round(time.time() - t_start),
    }
    _write(out / "scoreboard.json", scoreboard)

    # manager screen: actions from the replay, newest night first
    store_names = {str(s): f"Store {i + 101}" for i, s in enumerate(sorted(meta["store_id"].astype(str).unique()))}
    for a in actions:
        a["date"] = str(dates[a["night"]].date())
        a["sellby_date"] = str((dates[0] + pd.Timedelta(days=int(a["sellby"]))).date())
        a["store_name"] = store_names[a["store_id"]]
        a["to_store_name"] = store_names.get(a["to_store"]) if a["to_store"] else None
    _write(out / "actions.json", {"source": panel.source, "nights": sorted({a["date"] for a in actions}),
                                  "actions": actions})
    print(f"[run] wrote {out}/scoreboard.json, actions.json, demand_truth.json "
          f"in {time.time() - t_start:.0f}s")
    print(f"[headline] waste down {headline['waste_reduction_pct']:.1f}% vs baseline, "
          f"net margin {headline['net_margin_gain_rs']:+,} ₹, service {headline['service_change_pts']:+.2f} pts")
    return scoreboard


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Unwaste nightly pipeline + replay")
    ap.add_argument("--synthetic", action="store_true", help="use stand-in data (testing only)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--city", type=int, default=None)
    ap.add_argument("--stores", type=int, default=None)
    ap.add_argument("--products", type=int, default=None)
    ap.add_argument("--paths", type=int, default=None, help="Monte Carlo paths")
    ap.add_argument("--transfer-cost", type=float, default=None)
    ap.add_argument("--sweep", action="store_true", help="robustness sweep on transfer cost")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = default_config(use_synthetic=a.synthetic, data_dir=a.data_dir, city_id=a.city,
                         max_stores=a.stores, max_products=a.products, mc_paths=a.paths,
                         transfer_cost_per_unit=a.transfer_cost, out_dir=a.out)
    run(cfg, sweep=a.sweep)


if __name__ == "__main__":
    main()
