"""End-to-end smoke test on tiny stand-in data (about 10 seconds)."""
import json

from unwaste.config import default_config
from unwaste.pipeline import run


def test_pipeline_runs_and_writes_outputs(tmp_path):
    cfg = default_config(use_synthetic=True, max_stores=4, max_products=5, mc_paths=60,
                         n_estimators=30, out_dir=str(tmp_path))
    sb = run(cfg)
    for name in ("scoreboard", "actions", "demand_truth"):
        assert (tmp_path / f"{name}.json").exists()
    assert sb["policies"]["unwaste"]["shelf_life_breaches"] == 0
    assert json.loads((tmp_path / "scoreboard.json").read_text())["source"] == "SYNTHETIC"
