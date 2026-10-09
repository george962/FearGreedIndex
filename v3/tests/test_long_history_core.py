from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from v3.features.build_long_history_core import FEATURE_VERSION, build_feature_frame
from v3.labels.build_long_history_labels import add_favorable_entry_target

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "v3" / "reports" / "feature_registry_long_history_core.json"


class LongHistoryCoreTests(unittest.TestCase):
    def test_registry_is_fixed_and_excludes_fear_greed(self) -> None:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        self.assertEqual(registry["feature_set_version"], FEATURE_VERSION)
        names = [item["name"] for item in registry["features"]]
        self.assertEqual(len(names), 47)
        self.assertEqual(len(set(names)), 47)
        self.assertFalse(any("fear_greed" in name.lower() for name in names))
        self.assertFalse(registry["cnn_fear_greed_included"])
        self.assertFalse(registry["interactions_included"])

    def test_feature_builder_is_past_only(self) -> None:
        rows = 320
        dates = pd.bdate_range("2020-01-02", periods=rows)
        close = pd.Series(np.linspace(100.0, 180.0, rows))
        aligned = pd.DataFrame(
            {
                "decision_date": dates,
                "open": close.to_numpy() - 0.25,
                "high": close.to_numpy() + 1.0,
                "low": close.to_numpy() - 1.0,
                "close": close.to_numpy(),
                "adj_close": close.to_numpy(),
                "volume": np.arange(rows) + 1000,
                "vix_observation_date": dates,
                "vix_close": np.linspace(30.0, 15.0, rows),
                "treasury_observation_date": dates,
                "dgs2": np.linspace(1.0, 3.0, rows),
                "dgs10": np.linspace(2.0, 4.0, rows),
            }
        )
        features = build_feature_frame(aligned)
        last = features.iloc[-1]
        self.assertAlmostEqual(last["spx_return_1"], close.iloc[-1] / close.iloc[-2] - 1.0)
        self.assertAlmostEqual(last["treasury_10y_2y_slope"], 1.0)
        self.assertGreaterEqual(last["vix_percentile_252"], 0.0)
        self.assertLessEqual(last["vix_percentile_252"], 1.0)

        modified = aligned.copy()
        modified.loc[modified.index[-1], "close"] = 9999.0
        rebuilt = build_feature_frame(modified)
        pd.testing.assert_series_equal(
            features.iloc[-2].drop(labels=["close"], errors="ignore"),
            rebuilt.iloc[-2].drop(labels=["close"], errors="ignore"),
            check_names=False,
        )

    def test_favorable_entry_target_matches_exp006_contract(self) -> None:
        labels = pd.DataFrame(
            {
                "forward_return_20d": [0.03, 0.03, 0.01, np.nan],
                "max_drawdown_20d": [-0.02, -0.05, -0.01, np.nan],
                "_forward_20d_known_date": pd.to_datetime(
                    ["2020-02-01", "2020-02-01", "2020-02-01", None]
                ),
            }
        )
        result = add_favorable_entry_target(labels)
        self.assertTrue(bool(result.loc[0, "favorable_entry_20d"]))
        self.assertFalse(bool(result.loc[1, "favorable_entry_20d"]))
        self.assertFalse(bool(result.loc[2, "favorable_entry_20d"]))
        self.assertTrue(pd.isna(result.loc[3, "favorable_entry_20d"]))


if __name__ == "__main__":
    unittest.main()
