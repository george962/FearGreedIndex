from __future__ import annotations

import unittest

import pandas as pd

from v3.evaluation.retained_combined_ablation import family_decision
from v3.features.build_retained_features import COMBINED_FEATURES, build_registry


class RetainedCombinedTests(unittest.TestCase):
    def test_registry_has_expected_76_unique_features(self) -> None:
        registry = build_registry()
        names = [item["name"] for item in registry["features"]]
        self.assertEqual(registry["version"], "v3-features-006-retained-combined")
        self.assertEqual(len(COMBINED_FEATURES), 35)
        self.assertEqual(len(names), 76)
        self.assertEqual(len(names), len(set(names)))

    @staticmethod
    def _lanes() -> pd.DataFrame:
        rows = []
        for experiment_id, model_name in (("EXP-003", "gb"), ("EXP-004", "rf")):
            for target_type in ("classification", "return_regression", "drawdown_regression"):
                rows.append({"experiment_id": experiment_id, "model_name": model_name, "target_type": target_type, "robust_improvement": target_type != "drawdown_regression"})
        return pd.DataFrame(rows)

    def test_two_robust_lanes_retains_combined_set(self) -> None:
        decision = family_decision(self._lanes())
        self.assertTrue(decision["retain_combined"])
        self.assertEqual(decision["robust_lane_count"], 2)

    def test_failure_in_one_full_model_breaks_lane(self) -> None:
        lanes = self._lanes()
        lanes.loc[(lanes["experiment_id"] == "EXP-004") & (lanes["target_type"] == "return_regression"), "robust_improvement"] = False
        decision = family_decision(lanes)
        self.assertFalse(decision["retain_combined"])
        self.assertEqual(decision["robust_lane_count"], 1)


if __name__ == "__main__":
    unittest.main()
