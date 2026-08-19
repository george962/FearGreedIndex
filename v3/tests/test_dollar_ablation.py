from __future__ import annotations

import unittest
import pandas as pd
from v3.evaluation.dollar_ablation import dollar_family_decision


class DollarAblationTests(unittest.TestCase):
    @staticmethod
    def _lanes() -> pd.DataFrame:
        rows = []
        for experiment_id, model_name in (("EXP-003", "gb"), ("EXP-004", "rf")):
            for target_type in ("classification", "return_regression", "drawdown_regression"):
                rows.append({"experiment_id": experiment_id, "model_name": model_name, "target_type": target_type, "robust_improvement": target_type != "drawdown_regression"})
        return pd.DataFrame(rows)

    def test_two_robust_lanes_in_both_models_retains_dollar(self) -> None:
        decision = dollar_family_decision(self._lanes())
        self.assertTrue(decision["retain_dollar"])
        self.assertEqual(decision["robust_lane_count"], 2)

    def test_single_model_failure_breaks_lane(self) -> None:
        lanes = self._lanes()
        lanes.loc[(lanes["experiment_id"] == "EXP-004") & (lanes["target_type"] == "return_regression"), "robust_improvement"] = False
        decision = dollar_family_decision(lanes)
        self.assertFalse(decision["retain_dollar"])
        self.assertEqual(decision["robust_lane_count"], 1)

    def test_missing_full_model_is_rejected(self) -> None:
        lanes = self._lanes().loc[lambda x: x["experiment_id"] != "EXP-004"]
        with self.assertRaisesRegex(ValueError, "Missing full-interface"):
            dollar_family_decision(lanes)


if __name__ == "__main__":
    unittest.main()
