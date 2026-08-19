from __future__ import annotations

import unittest

import pandas as pd

from v3.evaluation.relative_strength_ablation import relative_strength_family_decision


class RelativeStrengthAblationTests(unittest.TestCase):
    @staticmethod
    def _lanes() -> pd.DataFrame:
        rows = []
        for experiment_id, model_name in (("EXP-003", "gb"), ("EXP-004", "rf")):
            for target_type in (
                "classification",
                "return_regression",
                "drawdown_regression",
            ):
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "model_name": model_name,
                        "target_type": target_type,
                        "robust_improvement": target_type != "drawdown_regression",
                    }
                )
        return pd.DataFrame(rows)

    def test_two_robust_lanes_in_both_models_retains_family(self) -> None:
        decision = relative_strength_family_decision(self._lanes())
        self.assertTrue(decision["retain_relative_strength"])
        self.assertEqual(decision["robust_lane_count"], 2)

    def test_one_model_failure_breaks_family_lane(self) -> None:
        lanes = self._lanes()
        mask = (
            lanes["experiment_id"].eq("EXP-004")
            & lanes["target_type"].eq("return_regression")
        )
        lanes.loc[mask, "robust_improvement"] = False
        decision = relative_strength_family_decision(lanes)
        self.assertFalse(decision["retain_relative_strength"])
        self.assertEqual(decision["robust_lane_count"], 1)

    def test_missing_full_model_is_rejected(self) -> None:
        lanes = self._lanes()
        lanes = lanes.loc[~lanes["experiment_id"].eq("EXP-004")]
        with self.assertRaisesRegex(ValueError, "Missing full-interface"):
            relative_strength_family_decision(lanes)


if __name__ == "__main__":
    unittest.main()
