from __future__ import annotations

import unittest

import pandas as pd

from v3.evaluation.exp006_opportunity import (
    add_opportunity_targets,
    eligible_training_mask,
    sample_hash,
    summarize_model,
)


class Exp006OpportunityTests(unittest.TestCase):
    def test_target_boundaries_and_state_precedence(self) -> None:
        frame = pd.DataFrame(
            {
                "forward_return_20d": [-0.02, -0.019, 0.02, 0.049, 0.05, 0.03],
                "max_drawdown_20d": [-0.01, -0.01, -0.049, -0.01, -0.01, -0.05],
                "_forward_20d_known_date": pd.to_datetime(["2024-02-01"] * 6),
            }
        )
        result = add_opportunity_targets(frame)
        self.assertEqual(
            result["opportunity_state_20d"].tolist(),
            ["BAD", "NORMAL", "GOOD", "GOOD", "EXCELLENT", "BAD"],
        )
        self.assertEqual(
            result["favorable_entry_20d"].astype(bool).tolist(),
            [False, False, True, True, True, False],
        )

    def test_immature_target_stays_missing(self) -> None:
        frame = pd.DataFrame(
            {
                "forward_return_20d": [0.04, None],
                "max_drawdown_20d": [-0.01, None],
                "_forward_20d_known_date": pd.to_datetime(["2024-02-01", None]),
            }
        )
        result = add_opportunity_targets(frame)
        self.assertEqual(result.loc[0, "opportunity_state_20d"], "GOOD")
        self.assertTrue(pd.isna(result.loc[1, "opportunity_state_20d"]))
        self.assertTrue(pd.isna(result.loc[1, "favorable_entry_20d"]))

    def test_training_eligibility_uses_outcome_known_date(self) -> None:
        frame = pd.DataFrame(
            {
                "decision_date": pd.to_datetime(["2023-11-01", "2023-12-20", "2024-01-02"]),
                "forward_return_20d": [0.03, 0.03, 0.03],
                "max_drawdown_20d": [-0.01, -0.01, -0.01],
                "_forward_20d_known_date": pd.to_datetime(["2023-11-30", "2024-01-22", "2024-02-01"]),
            }
        )
        result = add_opportunity_targets(frame)
        mask = eligible_training_mask(result, "2023-12-31")
        self.assertEqual(mask.tolist(), [True, False, False])

    def test_viability_gate_requires_both_brier_and_auc_robustness(self) -> None:
        passing = pd.DataFrame(
            {
                "relative_brier_improvement": [0.03, 0.02, -0.005],
                "roc_auc": [0.56, 0.54, 0.51],
                "brier_score": [0.20, 0.21, 0.22],
                "expected_calibration_error": [0.08, 0.09, 0.10],
            }
        )
        self.assertTrue(summarize_model(passing)["viability_gate_pass"])

        failing = passing.copy()
        failing["relative_brier_improvement"] = [-0.01, 0.02, -0.005]
        self.assertFalse(summarize_model(failing)["viability_gate_pass"])

    def test_sample_hash_is_order_sensitive_and_deterministic(self) -> None:
        dates = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03"]))
        self.assertEqual(sample_hash(dates), sample_hash(dates.copy()))
        self.assertNotEqual(sample_hash(dates), sample_hash(dates.iloc[::-1].reset_index(drop=True)))


if __name__ == "__main__":
    unittest.main()
