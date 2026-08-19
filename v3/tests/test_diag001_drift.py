from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from v3.diagnostics.diag001_drift import (
    adjacent_sign_transitions,
    association_metrics,
    association_sign,
    distribution_metrics,
    pooled_standardized_mean_difference,
    sign_reversal,
)


class Diag001DriftTests(unittest.TestCase):
    def test_pooled_smd_preserves_direction(self) -> None:
        train = np.array([0.0, 1.0, 2.0, 3.0])
        test = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertGreater(pooled_standardized_mean_difference(train, test), 0.0)
        self.assertLess(pooled_standardized_mean_difference(test, train), 0.0)
        self.assertTrue(
            np.isnan(
                pooled_standardized_mean_difference(
                    np.array([1.0, 1.0]), np.array([1.0, 1.0])
                )
            )
        )

    def test_distribution_metrics_include_missingness_and_shift(self) -> None:
        train = pd.Series([0.0, 1.0, 2.0, np.nan])
        test = pd.Series([1.0, 2.0, 3.0, 4.0])
        result = distribution_metrics(train, test)
        self.assertAlmostEqual(result["train_missing_rate"], 0.25)
        self.assertAlmostEqual(result["test_missing_rate"], 0.0)
        self.assertAlmostEqual(result["missing_rate_delta"], -0.25)
        self.assertGreater(result["standardized_mean_difference"], 0.0)
        self.assertTrue(np.isfinite(result["ks_statistic"]))

    def test_association_uses_raw_feature_orientation(self) -> None:
        target = [0] * 10 + [1] * 10
        upward = pd.DataFrame(
            {
                "feature": list(range(20)),
                "favorable_entry_20d": target,
            }
        )
        upward_result = association_metrics(upward, "feature")
        self.assertTrue(upward_result["available"])
        self.assertAlmostEqual(upward_result["roc_auc_raw_upward"], 1.0)
        self.assertEqual(upward_result["sign"], "POSITIVE")

        downward = upward.copy()
        downward["feature"] = list(reversed(range(20)))
        downward_result = association_metrics(downward, "feature")
        self.assertAlmostEqual(downward_result["roc_auc_raw_upward"], 0.0)
        self.assertEqual(downward_result["sign"], "NEGATIVE")

    def test_association_requires_minimum_rows_and_two_classes(self) -> None:
        short = pd.DataFrame(
            {
                "feature": list(range(19)),
                "favorable_entry_20d": [0, 1] * 9 + [0],
            }
        )
        result = association_metrics(short, "feature")
        self.assertFalse(result["available"])
        self.assertEqual(result["sign"], "UNAVAILABLE")

        one_class = pd.DataFrame(
            {"feature": list(range(20)), "favorable_entry_20d": [1] * 20}
        )
        result_one_class = association_metrics(one_class, "feature")
        self.assertFalse(result_one_class["available"])

    def test_sign_rules_do_not_hide_reversals(self) -> None:
        self.assertEqual(association_sign(0.1), "POSITIVE")
        self.assertEqual(association_sign(-0.1), "NEGATIVE")
        self.assertEqual(association_sign(0.0), "ZERO")
        self.assertEqual(association_sign(float("nan")), "UNAVAILABLE")
        self.assertTrue(sign_reversal("POSITIVE", "NEGATIVE"))
        self.assertFalse(sign_reversal("POSITIVE", "POSITIVE"))
        self.assertIsNone(sign_reversal("ZERO", "POSITIVE"))
        self.assertEqual(
            adjacent_sign_transitions(
                ["POSITIVE", "UNAVAILABLE", "NEGATIVE", "NEGATIVE"]
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
