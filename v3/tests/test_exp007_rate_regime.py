from __future__ import annotations

import unittest

import pandas as pd

from v3.evaluation.exp007_rate_regime import (
    FALLING,
    RISING,
    add_rate_regime,
    summarize_viability,
    validate_regime_training_support,
)


class Exp007RateRegimeTests(unittest.TestCase):
    def test_regime_boundary_is_frozen_at_zero(self) -> None:
        frame = pd.DataFrame({"treasury_10y_change_20": [0.1, 0.0, -0.1, None]})
        result = add_rate_regime(frame)
        self.assertEqual(result.loc[0, "exp007_rate_regime"], RISING)
        self.assertEqual(result.loc[1, "exp007_rate_regime"], FALLING)
        self.assertEqual(result.loc[2, "exp007_rate_regime"], FALLING)
        self.assertTrue(pd.isna(result.loc[3, "exp007_rate_regime"]))

    def test_training_support_fails_closed_below_minimum(self) -> None:
        frame = pd.DataFrame(
            {
                "exp007_rate_regime": [RISING] * 99 + [FALLING] * 101,
                "favorable_entry_20d": ([0, 1] * 49) + [0] + ([0, 1] * 50) + [0],
            }
        )
        with self.assertRaises(ValueError):
            validate_regime_training_support(frame)

    def test_training_support_requires_both_classes(self) -> None:
        frame = pd.DataFrame(
            {
                "exp007_rate_regime": [RISING] * 100 + [FALLING] * 100,
                "favorable_entry_20d": [1] * 100 + ([0, 1] * 50),
            }
        )
        with self.assertRaises(ValueError):
            validate_regime_training_support(frame)

    def test_viability_gate_rejects_severe_single_fold_reversal(self) -> None:
        metrics = pd.DataFrame(
            {
                "relative_brier_improvement": [0.05, 0.04, 0.01],
                "roc_auc": [0.61, 0.58, 0.44],
            }
        )
        result = summarize_viability(
            metrics,
            full_coverage=True,
            sample_hashes_match=True,
            support_pass=True,
        )
        self.assertFalse(result["viability_gate_pass"])
        self.assertEqual(result["minimum_fold_roc_auc"], 0.44)

    def test_viability_gate_requires_all_contracts(self) -> None:
        metrics = pd.DataFrame(
            {
                "relative_brier_improvement": [0.04, 0.03, -0.005],
                "roc_auc": [0.58, 0.56, 0.50],
            }
        )
        passing = summarize_viability(
            metrics,
            full_coverage=True,
            sample_hashes_match=True,
            support_pass=True,
        )
        self.assertTrue(passing["viability_gate_pass"])

        no_hash = summarize_viability(
            metrics,
            full_coverage=True,
            sample_hashes_match=False,
            support_pass=True,
        )
        self.assertFalse(no_hash["viability_gate_pass"])


if __name__ == "__main__":
    unittest.main()
