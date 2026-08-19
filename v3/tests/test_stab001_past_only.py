#!/usr/bin/env python3
"""Tests for STAB-001 past-only stability selection."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from v3.evaluation.stab001_past_only import (
    empirical_percentile,
    feature_family,
    score_to_probability,
    select_stable_features,
)


class Stab001PastOnlyTests(unittest.TestCase):
    @staticmethod
    def _frame(signs: list[int], weak: bool = False) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        date = pd.Timestamp("2020-01-01")
        for block_index, sign in enumerate(signs):
            for i in range(120):
                target = i % 2
                if weak:
                    value = float((i * 37 + block_index * 11) % 101)
                elif sign > 0:
                    value = float(target * 100 + i * 0.001)
                else:
                    value = float((1 - target) * 100 + i * 0.001)
                rows.append(
                    {
                        "decision_date": date,
                        "favorable_entry_20d": target,
                        "feature_x": value,
                    }
                )
                date += pd.Timedelta(days=1)
        return pd.DataFrame(rows)

    def test_consistent_feature_is_selected(self) -> None:
        selected, diagnostics = select_stable_features(
            self._frame([1, 1, 1, 1]), ["feature_x"]
        )
        self.assertEqual(len(selected), 1)
        self.assertTrue(diagnostics[0]["selected"])
        self.assertEqual(diagnostics[0]["majority_sign"], 1)
        self.assertGreaterEqual(diagnostics[0]["sign_consistency"], 0.75)
        self.assertGreaterEqual(diagnostics[0]["median_abs_spearman"], 0.05)

    def test_recent_sign_reversal_rejects_feature(self) -> None:
        selected, diagnostics = select_stable_features(
            self._frame([1, 1, 1, -1]), ["feature_x"]
        )
        self.assertEqual(selected, [])
        self.assertFalse(diagnostics[0]["recent_block_agrees"])
        self.assertFalse(diagnostics[0]["selected"])

    def test_three_of_four_consistent_with_recent_agreement_is_allowed(self) -> None:
        selected, diagnostics = select_stable_features(
            self._frame([-1, 1, 1, 1]), ["feature_x"]
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(diagnostics[0]["majority_sign"], 1)
        self.assertAlmostEqual(diagnostics[0]["sign_consistency"], 0.75)

    def test_weak_association_is_rejected(self) -> None:
        selected, diagnostics = select_stable_features(
            self._frame([1, 1, 1, 1], weak=True), ["feature_x"]
        )
        self.assertEqual(selected, [])
        self.assertFalse(diagnostics[0]["selected"])

    def test_feature_family_contract(self) -> None:
        self.assertEqual(feature_family("fear_greed"), "fear_greed")
        self.assertEqual(feature_family("fg_change_5"), "fear_greed")
        self.assertEqual(feature_family("treasury_10y_level"), "treasury")
        self.assertEqual(feature_family("spx_realized_vol_20"), "spx_interaction")
        self.assertEqual(feature_family("interaction_fg_x_vol_20"), "spx_interaction")

    def test_empirical_percentile_uses_training_distribution_only(self) -> None:
        train = pd.Series([1.0, 2.0, 3.0, 4.0])
        values = pd.Series([0.0, 2.0, 10.0, np.nan])
        result = empirical_percentile(train, values)
        self.assertTrue(np.allclose(result, [0.0, 0.5, 1.0, 0.5]))

    def test_score_probability_is_monotonic_training_cdf(self) -> None:
        train_scores = np.array([-0.5, -0.1, 0.2, 0.4])
        scores = np.array([-1.0, -0.1, 0.3, 1.0])
        result = score_to_probability(train_scores, scores)
        self.assertTrue(np.allclose(result, [0.0, 0.5, 0.75, 1.0]))
        self.assertTrue(np.all(np.diff(result) >= 0.0))


if __name__ == "__main__":
    unittest.main()
