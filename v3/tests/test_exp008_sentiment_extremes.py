from __future__ import annotations

import unittest

import pandas as pd

from v3.evaluation.exp008_sentiment_extremes import (
    EXTREME_FEAR,
    EXTREME_GREED,
    NEUTRAL_RANGE,
    add_sentiment_state,
    ordered_as_hypothesized,
    state_training_evidence,
    summarize_viability,
)


class Exp008SentimentExtremesTests(unittest.TestCase):
    def test_state_boundaries_are_frozen(self) -> None:
        frame = pd.DataFrame({"fear_greed": [0, 25, 25.1, 50, 74.9, 75, 100, None]})
        result = add_sentiment_state(frame)
        self.assertEqual(
            result["exp008_sentiment_state"].tolist()[:7],
            [
                EXTREME_FEAR,
                EXTREME_FEAR,
                NEUTRAL_RANGE,
                NEUTRAL_RANGE,
                NEUTRAL_RANGE,
                EXTREME_GREED,
                EXTREME_GREED,
            ],
        )
        self.assertTrue(pd.isna(result.loc[7, "exp008_sentiment_state"]))

    def test_hypothesized_order_is_strict(self) -> None:
        self.assertTrue(
            ordered_as_hypothesized(
                {EXTREME_FEAR: 0.7, NEUTRAL_RANGE: 0.5, EXTREME_GREED: 0.3}
            )
        )
        self.assertFalse(
            ordered_as_hypothesized(
                {EXTREME_FEAR: 0.5, NEUTRAL_RANGE: 0.5, EXTREME_GREED: 0.3}
            )
        )

    def test_support_requires_rows_and_both_classes(self) -> None:
        rows = []
        for state in (EXTREME_FEAR, NEUTRAL_RANGE, EXTREME_GREED):
            for index in range(50):
                rows.append(
                    {
                        "exp008_sentiment_state": state,
                        "favorable_entry_20d": bool(index % 2),
                    }
                )
        evidence, passed = state_training_evidence(pd.DataFrame(rows))
        self.assertTrue(passed)
        self.assertEqual(evidence[EXTREME_FEAR]["training_rows"], 50)

        bad = pd.DataFrame(rows[:-1])
        _, passed_bad = state_training_evidence(bad)
        self.assertFalse(passed_bad)

    def test_viability_gate_requires_all_registered_conditions(self) -> None:
        metrics = pd.DataFrame(
            {
                "relative_brier_improvement": [0.03, 0.02, -0.01],
                "roc_auc": [0.56, 0.55, 0.48],
            }
        )
        result = summarize_viability(
            metrics,
            full_coverage=True,
            sample_hashes_match=True,
            support_pass=True,
            ordered_folds=2,
        )
        self.assertTrue(result["viability_gate_pass"])

        result_bad_order = summarize_viability(
            metrics,
            full_coverage=True,
            sample_hashes_match=True,
            support_pass=True,
            ordered_folds=1,
        )
        self.assertFalse(result_bad_order["viability_gate_pass"])

        result_bad_min_auc = summarize_viability(
            pd.DataFrame(
                {
                    "relative_brier_improvement": [0.03, 0.02, 0.01],
                    "roc_auc": [0.60, 0.58, 0.44],
                }
            ),
            full_coverage=True,
            sample_hashes_match=True,
            support_pass=True,
            ordered_folds=3,
        )
        self.assertFalse(result_bad_min_auc["viability_gate_pass"])


if __name__ == "__main__":
    unittest.main()
