import unittest

import pandas as pd

from scripts.strategy_validation import (
    brier_score,
    calibrated_probability,
    evaluate_fold,
    fit_calibrator,
)


class StrategyValidationTests(unittest.TestCase):
    def test_calibrator_uses_training_rows_only(self):
        train = pd.DataFrame(
            {
                "action": ["BUY GRADUALLY"] * 30 + ["WAIT ON BUYING"] * 30,
                "market_regime": ["correction"] * 60,
                "timing_side": ["BUY"] * 30 + ["TRIM"] * 30,
                "forward_5d": [0.01] * 24
                + [-0.01] * 6
                + [0.01] * 9
                + [-0.01] * 21,
            }
        )
        calibrator = fit_calibrator(
            train,
            minimum_group_sample=20,
            shrinkage_strength=0.0,
        )

        buy_row = pd.Series(
            {
                "action": "BUY GRADUALLY",
                "market_regime": "correction",
                "timing_side": "BUY",
            }
        )
        wait_row = pd.Series(
            {
                "action": "WAIT ON BUYING",
                "market_regime": "correction",
                "timing_side": "TRIM",
            }
        )
        p_buy, _, _ = calibrated_probability(buy_row, calibrator)
        p_wait, _, _ = calibrated_probability(wait_row, calibrator)
        self.assertGreater(p_buy, p_wait)

    def test_brier_score(self):
        score = brier_score(
            pd.Series([1, 0]).to_numpy(),
            pd.Series([0.8, 0.2]).to_numpy(),
        )
        self.assertAlmostEqual(score, 0.04)

    def test_fold_respects_train_end(self):
        history = pd.DataFrame(
            {
                "decision_date": pd.date_range(
                    "2023-01-01",
                    periods=500,
                    freq="D",
                ).strftime("%Y-%m-%d"),
                "action": ["BUY GRADUALLY"] * 500,
                "market_regime": ["correction"] * 500,
                "timing_side": ["BUY"] * 500,
                "forward_5d": [0.01] * 500,
                "_forward_5d_known_date": (
                    pd.date_range("2023-01-06", periods=500, freq="D")
                    .strftime("%Y-%m-%d")
                ),
            }
        )
        summary, predicted = evaluate_fold(
            history,
            {
                "name": "test",
                "train_end": "2023-12-31",
                "test_start": "2024-01-01",
                "test_end": "2024-12-31",
            },
            minimum_group_sample=10,
            shrinkage_strength=5.0,
            minimum_test_rows=10,
            minimum_relative_brier_improvement=-1.0,
        )
        self.assertEqual(summary["train_rows"], 360)
        self.assertTrue(
            (pd.to_datetime(predicted["decision_date"]) >= pd.Timestamp("2024-01-01")).all()
        )

    def test_fold_excludes_training_outcome_that_matures_after_cutoff(self):
        history = pd.DataFrame(
            {
                "decision_date": ["2023-12-20", "2023-12-29", "2024-01-05"],
                "_forward_5d_known_date": [
                    "2023-12-27",
                    "2024-01-08",
                    "2024-01-12",
                ],
                "action": ["BUY GRADUALLY"] * 3,
                "market_regime": ["correction"] * 3,
                "timing_side": ["BUY"] * 3,
                "forward_5d": [0.01, 0.02, -0.01],
            }
        )

        summary, _ = evaluate_fold(
            history,
            {
                "name": "test",
                "train_end": "2023-12-31",
                "test_start": "2024-01-01",
                "test_end": "2024-12-31",
            },
            minimum_group_sample=1,
            shrinkage_strength=1.0,
            minimum_test_rows=1,
            minimum_relative_brier_improvement=-1.0,
        )
        self.assertEqual(summary["train_rows"], 1)


if __name__ == "__main__":
    unittest.main()
