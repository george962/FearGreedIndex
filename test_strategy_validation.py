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
        self.assertEqual(summary["train_rows"], 365)
        self.assertTrue(
            (pd.to_datetime(predicted["decision_date"]) >= pd.Timestamp("2024-01-01")).all()
        )


if __name__ == "__main__":
    unittest.main()
