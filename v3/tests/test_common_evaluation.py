from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from v3.evaluation.backtest import simulate_exposure_strategy
from v3.evaluation.metrics import (
    classification_metrics,
    performance_metrics,
    regression_metrics,
)
from v3.evaluation.walk_forward import validate_common_samples


class CommonEvaluationTests(unittest.TestCase):
    def test_classification_metrics_include_calibration_and_baseline(self) -> None:
        y = np.array([0, 0, 1, 1], dtype=float)
        p = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
        metrics = classification_metrics(y, p, calibration_bins=5)
        self.assertLess(metrics["brier_score"], metrics["baseline_brier_score"])
        self.assertGreater(metrics["relative_brier_improvement"], 0.0)
        self.assertGreaterEqual(metrics["expected_calibration_error"], 0.0)
        self.assertLessEqual(metrics["expected_calibration_error"], 1.0)

    def test_regression_metrics_include_rank_correlation(self) -> None:
        actual = np.array([-0.02, 0.00, 0.01, 0.03, 0.05])
        predicted = np.array([-0.01, 0.00, 0.02, 0.025, 0.04])
        metrics = regression_metrics(actual, predicted)
        self.assertGreater(metrics["spearman_rank_correlation"], 0.9)
        self.assertGreater(metrics["rmse"], 0.0)
        self.assertGreater(metrics["mae"], 0.0)

    def test_performance_metrics_and_backtest_are_policy_agnostic(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=260)
        market = pd.Series(0.0004, index=dates)
        exposure = pd.Series(1.0, index=dates)
        frame, summary = simulate_exposure_strategy(
            market,
            exposure,
            transaction_cost_bps_per_1x_turnover=2.0,
        )
        self.assertAlmostEqual(
            summary["strategy"]["total_return"],
            summary["benchmark"]["total_return"],
            places=12,
        )
        self.assertEqual(summary["total_turnover"], 0.0)
        direct = performance_metrics(market)
        self.assertGreater(direct["annualized_return"], 0.0)
        self.assertEqual(direct["max_drawdown"], 0.0)
        self.assertEqual(len(frame), len(market))

    def test_common_sample_contract_rejects_mismatched_dates(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "fold": "2024",
                    "target_type": "classification",
                    "target": "p_up_20d",
                    "horizon": 20,
                    "sample_sha256": "aaa",
                },
                {
                    "fold": "2024",
                    "target_type": "classification",
                    "target": "p_up_20d",
                    "horizon": 20,
                    "sample_sha256": "bbb",
                },
            ]
        )
        with self.assertRaises(ValueError):
            validate_common_samples(metrics)

    def test_common_sample_contract_accepts_identical_dates(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "fold": "2025",
                    "target_type": "return_regression",
                    "target": "return_60d",
                    "horizon": 60,
                    "sample_sha256": "same",
                },
                {
                    "fold": "2025",
                    "target_type": "return_regression",
                    "target": "return_60d",
                    "horizon": 60,
                    "sample_sha256": "same",
                },
            ]
        )
        validate_common_samples(metrics)


if __name__ == "__main__":
    unittest.main()
