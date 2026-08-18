from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from v3.models.common import eligible_training_mask
from v3.models.return_regression import fit_predict_return, predict_fold


class ReturnRegressionTests(unittest.TestCase):
    def test_regression_target_uses_same_maturity_gate(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=30)
        frame = pd.DataFrame({"decision_date": dates})
        for horizon in (5, 20, 60):
            frame[f"forward_return_{horizon}d"] = np.linspace(-0.03, 0.04, len(frame))
            frame[f"forward_positive_{horizon}d"] = pd.array(
                frame[f"forward_return_{horizon}d"] > 0.0,
                dtype="boolean",
            )
            frame[f"_forward_{horizon}d_known_date"] = (
                frame["decision_date"] + pd.offsets.BDay(horizon)
            )

        cutoff = dates[15]
        mask = eligible_training_mask(
            frame,
            5,
            cutoff,
            target_column="forward_return_5d",
        )
        eligible = frame.loc[mask]
        self.assertGreater(len(eligible), 0)
        self.assertTrue((eligible["_forward_5d_known_date"] <= cutoff).all())
        self.assertLess(len(eligible), int((frame["decision_date"] <= cutoff).sum()))

    def test_ridge_prediction_is_deterministic_and_finite(self) -> None:
        x_train = pd.DataFrame(
            {
                "f1": np.linspace(-2.0, 2.0, 80),
                "f2": np.sin(np.arange(80) / 7.0),
            }
        )
        y_train = 0.01 + 0.02 * x_train["f1"] - 0.005 * x_train["f2"]
        x_test = pd.DataFrame(
            {
                "f1": np.linspace(-1.0, 1.0, 15),
                "f2": np.cos(np.arange(15) / 5.0),
            }
        )
        first = fit_predict_return(x_train, y_train, x_test)
        second = fit_predict_return(x_train, y_train, x_test)
        np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)
        self.assertTrue(np.isfinite(first).all())

    def test_fold_prediction_uses_standard_return_columns(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=420)
        index = np.arange(len(dates), dtype=float)
        frame = pd.DataFrame(
            {
                "decision_date": dates,
                "f1": np.sin(index / 9.0),
                "f2": np.cos(index / 13.0),
            }
        )
        for horizon in (5, 20, 60):
            target = 0.001 * np.sin((index + horizon) / 11.0) + 0.0002 * index / len(index)
            frame[f"forward_return_{horizon}d"] = target
            frame[f"forward_positive_{horizon}d"] = pd.array(target > 0.0, dtype="boolean")
            frame[f"_forward_{horizon}d_known_date"] = (
                frame["decision_date"] + pd.offsets.BDay(horizon)
            )

        fold = {
            "name": "test",
            "train_end": "2023-12-29",
            "test_start": "2024-01-02",
            "test_end": "2024-06-28",
        }
        predicted, training_rows = predict_fold(
            frame,
            ["f1", "f2"],
            fold,
            minimum_training_rows=50,
        )
        self.assertGreater(len(predicted), 0)
        for horizon in (5, 20, 60):
            column = f"predicted_return_{horizon}d"
            self.assertIn(column, predicted.columns)
            self.assertTrue(np.isfinite(predicted[column]).all())
        self.assertLess(training_rows[60], training_rows[5])

    def test_missing_regression_target_is_excluded(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=15)
        frame = pd.DataFrame(
            {
                "decision_date": dates,
                "forward_return_5d": np.linspace(-0.01, 0.02, 15),
                "forward_positive_5d": pd.array([True] * 15, dtype="boolean"),
                "_forward_5d_known_date": dates,
            }
        )
        frame.loc[3, "forward_return_5d"] = np.nan
        mask = eligible_training_mask(
            frame,
            5,
            dates[-1],
            target_column="forward_return_5d",
        )
        self.assertFalse(bool(mask.iloc[3]))


if __name__ == "__main__":
    unittest.main()
