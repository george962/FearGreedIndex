from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from v3.models.gradient_boosting import (
    fit_probability,
    fit_regression,
    predict_fold,
)


class GradientBoostingTests(unittest.TestCase):
    def test_fixed_models_are_deterministic(self) -> None:
        x_train = pd.DataFrame(
            {
                "f1": np.sin(np.arange(140) / 8.0),
                "f2": np.cos(np.arange(140) / 11.0),
            }
        )
        y_class = pd.Series(np.arange(140) % 3 != 0)
        y_return = pd.Series(0.01 * x_train["f1"] - 0.005 * x_train["f2"])
        x_test = x_train.iloc[-20:].copy()

        first_p = fit_probability(x_train, y_class, x_test)
        second_p = fit_probability(x_train, y_class, x_test)
        first_r = fit_regression(x_train, y_return, x_test)
        second_r = fit_regression(x_train, y_return, x_test)

        np.testing.assert_allclose(first_p, second_p, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(first_r, second_r, rtol=0.0, atol=0.0)
        self.assertTrue(np.all((first_p >= 0.0) & (first_p <= 1.0)))
        self.assertTrue(np.isfinite(first_r).all())

    def test_fold_candidate_produces_common_interface(self) -> None:
        dates = pd.bdate_range("2022-01-03", periods=620)
        i = np.arange(len(dates), dtype=float)
        frame = pd.DataFrame(
            {
                "decision_date": dates,
                "f1": np.sin(i / 10.0),
                "f2": np.cos(i / 17.0),
            }
        )
        for horizon in (5, 20, 60):
            future = 0.01 * np.sin((i + horizon) / 15.0) + 0.002 * frame["f1"]
            frame[f"forward_return_{horizon}d"] = future
            frame[f"forward_positive_{horizon}d"] = pd.array(future > 0.0, dtype="boolean")
            frame[f"max_drawdown_{horizon}d"] = -np.abs(future) - 0.005
            frame[f"_forward_{horizon}d_known_date"] = (
                frame["decision_date"] + pd.offsets.BDay(horizon)
            )

        fold = {
            "name": "2024",
            "train_end": "2023-12-29",
            "test_start": "2024-01-02",
            "test_end": "2024-05-31",
        }
        predicted, training_rows = predict_fold(
            frame,
            ["f1", "f2"],
            fold,
            minimum_training_rows=100,
        )
        self.assertGreater(len(predicted), 0)
        for horizon in (5, 20, 60):
            self.assertIn(f"predicted_p_up_{horizon}d", predicted.columns)
            self.assertIn(f"predicted_return_{horizon}d", predicted.columns)
            self.assertTrue(predicted[f"predicted_p_up_{horizon}d"].between(0, 1).all())
        self.assertIn("predicted_drawdown_20d", predicted.columns)
        self.assertTrue(np.isfinite(predicted["predicted_drawdown_20d"]).all())
        self.assertLess(training_rows["p_up_60d"], training_rows["p_up_5d"])

    def test_constant_class_probability_is_supported(self) -> None:
        x_train = pd.DataFrame({"f1": np.arange(30), "f2": np.arange(30) ** 2})
        y_train = pd.Series([False] * 30)
        x_test = x_train.iloc[:4]
        result = fit_probability(x_train, y_train, x_test)
        np.testing.assert_array_equal(result, np.zeros(4))


if __name__ == "__main__":
    unittest.main()
