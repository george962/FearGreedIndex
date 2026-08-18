from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from v3.models.common import eligible_training_mask
from v3.models.logistic_baseline import (
    fit_predict_probability,
    predict_fold,
)


class LogisticBaselineTests(unittest.TestCase):
    def test_eligible_training_mask_requires_matured_outcome(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=20)
        frame = pd.DataFrame({"decision_date": dates})
        for horizon in (5, 20, 60):
            frame[f"forward_positive_{horizon}d"] = pd.array(
                [index % 2 == 0 for index in range(len(frame))],
                dtype="boolean",
            )
            frame[f"_forward_{horizon}d_known_date"] = (
                frame["decision_date"] + pd.offsets.BDay(horizon)
            )

        cutoff = dates[10]
        mask = eligible_training_mask(frame, 5, cutoff)
        eligible = frame.loc[mask]
        self.assertTrue((eligible["decision_date"] <= cutoff).all())
        self.assertTrue((eligible["_forward_5d_known_date"] <= cutoff).all())
        self.assertLess(len(eligible), int((frame["decision_date"] <= cutoff).sum()))

    def test_probability_fit_is_deterministic_and_bounded(self) -> None:
        x_train = pd.DataFrame(
            {
                "f1": np.linspace(-2.0, 2.0, 60),
                "f2": np.sin(np.arange(60) / 5.0),
            }
        )
        y_train = pd.Series([index % 3 != 0 for index in range(60)])
        x_test = pd.DataFrame(
            {
                "f1": np.linspace(-1.5, 1.5, 12),
                "f2": np.cos(np.arange(12) / 4.0),
            }
        )

        first = fit_predict_probability(x_train, y_train, x_test)
        second = fit_predict_probability(x_train, y_train, x_test)
        np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)
        self.assertTrue(np.all(first >= 0.0))
        self.assertTrue(np.all(first <= 1.0))

    def test_constant_training_class_is_supported(self) -> None:
        x_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.0, 0.0, 1.0]})
        y_train = pd.Series([True, True, True])
        x_test = pd.DataFrame({"f1": [4.0, 5.0], "f2": [1.0, 2.0]})
        result = fit_predict_probability(x_train, y_train, x_test)
        np.testing.assert_array_equal(result, np.ones(2))

    def test_fold_prediction_uses_standard_probability_columns(self) -> None:
        dates = pd.bdate_range("2023-10-02", periods=180)
        index = np.arange(len(dates), dtype=float)
        frame = pd.DataFrame(
            {
                "decision_date": dates,
                "f1": np.sin(index / 9.0),
                "f2": np.cos(index / 13.0),
            }
        )
        for horizon in (5, 20, 60):
            frame[f"forward_positive_{horizon}d"] = pd.array(
                ((np.arange(len(frame)) + horizon) % 4 != 0),
                dtype="boolean",
            )
            frame[f"_forward_{horizon}d_known_date"] = (
                frame["decision_date"] + pd.offsets.BDay(horizon)
            )

        fold = {
            "name": "2024_q2",
            "train_end": "2024-03-29",
            "test_start": "2024-04-01",
            "test_end": "2024-06-28",
        }
        predicted, training_rows = predict_fold(
            frame,
            ["f1", "f2"],
            fold,
            minimum_training_rows=10,
        )

        self.assertGreater(len(predicted), 0)
        for horizon in (5, 20, 60):
            column = f"predicted_p_up_{horizon}d"
            self.assertIn(column, predicted.columns)
            self.assertTrue(predicted[column].between(0.0, 1.0).all())
            self.assertGreaterEqual(training_rows[horizon], 10)
        self.assertLess(training_rows[60], training_rows[5])


if __name__ == "__main__":
    unittest.main()
