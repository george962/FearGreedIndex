from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from v3.models.extra_trees_calibrated import (
    MINIMUM_CALIBRATION_ROWS,
    chronological_calibration_split,
    fit_calibrated_probability,
    fit_regression,
)


class ExtraTreesCalibratedTests(unittest.TestCase):
    @staticmethod
    def _classification_frame(rows: int = 500) -> pd.DataFrame:
        index = np.arange(rows, dtype=float)
        return pd.DataFrame(
            {
                "decision_date": pd.date_range("2020-01-01", periods=rows, freq="D"),
                "f1": np.sin(index / 13.0),
                "f2": np.cos(index / 29.0),
                "target": (index.astype(int) % 3 != 0).astype(int),
            }
        )

    def test_chronological_split_uses_earlier_fit_and_later_calibration(self) -> None:
        frame = self._classification_frame(500)
        fit, calibration = chronological_calibration_split(frame)
        self.assertEqual(len(fit), 400)
        self.assertEqual(len(calibration), 100)
        self.assertLess(fit["decision_date"].max(), calibration["decision_date"].min())
        self.assertEqual(fit.index.intersection(calibration.index).size, 0)

    def test_calibration_minimum_fails_closed(self) -> None:
        frame = self._classification_frame(300)
        with self.assertRaisesRegex(ValueError, "calibration segment"):
            chronological_calibration_split(frame)

    def test_calibration_requires_both_classes(self) -> None:
        frame = self._classification_frame(500)
        frame.loc[frame.index >= 400, "target"] = 1
        with self.assertRaisesRegex(ValueError, "both binary classes"):
            fit_calibrated_probability(
                frame,
                ["f1", "f2"],
                "target",
                frame.iloc[-20:][["f1", "f2"]],
            )

    def test_calibrated_probability_is_deterministic_and_bounded(self) -> None:
        frame = self._classification_frame(500)
        test = frame.iloc[-30:][["f1", "f2"]].copy()
        first, evidence_first = fit_calibrated_probability(
            frame, ["f1", "f2"], "target", test
        )
        second, evidence_second = fit_calibrated_probability(
            frame, ["f1", "f2"], "target", test
        )
        np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(((first >= 0.0) & (first <= 1.0)).all())
        self.assertEqual(evidence_first, evidence_second)
        self.assertGreaterEqual(evidence_first["calibration_rows"], MINIMUM_CALIBRATION_ROWS)

    def test_regression_is_deterministic_and_finite(self) -> None:
        rows = 400
        index = np.arange(rows, dtype=float)
        features = pd.DataFrame(
            {
                "f1": np.sin(index / 17.0),
                "f2": np.cos(index / 31.0),
            }
        )
        target = pd.Series(0.3 * features["f1"] - 0.2 * features["f2"])
        test = features.iloc[-25:].copy()
        first = fit_regression(features, target, test)
        second = fit_regression(features, target, test)
        np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)
        self.assertTrue(np.isfinite(first).all())


if __name__ == "__main__":
    unittest.main()
