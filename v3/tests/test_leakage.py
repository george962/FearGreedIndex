from __future__ import annotations

import unittest

import pandas as pd

from v3.evaluation.validate_dataset import validate_frames


class LeakageValidationTests(unittest.TestCase):
    @staticmethod
    def _valid_labels() -> pd.DataFrame:
        decision = pd.to_datetime(["2024-01-02", "2024-01-03"])
        entry = pd.to_datetime(["2024-01-03", "2024-01-04"])
        frame = pd.DataFrame(
            {
                "decision_date": decision,
                "entry_date": entry,
                "entry_price": [100.0, 101.0],
                "further_5pct_decline_20d": pd.array([False, False], dtype="boolean"),
            }
        )
        for horizon, known in ((5, "2024-01-10"), (20, "2024-01-31"), (60, "2024-03-28")):
            frame[f"forward_return_{horizon}d"] = [0.01, -0.01]
            frame[f"forward_positive_{horizon}d"] = pd.array([True, False], dtype="boolean")
            frame[f"max_drawdown_{horizon}d"] = [-0.01, -0.02]
            frame[f"_forward_{horizon}d_known_date"] = pd.to_datetime([known, known])
        return frame

    def test_valid_dataset_passes(self) -> None:
        features = pd.DataFrame(
            {
                "decision_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "fear_greed_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "fear_greed": [30.0, 30.0],
                "spx_return_1": [0.0, 0.01],
            }
        )
        report = validate_frames(features, self._valid_labels())
        self.assertEqual(report["status"], "PASS", report)

    def test_forward_column_in_features_fails(self) -> None:
        features = pd.DataFrame(
            {
                "decision_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "fear_greed_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "forward_return_5d": [0.01, 0.02],
            }
        )
        report = validate_frames(features, self._valid_labels())
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("forward/target" in error for error in report["errors"]))

    def test_future_source_date_fails(self) -> None:
        features = pd.DataFrame(
            {
                "decision_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "fear_greed_date": pd.to_datetime(["2024-01-04", "2024-01-03"]),
                "fear_greed": [30.0, 40.0],
            }
        )
        report = validate_frames(features, self._valid_labels())
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("future Fear & Greed" in error for error in report["errors"]))

    def test_same_day_entry_fails(self) -> None:
        labels = self._valid_labels()
        labels.loc[0, "entry_date"] = labels.loc[0, "decision_date"]
        features = pd.DataFrame(
            {
                "decision_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "fear_greed_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "fear_greed": [30.0, 40.0],
            }
        )
        report = validate_frames(features, labels)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("entry_date" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
