from __future__ import annotations

import unittest

import pandas as pd

from v3.evaluation.exp009_recent_window import (
    WINDOW_ROWS,
    recent_training_frame,
    summarize_viability,
)


class Exp009RecentWindowTests(unittest.TestCase):
    def _frame(self, rows: int = 600) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=rows, freq="D")
        return pd.DataFrame(
            {
                "decision_date": dates,
                "_forward_20d_known_date": dates,
                "favorable_entry_20d": [bool(index % 2) for index in range(rows)],
            }
        )

    def test_recent_window_uses_exact_tail(self) -> None:
        frame = self._frame()
        result = recent_training_frame(frame, frame["decision_date"].iloc[-1])
        self.assertEqual(len(result), WINDOW_ROWS)
        self.assertEqual(
            result["decision_date"].iloc[0],
            frame["decision_date"].iloc[-WINDOW_ROWS],
        )
        self.assertEqual(result["decision_date"].iloc[-1], frame["decision_date"].iloc[-1])

    def test_recent_window_fails_when_history_is_too_short(self) -> None:
        with self.assertRaises(ValueError):
            recent_training_frame(self._frame(503), pd.Timestamp("2030-01-01"))

    def test_viability_requires_absolute_and_full_history_improvement(self) -> None:
        metrics = pd.DataFrame(
            {
                "training_rows": [504, 504, 504],
                "sample_hash_matches_exp006": [True, True, True],
                "relative_brier_improvement": [0.03, 0.02, -0.01],
                "roc_auc": [0.58, 0.55, 0.49],
                "brier_improvement_vs_full_history": [0.02, 0.01, -0.01],
                "auc_improvement_vs_full_history": [0.04, 0.03, -0.01],
            }
        )
        result = summarize_viability(metrics)
        self.assertTrue(result["viability_gate_pass"])

        metrics.loc[1, "brier_improvement_vs_full_history"] = -0.02
        result_bad = summarize_viability(metrics)
        self.assertFalse(result_bad["viability_gate_pass"])


if __name__ == "__main__":
    unittest.main()
