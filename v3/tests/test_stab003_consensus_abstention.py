#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from v3.evaluation.stab003_consensus_abstention import (
    build_consensus,
    call_states,
    select_short_memory_features,
    training_abstention_thresholds,
)


class Stab003ConsensusTests(unittest.TestCase):
    def test_consensus_requires_same_direction(self) -> None:
        long_selected = [
            {"feature": "a", "majority_sign": 1, "weight": 0.16},
            {"feature": "b", "majority_sign": -1, "weight": 0.09},
        ]
        short_selected = [
            {"feature": "a", "direction": 1, "weight": 0.09},
            {"feature": "b", "direction": 1, "weight": 0.16},
        ]
        result = build_consensus(long_selected, short_selected)
        self.assertEqual([item["feature"] for item in result], ["a"])
        self.assertAlmostEqual(result[0]["consensus_weight"], 0.12)

    def test_training_quantiles_drive_abstention(self) -> None:
        scores = np.arange(100, dtype=float)
        low, high = training_abstention_thresholds(scores)
        states = call_states(np.array([0.0, 50.0, 99.0]), low, high)
        self.assertEqual(states.tolist(), ["STRONG_UNFAVORABLE", "ABSTAIN", "STRONG_FAVORABLE"])

    def test_short_selector_requires_consistent_sign(self) -> None:
        rows = 504
        dates = pd.date_range("2024-01-01", periods=rows, freq="D")
        target = np.tile([0, 1], rows // 2)
        stable = target.astype(float) + np.linspace(0, 0.01, rows)
        reversing = stable.copy()
        reversing[-168:] *= -1
        frame = pd.DataFrame(
            {
                "decision_date": dates,
                "favorable_entry_20d": target,
                "stable": stable,
                "reversing": reversing,
            }
        )
        selected, _ = select_short_memory_features(frame, ["stable", "reversing"])
        names = {item["feature"] for item in selected}
        self.assertIn("stable", names)
        self.assertNotIn("reversing", names)

    def test_short_selector_fails_closed_with_too_few_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "decision_date": pd.date_range("2024-01-01", periods=300, freq="D"),
                "favorable_entry_20d": np.tile([0, 1], 150),
                "x": np.arange(300),
            }
        )
        selected, diagnostics = select_short_memory_features(frame, ["x"])
        self.assertEqual(selected, [])
        self.assertEqual(diagnostics, [])


if __name__ == "__main__":
    unittest.main()
