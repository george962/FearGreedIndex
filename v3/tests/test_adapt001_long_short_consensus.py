#!/usr/bin/env python3
"""Tests for ADAPT-001 long/short consensus and abstention."""

from __future__ import annotations

import unittest

import pandas as pd

from v3.evaluation.adapt001_long_short_consensus import (
    active_gate,
    consensus_features,
    short_memory_association,
)


class Adapt001ConsensusTests(unittest.TestCase):
    @staticmethod
    def _history(sign: int = 1) -> pd.DataFrame:
        rows = []
        for index in range(150):
            target = index % 2
            value = float(target * 10 + index * 0.001) if sign > 0 else float((1 - target) * 10 + index * 0.001)
            rows.append(
                {
                    "decision_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=index),
                    "favorable_entry_20d": target,
                    "spx_return_60": value,
                    "treasury_10y_2y_slope": value,
                    "fg_min_20": value,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _long_selected(sign: int = 1) -> list[dict[str, object]]:
        return [
            {"feature": "spx_return_60", "majority_sign": sign, "weight": 0.30},
            {"feature": "treasury_10y_2y_slope", "majority_sign": sign, "weight": 0.25},
            {"feature": "fg_min_20", "majority_sign": sign, "weight": 0.20},
        ]

    def test_short_memory_association_detects_direction(self) -> None:
        rho, observed, usable = short_memory_association(self._history(1).iloc[-126:], "spx_return_60")
        self.assertGreater(rho, 0.03)
        self.assertGreaterEqual(observed, 100)
        self.assertTrue(usable)

    def test_consensus_keeps_features_when_short_and_long_agree(self) -> None:
        active, share, _ = consensus_features(self._history(1), self._long_selected(1))
        self.assertEqual(len(active), 3)
        self.assertAlmostEqual(share, 1.0)
        passed, families = active_gate(active, share)
        self.assertTrue(passed)
        self.assertEqual(families, 3)

    def test_consensus_rejects_features_when_recent_direction_flips(self) -> None:
        active, share, diagnostics = consensus_features(self._history(-1), self._long_selected(1))
        self.assertEqual(active, [])
        self.assertAlmostEqual(share, 0.0)
        self.assertTrue(all(not row["consensus"] for row in diagnostics))
        passed, _ = active_gate(active, share)
        self.assertFalse(passed)

    def test_active_gate_requires_weight_share_and_family_diversity(self) -> None:
        selected = self._long_selected(1)
        passed, families = active_gate(selected, 0.59)
        self.assertFalse(passed)
        self.assertEqual(families, 3)

        same_family = [
            {"feature": "spx_return_60", "majority_sign": 1, "weight": 0.3},
            {"feature": "spx_realized_vol_20", "majority_sign": 1, "weight": 0.2},
            {"feature": "spx_distance_ma_200", "majority_sign": 1, "weight": 0.2},
        ]
        passed, families = active_gate(same_family, 0.80)
        self.assertFalse(passed)
        self.assertEqual(families, 1)


if __name__ == "__main__":
    unittest.main()
