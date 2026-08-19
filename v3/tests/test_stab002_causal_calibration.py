#!/usr/bin/env python3
"""Tests for STAB-002 nested causal calibration."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from v3.evaluation.stab002_causal_calibration import (
    CALIBRATION_BLOCK_ROWS,
    CALIBRATION_ROWS,
    calibration_blocks,
    fit_platt,
)


class Stab002CausalCalibrationTests(unittest.TestCase):
    @staticmethod
    def _outer_train(rows: int = 714) -> pd.DataFrame:
        dates = pd.date_range("2021-01-01", periods=rows, freq="D")
        return pd.DataFrame(
            {
                "decision_date": dates,
                "favorable_entry_20d": [(index % 3) != 0 for index in range(rows)],
            }
        )

    def test_calibration_region_is_exact_tail_240_in_three_80_row_blocks(self) -> None:
        train = self._outer_train()
        blocks = calibration_blocks(train)
        self.assertEqual(len(blocks), 3)
        self.assertTrue(all(len(block) == CALIBRATION_BLOCK_ROWS for block in blocks))
        combined = pd.concat(blocks, ignore_index=True)
        self.assertEqual(len(combined), CALIBRATION_ROWS)
        expected = train.sort_values("decision_date").iloc[-CALIBRATION_ROWS:]["decision_date"].tolist()
        self.assertEqual(combined["decision_date"].tolist(), expected)

    def test_calibration_region_rejects_insufficient_history(self) -> None:
        with self.assertRaises(ValueError):
            calibration_blocks(self._outer_train(rows=639))

    def test_platt_positive_relation_has_positive_slope(self) -> None:
        score = np.linspace(-0.5, 0.5, CALIBRATION_ROWS)
        target = (score > 0.0).astype(int)
        oof = pd.DataFrame(
            {
                "raw_stability_score": score,
                "favorable_entry_20d": target,
            }
        )
        _, slope, _, positive = fit_platt(oof)
        self.assertGreater(slope, 0.0)
        self.assertTrue(positive)

    def test_platt_negative_relation_is_not_inverted(self) -> None:
        score = np.linspace(-0.5, 0.5, CALIBRATION_ROWS)
        target = (score < 0.0).astype(int)
        oof = pd.DataFrame(
            {
                "raw_stability_score": score,
                "favorable_entry_20d": target,
            }
        )
        _, slope, _, positive = fit_platt(oof)
        self.assertLess(slope, 0.0)
        self.assertFalse(positive)

    def test_platt_requires_both_classes(self) -> None:
        oof = pd.DataFrame(
            {
                "raw_stability_score": np.linspace(-0.5, 0.5, CALIBRATION_ROWS),
                "favorable_entry_20d": np.ones(CALIBRATION_ROWS, dtype=int),
            }
        )
        with self.assertRaises(ValueError):
            fit_platt(oof)


if __name__ == "__main__":
    unittest.main()
