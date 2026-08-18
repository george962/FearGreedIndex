from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from v3.labels.build_labels import build_labels


class LabelBuilderTests(unittest.TestCase):
    @staticmethod
    def _market(periods: int = 80) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-02", periods=periods)
        base = 100.0 + np.arange(periods, dtype=float)
        return pd.DataFrame(
            {
                "date": dates,
                "open": base,
                "high": base + 2.0,
                "low": base - 1.0,
                "close": base + 0.5,
            }
        )

    def test_next_session_open_is_entry(self) -> None:
        market = self._market()
        decision = market.loc[0, "date"]
        labels = build_labels(pd.Series([decision]), market)
        self.assertEqual(labels.loc[0, "entry_date"], market.loc[1, "date"])
        self.assertEqual(labels.loc[0, "entry_price"], market.loc[1, "open"])

    def test_five_session_return_and_known_date_are_hand_checkable(self) -> None:
        market = self._market()
        labels = build_labels(pd.Series([market.loc[0, "date"]]), market)
        expected_entry = float(market.loc[1, "open"])
        expected_close = float(market.loc[5, "close"])
        expected_return = expected_close / expected_entry - 1.0
        self.assertAlmostEqual(labels.loc[0, "forward_return_5d"], expected_return)
        self.assertEqual(labels.loc[0, "_forward_5d_known_date"], market.loc[5, "date"])

    def test_drawdown_uses_post_entry_path_only(self) -> None:
        market = self._market()
        market.loc[0, "low"] = 1.0
        labels = build_labels(pd.Series([market.loc[0, "date"]]), market)
        expected = market.loc[1:5, "low"].min() / market.loc[1, "open"] - 1.0
        self.assertAlmostEqual(labels.loc[0, "max_drawdown_5d"], expected)

    def test_unmatured_horizons_remain_missing(self) -> None:
        market = self._market(10)
        labels = build_labels(pd.Series([market.loc[8, "date"]]), market)
        self.assertTrue(pd.isna(labels.loc[0, "forward_return_5d"]))
        self.assertTrue(pd.isna(labels.loc[0, "_forward_5d_known_date"]))


if __name__ == "__main__":
    unittest.main()
