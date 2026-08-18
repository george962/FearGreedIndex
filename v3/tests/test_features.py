from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from v3.features.build_features import build_feature_frame, feature_columns


class FeatureBuilderTests(unittest.TestCase):
    @staticmethod
    def _market(periods: int = 320) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-02", periods=periods)
        trend = np.arange(periods, dtype=float)
        close = 100.0 + 0.08 * trend + np.sin(trend / 9.0)
        return pd.DataFrame(
            {
                "decision_date": dates,
                "open": close - 0.20,
                "high": close + 0.70,
                "low": close - 0.80,
                "close": close,
            }
        )

    @staticmethod
    def _fear_greed(market: pd.DataFrame) -> pd.DataFrame:
        dates = market["decision_date"].iloc[::2].reset_index(drop=True)
        values = 50.0 + 20.0 * np.sin(np.arange(len(dates)) / 11.0)
        return pd.DataFrame({"fear_greed_date": dates, "fear_greed": values})

    def test_build_is_deterministic_and_unique(self) -> None:
        market = self._market()
        fear_greed = self._fear_greed(market)
        first = build_feature_frame(fear_greed, market)
        second = build_feature_frame(fear_greed, market)
        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(first["decision_date"].duplicated().any())
        self.assertTrue(first["decision_date"].is_monotonic_increasing)

    def test_asof_join_never_uses_future_fear_greed(self) -> None:
        market = self._market()
        fear_greed = self._fear_greed(market)
        frame = build_feature_frame(fear_greed, market)
        valid = frame["fear_greed_date"].notna()
        self.assertTrue(
            (frame.loc[valid, "fear_greed_date"] <= frame.loc[valid, "decision_date"]).all()
        )

    def test_feature_columns_do_not_contain_outcomes(self) -> None:
        market = self._market()
        frame = build_feature_frame(self._fear_greed(market), market)
        forbidden = ("forward_", "future_", "target", "label", "outcome")
        leaked = [
            column
            for column in feature_columns(frame)
            if any(token in column.lower() for token in forbidden)
        ]
        self.assertEqual(leaked, [])

    def test_features_change_only_after_source_information_changes(self) -> None:
        market = self._market(40)
        fear_greed = pd.DataFrame(
            {
                "fear_greed_date": [market.loc[0, "decision_date"], market.loc[10, "decision_date"]],
                "fear_greed": [25.0, 75.0],
            }
        )
        frame = build_feature_frame(fear_greed, market)
        self.assertEqual(frame.loc[9, "fear_greed"], 25.0)
        self.assertEqual(frame.loc[10, "fear_greed"], 75.0)


if __name__ == "__main__":
    unittest.main()
