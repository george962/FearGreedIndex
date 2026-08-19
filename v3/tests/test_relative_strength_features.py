from __future__ import annotations

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from v3.data_sources.fetch_relative_strength import (
    compressed_snapshot_bytes,
    normalized_csv_bytes,
)
from v3.features.build_relative_strength_features import (
    RELATIVE_STRENGTH_FEATURES,
    build_expanded_features,
    build_output_registry,
    build_source_history,
)


class RelativeStrengthFeatureTests(unittest.TestCase):
    @staticmethod
    def _base() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "decision_date": pd.to_datetime(
                    ["2024-01-03", "2024-01-04", "2024-01-05"]
                ),
                "fear_greed_date": pd.to_datetime(
                    ["2024-01-03", "2024-01-04", "2024-01-05"]
                ),
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "fear_greed": [30.0, 35.0, 40.0],
            }
        )

    @staticmethod
    def _source() -> pd.DataFrame:
        dates = pd.bdate_range("2023-01-02", "2024-01-05")
        index = pd.Series(range(len(dates)), dtype=float)
        return pd.DataFrame(
            {
                "relative_strength_date": dates,
                "qqq_close": 250.0 + index * 0.30,
                "spy_close": 380.0 + index * 0.20,
            }
        )

    def test_snapshot_encoding_is_deterministic(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "qqq_close": [400.0, 402.0],
                "spy_close": [475.0, 476.0],
            }
        )
        first = normalized_csv_bytes(frame)
        second = normalized_csv_bytes(frame.copy())
        self.assertEqual(first, second)
        self.assertEqual(compressed_snapshot_bytes(first), compressed_snapshot_bytes(second))

    def test_asof_join_never_uses_future_source(self) -> None:
        base = self._base()
        source = pd.DataFrame(
            {
                "relative_strength_date": pd.to_datetime(
                    ["2024-01-02", "2024-01-05"]
                ),
                "qqq_close": [400.0, 410.0],
                "spy_close": [470.0, 472.0],
            }
        )
        expanded = build_expanded_features(base, source)
        self.assertEqual(
            expanded["relative_strength_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2024-01-02", "2024-01-02", "2024-01-05"],
        )
        self.assertTrue(
            (expanded["relative_strength_date"] <= expanded["decision_date"]).all()
        )

    def test_expansion_preserves_baseline_columns(self) -> None:
        base = self._base()
        expanded = build_expanded_features(base, self._source())
        assert_frame_equal(
            expanded[base.columns].reset_index(drop=True),
            base.reset_index(drop=True),
            check_dtype=True,
        )
        self.assertTrue(set(RELATIVE_STRENGTH_FEATURES).issubset(expanded.columns))

    def test_relative_returns_follow_qqq_vs_spy_growth(self) -> None:
        source = self._source()
        history = build_source_history(source)
        row = history.iloc[-1]
        qqq = source["qqq_close"]
        spy = source["spy_close"]
        expected = (qqq.iloc[-1] / qqq.iloc[-6]) / (spy.iloc[-1] / spy.iloc[-6]) - 1.0
        self.assertAlmostEqual(row["qqq_spy_relative_return_5"], expected, places=12)

    def test_generated_registry_is_baseline_plus_twelve_features(self) -> None:
        registry = build_output_registry()
        names = [item["name"] for item in registry["features"]]
        self.assertEqual(registry["version"], "v3-features-003-relative-strength")
        self.assertEqual(len(RELATIVE_STRENGTH_FEATURES), 12)
        self.assertEqual(len(names), 53)
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
