from __future__ import annotations

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from v3.data_sources.fetch_treasury import (
    compressed_snapshot_bytes,
    normalize_series_csv,
    normalized_csv_bytes,
)
from v3.features.build_treasury_features import (
    TREASURY_FEATURES,
    build_expanded_features,
    build_output_registry,
    build_source_history,
)


class TreasuryFeatureTests(unittest.TestCase):
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
        dates = pd.bdate_range("2022-12-01", "2024-01-05")
        values = pd.Series(range(len(dates)), dtype=float)
        return pd.DataFrame(
            {
                "treasury_date": dates,
                "dgs2": 4.0 + values * 0.001,
                "dgs10": 3.5 + values * 0.0015,
            }
        )

    def test_fred_normalizer_handles_missing_dot_values(self) -> None:
        payload = (
            b"observation_date,DGS10\n"
            b"2024-01-02,3.95\n"
            b"2024-01-03,.\n"
            b"2024-01-04,4.01\n"
        )
        normalized = normalize_series_csv(payload, "DGS10")
        self.assertEqual(normalized["date"].dt.strftime("%Y-%m-%d").tolist(), ["2024-01-02", "2024-01-04"])
        self.assertEqual(normalized["dgs10"].tolist(), [3.95, 4.01])

    def test_snapshot_encoding_is_deterministic(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "dgs2": [4.3, 4.2],
                "dgs10": [4.0, 3.95],
            }
        )
        first = normalized_csv_bytes(frame)
        second = normalized_csv_bytes(frame.copy())
        self.assertEqual(first, second)
        self.assertEqual(compressed_snapshot_bytes(first), compressed_snapshot_bytes(second))

    def test_asof_join_never_uses_future_treasury(self) -> None:
        source = pd.DataFrame(
            {
                "treasury_date": pd.to_datetime(["2024-01-02", "2024-01-05"]),
                "dgs2": [4.3, 4.2],
                "dgs10": [4.0, 3.9],
            }
        )
        expanded = build_expanded_features(self._base(), source)
        self.assertEqual(
            expanded["treasury_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2024-01-02", "2024-01-02", "2024-01-05"],
        )
        self.assertTrue((expanded["treasury_date"] <= expanded["decision_date"]).all())

    def test_expansion_preserves_baseline_columns(self) -> None:
        base = self._base()
        expanded = build_expanded_features(base, self._source())
        assert_frame_equal(
            expanded[base.columns].reset_index(drop=True),
            base.reset_index(drop=True),
            check_dtype=True,
        )
        self.assertTrue(set(TREASURY_FEATURES).issubset(expanded.columns))

    def test_slope_and_changes_use_percentage_points(self) -> None:
        source = self._source()
        history = build_source_history(source)
        row = history.iloc[-1]
        expected_slope = source["dgs10"].iloc[-1] - source["dgs2"].iloc[-1]
        expected_change_5 = source["dgs10"].iloc[-1] - source["dgs10"].iloc[-6]
        self.assertAlmostEqual(row["treasury_10y_2y_slope"], expected_slope, places=12)
        self.assertAlmostEqual(row["treasury_10y_change_5"], expected_change_5, places=12)
        self.assertTrue(0.0 <= row["treasury_10y_percentile_252"] <= 1.0)

    def test_registry_is_baseline_plus_twelve_features(self) -> None:
        registry = build_output_registry()
        names = [item["name"] for item in registry["features"]]
        self.assertEqual(registry["version"], "v3-features-004-treasury")
        self.assertEqual(len(TREASURY_FEATURES), 12)
        self.assertEqual(len(names), 53)
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
