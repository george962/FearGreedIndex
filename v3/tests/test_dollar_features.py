from __future__ import annotations

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from v3.data_sources.fetch_dollar import compressed_snapshot_bytes, normalize_csv, normalized_csv_bytes
from v3.features.build_dollar_features import DOLLAR_FEATURES, build_expanded_features, build_history, output_registry


class DollarFeatureTests(unittest.TestCase):
    @staticmethod
    def _base() -> pd.DataFrame:
        return pd.DataFrame({
            "decision_date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
            "fear_greed_date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
            "open": [100.0, 101.0, 102.0], "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0], "close": [101.0, 102.0, 103.0],
            "fear_greed": [30.0, 35.0, 40.0],
        })

    @staticmethod
    def _source() -> pd.DataFrame:
        dates = pd.bdate_range("2022-12-01", "2024-01-04")
        values = pd.Series(range(len(dates)), dtype=float)
        return pd.DataFrame({"dollar_observation_date": dates, "dollar_index": 115.0 + values * 0.01})

    def test_fred_normalizer_handles_missing_values(self) -> None:
        payload = b"observation_date,DTWEXBGS\n2024-01-02,120.1\n2024-01-03,.\n2024-01-04,120.3\n"
        frame = normalize_csv(payload)
        self.assertEqual(frame["date"].dt.strftime("%Y-%m-%d").tolist(), ["2024-01-02", "2024-01-04"])

    def test_snapshot_encoding_is_deterministic(self) -> None:
        frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "dollar_index": [120.1, 120.2]})
        first = normalized_csv_bytes(frame)
        self.assertEqual(first, normalized_csv_bytes(frame.copy()))
        self.assertEqual(compressed_snapshot_bytes(first), compressed_snapshot_bytes(first))

    def test_one_day_availability_lag_is_enforced(self) -> None:
        source = pd.DataFrame({
            "dollar_observation_date": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "dollar_index": [120.0, 121.0],
        })
        expanded = build_expanded_features(self._base(), source)
        self.assertEqual(expanded["dollar_observation_date"].dt.strftime("%Y-%m-%d").tolist(), ["2024-01-02", "2024-01-02", "2024-01-04"])
        self.assertTrue((expanded["dollar_observation_date"] < expanded["decision_date"]).all())
        self.assertTrue((expanded["dollar_available_date"] <= expanded["decision_date"]).all())

    def test_baseline_is_preserved(self) -> None:
        base = self._base()
        expanded = build_expanded_features(base, self._source())
        assert_frame_equal(expanded[base.columns].reset_index(drop=True), base.reset_index(drop=True), check_dtype=True)

    def test_returns_are_percentage_changes(self) -> None:
        source = self._source()
        history = build_history(source)
        expected = source["dollar_index"].iloc[-1] / source["dollar_index"].iloc[-6] - 1.0
        self.assertAlmostEqual(history.iloc[-1]["dollar_return_5"], expected, places=12)

    def test_registry_is_baseline_plus_eleven_features(self) -> None:
        registry = output_registry()
        names = [item["name"] for item in registry["features"]]
        self.assertEqual(registry["version"], "v3-features-005-dollar")
        self.assertEqual(len(DOLLAR_FEATURES), 11)
        self.assertEqual(len(names), 52)
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
