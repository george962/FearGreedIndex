from __future__ import annotations

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from v3.data_sources.fetch_vix import normalize_vix_csv, normalized_csv_bytes
from v3.features.build_vix_features import (
    VIX_FEATURE_COLUMNS,
    build_expanded_features,
    build_vix_history,
)


class VixFeatureTests(unittest.TestCase):
    @staticmethod
    def _base_features() -> pd.DataFrame:
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
                "spx_return_1": [0.0, 0.01, 0.0098],
            }
        )

    @staticmethod
    def _vix_history() -> pd.DataFrame:
        dates = pd.bdate_range("2022-12-01", "2024-01-05")
        return pd.DataFrame(
            {
                "vix_date": dates,
                "vix_close": [20.0 + (index % 17) * 0.2 for index in range(len(dates))],
            }
        )

    def test_cboe_normalizer_accepts_case_insensitive_date_and_close(self) -> None:
        payload = b"DATE,OPEN,HIGH,LOW,CLOSE\n01/02/2024,13,14,12,13.5\n01/03/2024,14,15,13,14.5\n"
        normalized = normalize_vix_csv(payload)
        self.assertEqual(list(normalized.columns), ["date", "vix_close"])
        self.assertEqual(normalized["date"].dt.strftime("%Y-%m-%d").tolist(), ["2024-01-02", "2024-01-03"])
        self.assertEqual(normalized["vix_close"].tolist(), [13.5, 14.5])
        self.assertEqual(normalized_csv_bytes(normalized), normalized_csv_bytes(normalized.copy()))

    def test_asof_join_never_uses_future_vix(self) -> None:
        base = self._base_features()
        vix = pd.DataFrame(
            {
                "vix_date": pd.to_datetime(["2024-01-02", "2024-01-05"]),
                "vix_close": [15.0, 25.0],
            }
        )
        expanded = build_expanded_features(base, vix)
        self.assertEqual(
            expanded["vix_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2024-01-02", "2024-01-02", "2024-01-05"],
        )
        self.assertTrue((expanded["vix_date"] <= expanded["decision_date"]).all())

    def test_expansion_preserves_baseline_columns_exactly(self) -> None:
        base = self._base_features()
        expanded = build_expanded_features(base, self._vix_history())
        assert_frame_equal(
            expanded[base.columns].reset_index(drop=True),
            base.reset_index(drop=True),
            check_dtype=True,
        )
        self.assertTrue(set(VIX_FEATURE_COLUMNS).issubset(expanded.columns))

    def test_vix_feature_build_is_deterministic(self) -> None:
        vix = self._vix_history()
        first = build_vix_history(vix)
        second = build_vix_history(vix.copy())
        assert_frame_equal(first, second, check_dtype=True)
        self.assertEqual(list(first.columns), ["vix_date", *VIX_FEATURE_COLUMNS])
        self.assertTrue(first["vix_percentile_252"].dropna().between(0.0, 1.0).all())

    def test_duplicate_vix_dates_are_rejected_by_normalizer(self) -> None:
        payload = b"DATE,CLOSE\n01/02/2024,13.5\n01/02/2024,14.0\n"
        with self.assertRaisesRegex(ValueError, "Duplicate VIX dates"):
            normalize_vix_csv(payload)


if __name__ == "__main__":
    unittest.main()
