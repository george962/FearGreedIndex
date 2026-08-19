from __future__ import annotations

import unittest

import pandas as pd

from v3.baseline.freeze_v2_1 import canonical_input_sha256


class BaselineFreezeTests(unittest.TestCase):
    def test_post_cutoff_append_does_not_change_fingerprint(self) -> None:
        cutoff = pd.Timestamp("2026-08-18")
        base = pd.DataFrame(
            {"fear_greed": [40.0, 50.0]},
            index=pd.to_datetime(["2026-08-17", "2026-08-18"]),
        )
        appended = pd.concat(
            [
                base,
                pd.DataFrame(
                    {"fear_greed": [60.0]},
                    index=pd.to_datetime(["2026-08-19"]),
                ),
            ]
        )
        self.assertEqual(
            canonical_input_sha256(base, ["fear_greed"], cutoff),
            canonical_input_sha256(appended, ["fear_greed"], cutoff),
        )

    def test_in_sample_revision_changes_fingerprint(self) -> None:
        cutoff = pd.Timestamp("2026-08-18")
        original = pd.DataFrame(
            {"fear_greed": [40.0, 50.0]},
            index=pd.to_datetime(["2026-08-17", "2026-08-18"]),
        )
        revised = original.copy()
        revised.loc[pd.Timestamp("2026-08-18"), "fear_greed"] = 49.0
        self.assertNotEqual(
            canonical_input_sha256(original, ["fear_greed"], cutoff),
            canonical_input_sha256(revised, ["fear_greed"], cutoff),
        )

    def test_market_ohlc_fingerprint_ignores_future_rows(self) -> None:
        cutoff = pd.Timestamp("2026-08-18")
        market = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
            },
            index=pd.to_datetime(["2026-08-17", "2026-08-18"]),
        )
        future = market.copy()
        future.loc[pd.Timestamp("2026-08-19")] = [102.0, 104.0, 101.0, 103.0]
        columns = ["open", "high", "low", "close"]
        self.assertEqual(
            canonical_input_sha256(market, columns, cutoff),
            canonical_input_sha256(future, columns, cutoff),
        )


if __name__ == "__main__":
    unittest.main()
