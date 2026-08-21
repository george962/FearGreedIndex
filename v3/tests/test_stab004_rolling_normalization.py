#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from v3.evaluation.stab004_rolling_normalization import (
    redundancy_clusters,
    rolling_score_percentiles,
)


class Stab004RollingNormalizationTests(unittest.TestCase):
    def test_redundancy_clusters_collapse_highly_correlated_features(self) -> None:
        rows = 300
        base = np.linspace(-2.0, 2.0, rows)
        train = pd.DataFrame(
            {
                "a": base,
                "b": base * 3.0 + 1.0,
                "c": np.sin(np.linspace(0.0, 20.0, rows)),
            }
        )
        consensus = [
            {"feature": "a", "consensus_weight": 0.20, "direction": 1},
            {"feature": "b", "consensus_weight": 0.30, "direction": 1},
            {"feature": "c", "consensus_weight": 0.10, "direction": -1},
        ]
        representatives, clusters = redundancy_clusters(train, consensus)
        names = [item["feature"] for item in representatives]
        self.assertEqual(names, ["b", "c"])
        self.assertEqual(len({row["cluster_id"] for row in clusters}), 2)
        ab_rows = [row for row in clusters if row["member_feature"] in {"a", "b"}]
        self.assertTrue(all(row["representative_feature"] == "b" for row in ab_rows))

    def test_representative_tie_breaks_lexicographically(self) -> None:
        train = pd.DataFrame(
            {
                "a": np.arange(200, dtype=float),
                "b": np.arange(200, dtype=float),
            }
        )
        consensus = [
            {"feature": "b", "consensus_weight": 0.20, "direction": 1},
            {"feature": "a", "consensus_weight": 0.20, "direction": 1},
        ]
        representatives, _ = redundancy_clusters(train, consensus)
        self.assertEqual([item["feature"] for item in representatives], ["a"])

    def test_rolling_percentile_uses_only_prior_scores(self) -> None:
        dates = pd.Series(pd.date_range("2025-01-01", periods=10, freq="D"))
        scores = np.arange(10, dtype=float)
        target_dates = {pd.Timestamp("2025-01-06"), pd.Timestamp("2025-01-10")}
        result = rolling_score_percentiles(
            dates,
            scores,
            target_dates,
            reference_window=4,
            minimum_reference=4,
        ).set_index("decision_date")

        first = result.loc[pd.Timestamp("2025-01-06")]
        self.assertEqual(first["reference_count"], 4)
        self.assertEqual(first["rolling_percentile"], 1.0)
        self.assertEqual(first["call_state"], "STRONG_FAVORABLE")

        second = result.loc[pd.Timestamp("2025-01-10")]
        self.assertEqual(second["reference_count"], 4)
        self.assertEqual(second["rolling_percentile"], 1.0)

    def test_rolling_percentile_abstains_without_minimum_history(self) -> None:
        dates = pd.Series(pd.date_range("2025-01-01", periods=5, freq="D"))
        scores = np.arange(5, dtype=float)
        result = rolling_score_percentiles(
            dates,
            scores,
            {pd.Timestamp("2025-01-03")},
            reference_window=4,
            minimum_reference=4,
        )
        self.assertEqual(result.iloc[0]["reference_count"], 2)
        self.assertTrue(np.isnan(result.iloc[0]["rolling_percentile"]))
        self.assertEqual(result.iloc[0]["call_state"], "ABSTAIN")


if __name__ == "__main__":
    unittest.main()
