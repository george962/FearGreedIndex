#!/usr/bin/env python3
"""Regression tests for EVID-001 collection-date governance."""

from __future__ import annotations

import unittest

from v3.evidence.collect_forward_evidence import classify_collection_date


class ForwardCollectionPolicyTests(unittest.TestCase):
    def test_same_market_date_is_eligible(self) -> None:
        self.assertEqual(
            classify_collection_date("2026-08-19", "2026-08-19"),
            "ELIGIBLE",
        )

    def test_past_decision_is_marked_missed_not_backfilled(self) -> None:
        self.assertEqual(
            classify_collection_date("2026-08-19", "2026-08-20"),
            "MISSED",
        )

    def test_future_decision_relative_to_collector_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            classify_collection_date("2026-08-20", "2026-08-19")


if __name__ == "__main__":
    unittest.main()
