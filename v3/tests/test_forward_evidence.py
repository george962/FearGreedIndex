#!/usr/bin/env python3
"""Tests for the EVID-001 untouched forward evidence lane."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from v3.evidence.append_forward_snapshot import append_snapshot, write_ledger
from v3.evidence.verify_forward_lane import verify_lane


class ForwardEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.manifest = self.root / "manifest.json"
        self.registry = self.root / "registry.json"
        self.features = self.root / "features.parquet"
        self.ledger = self.root / "ledger.csv"
        self.checkpoints = self.root / "checkpoints.json"
        self.feature_names = ["f1", "f2"]

        self.manifest.write_text(
            json.dumps(
                {
                    "lane_id": "test-forward-lane",
                    "research_exposed_through": "2026-08-18",
                    "first_eligible_decision_date": "2026-08-19",
                    "feature_set_version": "test-features",
                    "feature_count": 2,
                    "collector_version": "test-collector-v1",
                    "checkpoint_registry_version": 1,
                    "forbidden_column_fragments": [
                        "forward_return",
                        "forward_positive",
                        "max_drawdown",
                        "known_date",
                        "realized",
                        "outcome",
                        "target",
                        "label",
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.registry.write_text(
            json.dumps(
                {
                    "version": "test-features",
                    "features": [{"name": name} for name in self.feature_names],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.checkpoints.write_text(
            json.dumps(
                {
                    "version": 1,
                    "lane_id": "test-forward-lane",
                    "research_exposed_through": "2026-08-18",
                    "checkpoints": [],
                }
            ),
            encoding="utf-8",
        )
        write_ledger(self.ledger, self.feature_names, [])
        self._write_features(
            [
                {
                    "decision_date": "2026-08-19",
                    "fear_greed_date": "2026-08-19",
                    "treasury_date": "2026-08-18",
                    "f1": 1.25,
                    "f2": -0.5,
                },
                {
                    "decision_date": "2026-08-20",
                    "fear_greed_date": "2026-08-20",
                    "treasury_date": "2026-08-19",
                    "f1": 1.5,
                    "f2": -0.25,
                },
            ]
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_features(self, rows: list[dict[str, object]]) -> None:
        frame = pd.DataFrame(rows)
        for column in ("decision_date", "fear_greed_date", "treasury_date"):
            frame[column] = pd.to_datetime(frame[column])
        frame.to_parquet(self.features, index=False, engine="pyarrow")

    def _append(self, date: str) -> str:
        return append_snapshot(
            decision_date=date,
            features_path=self.features,
            registry_path=self.registry,
            manifest_path=self.manifest,
            ledger_path=self.ledger,
        )

    def _verify(self) -> dict[str, object]:
        return verify_lane(
            manifest_path=self.manifest,
            registry_path=self.registry,
            ledger_path=self.ledger,
            checkpoints_path=self.checkpoints,
        )

    def test_append_is_idempotent_and_chain_verifies(self) -> None:
        self.assertEqual(self._append("2026-08-19"), "APPENDED")
        self.assertEqual(self._append("2026-08-19"), "IDEMPOTENT")
        self.assertEqual(self._append("2026-08-20"), "APPENDED")
        report = self._verify()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["ledger_rows"], 2)
        self.assertEqual(report["first_ledger_date"], "2026-08-19")
        self.assertEqual(report["last_ledger_date"], "2026-08-20")

    def test_rejects_exposed_history_date(self) -> None:
        self._write_features(
            [
                {
                    "decision_date": "2026-08-18",
                    "fear_greed_date": "2026-08-18",
                    "treasury_date": "2026-08-18",
                    "f1": 1.0,
                    "f2": 2.0,
                }
            ]
        )
        with self.assertRaises(ValueError):
            self._append("2026-08-18")

    def test_rejects_future_source_date_and_outcome_column(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "decision_date": pd.Timestamp("2026-08-19"),
                    "fear_greed_date": pd.Timestamp("2026-08-20"),
                    "treasury_date": pd.Timestamp("2026-08-18"),
                    "f1": 1.0,
                    "f2": 2.0,
                }
            ]
        )
        frame.to_parquet(self.features, index=False, engine="pyarrow")
        with self.assertRaises(ValueError):
            self._append("2026-08-19")

        frame["fear_greed_date"] = pd.Timestamp("2026-08-19")
        frame["forward_return_20d"] = 0.1
        frame.to_parquet(self.features, index=False, engine="pyarrow")
        with self.assertRaises(ValueError):
            self._append("2026-08-19")

    def test_same_date_with_changed_snapshot_is_rejected(self) -> None:
        self.assertEqual(self._append("2026-08-19"), "APPENDED")
        self._write_features(
            [
                {
                    "decision_date": "2026-08-19",
                    "fear_greed_date": "2026-08-19",
                    "treasury_date": "2026-08-18",
                    "f1": 99.0,
                    "f2": -0.5,
                }
            ]
        )
        with self.assertRaises(ValueError):
            self._append("2026-08-19")

    def test_tampering_breaks_chain_verification(self) -> None:
        self._append("2026-08-19")
        with self.ledger.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fieldnames = list(rows[0].keys())
        rows[0]["f1"] = "999"
        with self.ledger.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(ValueError):
            self._verify()


if __name__ == "__main__":
    unittest.main()
