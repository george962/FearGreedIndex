"""Regression gates for the frozen pre-cutoff STAB-004 shadow context."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from v3.evidence import frozen_training, shadow_predictions
from v3.data_sources.fetch_treasury import compressed_snapshot_bytes


class FrozenTrainingTests(unittest.TestCase):
    def test_frozen_snapshots_exactly_match_original_git_blobs(self):
        fingerprints = frozen_training.verify_frozen_inputs()
        self.assertEqual(fingerprints, frozen_training.FROZEN_SOURCE_BLOBS)

    def test_reject_changed_frozen_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            altered = Path(tmp) / "fear_greed_daily.csv"
            altered.write_bytes(frozen_training.FROZEN_FEAR.read_bytes() + b"\n")
            with patch.object(frozen_training, "FROZEN_FEAR", altered):
                with self.assertRaisesRegex(ValueError, "was modified"):
                    frozen_training.verify_frozen_inputs()

    def test_all_sealed_shadow_rows_replay_identically(self):
        # Run the real verifier, not just tests of a mocked hash.
        ledger = shadow_predictions.PREDICTION_LEDGER
        before = hashlib.sha256(ledger.read_bytes()).hexdigest()
        result = shadow_predictions.verify_shadow_predictions()
        self.assertEqual(before, hashlib.sha256(ledger.read_bytes()).hexdigest())
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["outcomes_present"])
        self.assertFalse(result["champion_selected"])
        self.assertEqual(result["production_effect"], "NONE")

    def test_treasury_gzip_remains_original_frozen_bytes(self):
        root = frozen_training.ROOT
        manifest = json.loads((root / "v3/data/treasury_source.json").read_text())
        data = (root / "v3/data/treasury_daily.csv").read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), manifest["normalized_sha256"])
        compressed = compressed_snapshot_bytes(data)
        self.assertEqual(compressed[9], 3)
        self.assertEqual(hashlib.sha256(compressed).hexdigest(), manifest["snapshot_sha256"])


if __name__ == "__main__":
    unittest.main()
