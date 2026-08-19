from __future__ import annotations

import unittest

from v3.evaluation.build_current_champion_evidence import build_evidence


class CurrentChampionEvidenceTests(unittest.TestCase):
    def test_current_candidate_lineage_and_gate_state_are_fail_closed(self) -> None:
        evidence = build_evidence()
        self.assertEqual(evidence["candidate_id"], "UST-EXP-004")
        self.assertEqual(evidence["as_of"], "2026-08-18")
        self.assertFalse(evidence["evidence_complete"])
        self.assertEqual(evidence["lineage"]["feature_version"], "v3-features-004-treasury")
        self.assertEqual(evidence["lineage"]["model_version"], "random_forest_v1")
        self.assertEqual(evidence["lineage"]["training_version"], "EXP-004")
        self.assertEqual(evidence["lineage"]["label_version"], "v3-labels-001")
        self.assertFalse(evidence["prediction"]["absolute_prediction_gate_pass"])
        self.assertEqual(evidence["sizing_activation"], "BLOCKED")
        self.assertEqual(evidence["current_multiplier"], 1.0)
        self.assertTrue(evidence["data_quality"]["sample_hashes_match"])
        self.assertTrue(evidence["data_quality"]["frozen_v2_1_reproducible"])


if __name__ == "__main__":
    unittest.main()
