import copy
import unittest

from v3.evidence.shadow_predictions import (
    BASE_FIELDS,
    finalize_candidate,
    reconcile_prediction_rows,
    verify_hash_chain,
)


class ShadowPredictionLedgerTests(unittest.TestCase):
    def candidate(self, decision_date: str, *, percentile: str = "0.75") -> dict[str, str]:
        values = {
            "decision_date": decision_date,
            "research_cutoff": "2026-08-18",
            "method_id": "STAB-004",
            "method_status": "complete_reject",
            "feature_set_version": "v3-features-004-treasury",
            "forward_feature_row_sha256": "a" * 64,
            "feature_vector_sha256": "b" * 64,
            "source_feature_sha256": "c" * 64,
            "feature_registry_sha256": "d" * 64,
            "method_evaluation_sha256": "e" * 64,
            "method_manifest_sha256": "f" * 64,
            "representative_set_sha256": "1" * 64,
            "representative_feature_count": "3",
            "representative_features": "alpha|beta|gamma",
            "raw_score": "0.125",
            "rolling_percentile": percentile,
            "reference_count": "252",
            "call_state": "ABSTAIN",
            "collector_version": "v3-shadow-prediction-v1",
            "production_effect": "NONE",
            "sizing_multiplier": "1",
        }
        self.assertEqual(set(values), set(BASE_FIELDS))
        return values

    def test_reconcile_appends_and_is_idempotent(self):
        candidates = [
            self.candidate("2026-08-19", percentile="0.80"),
            self.candidate("2026-08-20", percentile="0.65"),
        ]
        rows, appended = reconcile_prediction_rows([], candidates)
        self.assertEqual(appended, 2)
        verify_hash_chain(rows)

        repeated, appended_again = reconcile_prediction_rows(rows, candidates)
        self.assertEqual(appended_again, 0)
        self.assertEqual(repeated, rows)

    def test_existing_prediction_cannot_be_rewritten(self):
        candidate = self.candidate("2026-08-19")
        row = finalize_candidate(candidate, "0" * 64)
        changed = self.candidate("2026-08-19", percentile="0.90")
        with self.assertRaisesRegex(ValueError, "Immutable shadow prediction drift"):
            reconcile_prediction_rows([row], [changed])

    def test_hash_chain_detects_tampering(self):
        candidates = [
            self.candidate("2026-08-19"),
            self.candidate("2026-08-20"),
        ]
        rows, _ = reconcile_prediction_rows([], candidates)
        tampered = copy.deepcopy(rows)
        tampered[0]["raw_score"] = "0.999"
        with self.assertRaisesRegex(ValueError, "row hash mismatch"):
            verify_hash_chain(tampered)

    def test_ledger_must_be_gap_free_prefix(self):
        candidates = [
            self.candidate("2026-08-19"),
            self.candidate("2026-08-20"),
            self.candidate("2026-08-21"),
        ]
        rows, _ = reconcile_prediction_rows([], candidates[:2])
        skipped = [candidates[0], candidates[2]]
        with self.assertRaisesRegex(ValueError, "exact prefix"):
            reconcile_prediction_rows(rows, skipped)


if __name__ == "__main__":
    unittest.main()
