from __future__ import annotations

import unittest

from v3.policy.decision_policy import ACTIONS, PredictionInput, decide, load_policy


class DecisionPolicyTests(unittest.TestCase):
    @staticmethod
    def _input(**overrides) -> PredictionInput:
        values = {
            "decision_date": "2026-08-18",
            "expected_return_20d": 0.02,
            "probability_up_20d": 0.60,
            "predicted_drawdown_20d": -0.08,
            "probability_further_5pct_decline_20d": 0.30,
            "uncertainty_score": 0.30,
            "calibration_quality": 0.75,
            "experiment_id": "COMBO-EXP-004",
            "model_name": "random_forest_v1",
            "model_version": "candidate-only",
            "feature_set_version": "v3-features-006-retained-combined",
            "label_version": "v3-labels-001",
            "training_cutoff": "2025-12-31",
            "prediction_sha256": "abc123",
        }
        values.update(overrides)
        return PredictionInput(**values)

    def test_strong_add_path(self) -> None:
        result = decide(self._input(
            expected_return_20d=0.04,
            probability_up_20d=0.70,
            predicted_drawdown_20d=-0.06,
            probability_further_5pct_decline_20d=0.20,
            uncertainty_score=0.25,
            calibration_quality=0.80,
        ))
        self.assertEqual(result["action"], "STRONG ADD")

    def test_add_modestly_path(self) -> None:
        result = decide(self._input())
        self.assertEqual(result["action"], "ADD MODESTLY")

    def test_wait_path_does_not_mean_sell(self) -> None:
        result = decide(self._input(
            expected_return_20d=-0.02,
            probability_up_20d=0.40,
            predicted_drawdown_20d=-0.16,
            probability_further_5pct_decline_20d=0.65,
        ))
        self.assertEqual(result["action"], "WAIT FOR BETTER ENTRY")
        self.assertFalse(result["sell_or_underweight_allowed"])
        self.assertIn("maintain baseline", result["action_semantics"].lower())

    def test_mixed_signal_stays_baseline(self) -> None:
        result = decide(self._input(
            expected_return_20d=0.005,
            probability_up_20d=0.52,
            predicted_drawdown_20d=-0.10,
            probability_further_5pct_decline_20d=0.50,
        ))
        self.assertEqual(result["action"], "BASELINE")

    def test_low_calibration_reduces_to_baseline(self) -> None:
        result = decide(self._input(
            expected_return_20d=0.05,
            probability_up_20d=0.75,
            predicted_drawdown_20d=-0.04,
            probability_further_5pct_decline_20d=0.15,
            calibration_quality=0.40,
        ))
        self.assertEqual(result["action"], "BASELINE")
        self.assertIn("CALIBRATION_QUALITY_LOW", result["reason_codes"])

    def test_high_uncertainty_reduces_to_baseline(self) -> None:
        result = decide(self._input(uncertainty_score=0.80))
        self.assertEqual(result["action"], "BASELINE")
        self.assertIn("UNCERTAINTY_HIGH", result["reason_codes"])

    def test_exact_strong_boundaries_are_inclusive(self) -> None:
        policy = load_policy()
        strong = policy["strong_add"]
        result = decide(self._input(
            expected_return_20d=strong["minimum_expected_return_20d"],
            probability_up_20d=strong["minimum_probability_up_20d"],
            predicted_drawdown_20d=strong["minimum_predicted_drawdown_20d"],
            probability_further_5pct_decline_20d=strong["maximum_probability_further_5pct_decline_20d"],
            calibration_quality=strong["minimum_calibration_quality"],
            uncertainty_score=strong["maximum_uncertainty_score"],
        ))
        self.assertEqual(result["action"], "STRONG ADD")

    def test_output_has_no_sizing(self) -> None:
        result = decide(self._input())
        self.assertFalse(result["sizing_defined"])
        self.assertNotIn("multiplier", result)
        self.assertNotIn("exposure", result)

    def test_same_input_is_deterministic(self) -> None:
        item = self._input()
        self.assertEqual(decide(item), decide(item))

    def test_invalid_probability_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "probability_up_20d"):
            decide(self._input(probability_up_20d=1.1))

    def test_action_vocabulary_is_exact(self) -> None:
        self.assertEqual(
            ACTIONS,
            ("STRONG ADD", "ADD MODESTLY", "BASELINE", "WAIT FOR BETTER ENTRY"),
        )


if __name__ == "__main__":
    unittest.main()
