from __future__ import annotations

import unittest

from v3.policy.sizing_policy import load_config, size_action


class SizingPolicyTests(unittest.TestCase):
    def test_unvalidated_strong_add_is_blocked_at_baseline(self) -> None:
        result = size_action(
            "STRONG ADD",
            promotion_ready_prediction=False,
            experiment_id="COMBO-EXP-004",
            decision_policy_version="v3-decision-policy-001",
        )
        self.assertEqual(result["multiplier"], 1.0)
        self.assertTrue(result["activation_blocked"])
        self.assertEqual(result["reason_code"], "PREDICTION_PROMOTION_GATE_NOT_MET")

    def test_validated_strong_add_can_use_only_1_10x(self) -> None:
        result = size_action(
            "STRONG ADD",
            promotion_ready_prediction=True,
            experiment_id="future-promoted-model",
            decision_policy_version="v3-decision-policy-001",
        )
        self.assertEqual(result["multiplier"], 1.1)
        self.assertFalse(result["activation_blocked"])

    def test_other_actions_always_remain_baseline(self) -> None:
        for action in ("ADD MODESTLY", "BASELINE", "WAIT FOR BETTER ENTRY"):
            with self.subTest(action=action):
                result = size_action(
                    action,
                    promotion_ready_prediction=True,
                    experiment_id="future-promoted-model",
                    decision_policy_version="v3-decision-policy-001",
                )
                self.assertEqual(result["multiplier"], 1.0)

    def test_underweight_and_larger_sizing_are_disabled(self) -> None:
        config = load_config()
        self.assertEqual(config["minimum_multiplier"], 1.0)
        self.assertEqual(config["maximum_multiplier"], 1.1)
        self.assertFalse(config["underweight_allowed"])
        self.assertFalse(config["larger_sizing_allowed"])

    def test_unknown_action_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown decision action"):
            size_action(
                "SELL",
                promotion_ready_prediction=True,
                experiment_id="x",
                decision_policy_version="v3-decision-policy-001",
            )

    def test_sizing_is_deterministic(self) -> None:
        kwargs = dict(
            action="STRONG ADD",
            promotion_ready_prediction=False,
            experiment_id="COMBO-EXP-004",
            decision_policy_version="v3-decision-policy-001",
        )
        self.assertEqual(size_action(**kwargs), size_action(**kwargs))


if __name__ == "__main__":
    unittest.main()
