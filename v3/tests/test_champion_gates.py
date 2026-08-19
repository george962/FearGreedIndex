from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path

from v3.evaluation.champion_gates import evaluate_candidate

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (ROOT / "v3" / "evaluation" / "champion_gates_v1.json").read_text(encoding="utf-8")
)


def passing_evidence() -> dict[str, object]:
    return {
        "candidate_id": "TEST-EXP-004",
        "as_of": "2026-08-18",
        "evidence_complete": True,
        "lineage": {
            "feature_version": "features-test",
            "model_version": "model-test",
            "label_version": "labels-test",
            "training_version": "training-test",
            "policy_version": "v3-decision-policy-001",
            "sizing_version": "v3-sizing-policy-001",
            "evidence_hashes": {
                "prediction": "a" * 64,
                "portfolio": "b" * 64,
                "policy": "c" * 64,
            },
        },
        "prediction": {
            "absolute_prediction_gate_pass": True,
            "source_gate_version": "v3-tournament-001",
        },
        "calibration": {
            "mean_ece": 0.08,
            "max_fold_horizon_ece": 0.18,
            "mean_relative_brier_improvement": 0.02,
        },
        "portfolio": {
            "base_cost_bps_per_1x_turnover": 2.0,
            "combined_excess_total_return": 0.06,
            "annualized_excess_return": 0.035,
        },
        "cross_year": {
            "fold_excess_total_returns": [0.02, 0.015, -0.005],
        },
        "risk": {
            "candidate_sharpe": 1.15,
            "benchmark_sharpe": 1.0,
            "v2_1_sharpe": 0.95,
            "candidate_max_drawdown": 0.20,
            "benchmark_max_drawdown": 0.19,
            "v2_1_max_drawdown": 0.18,
            "candidate_calmar": 1.10,
            "benchmark_calmar": 0.90,
            "v2_1_calmar": 0.85,
        },
        "cost_robustness": {
            "stress_5bps_excess_total_return": 0.025,
            "stress_10bps_annualized_excess_return": 0.001,
        },
        "parameter_robustness": {
            "scenarios": [
                {
                    "name": "return_stricter",
                    "combined_excess_total_return": 0.03,
                    "max_drawdown": 0.20,
                },
                {
                    "name": "return_looser",
                    "combined_excess_total_return": 0.025,
                    "max_drawdown": 0.21,
                },
                {
                    "name": "risk_stricter",
                    "combined_excess_total_return": 0.015,
                    "max_drawdown": 0.20,
                },
                {
                    "name": "risk_looser",
                    "combined_excess_total_return": -0.002,
                    "max_drawdown": 0.22,
                },
            ],
        },
        "data_quality": {
            "point_in_time_pass": True,
            "leakage_gate_pass": True,
            "sample_hashes_match": True,
            "frozen_v2_1_reproducible": True,
        },
    }


class ChampionGateTests(unittest.TestCase):
    def test_fully_passing_evidence_is_promotion_ready(self) -> None:
        report = evaluate_candidate(passing_evidence(), CONFIG)
        self.assertTrue(report["promotion_ready"])
        self.assertEqual(report["status"], "PROMOTION_READY")
        self.assertEqual(report["failed_or_blocked_gates"], [])
        self.assertFalse(report["champion_selected"])
        self.assertTrue(report["v3_019_eligible"])

    def test_prediction_failure_blocks_sizing_dependent_gates(self) -> None:
        evidence = passing_evidence()
        evidence["prediction"]["absolute_prediction_gate_pass"] = False  # type: ignore[index]
        report = evaluate_candidate(evidence, CONFIG)
        self.assertFalse(report["promotion_ready"])
        self.assertEqual(report["gates"]["prediction_prerequisite"]["status"], "FAIL")
        for name in (
            "calibration",
            "after_cost_portfolio",
            "cross_year_robustness",
            "sharpe",
            "maximum_drawdown",
            "cost_robustness",
            "parameter_robustness",
        ):
            self.assertEqual(report["gates"][name]["status"], "BLOCKED")

    def test_missing_evidence_fails_closed(self) -> None:
        evidence = passing_evidence()
        evidence["evidence_complete"] = False
        del evidence["lineage"]["evidence_hashes"]  # type: ignore[index]
        report = evaluate_candidate(evidence, CONFIG)
        self.assertEqual(report["gates"]["evidence_completeness"]["status"], "FAIL")
        self.assertFalse(report["promotion_ready"])

    def test_nan_calibration_fails_closed(self) -> None:
        evidence = passing_evidence()
        evidence["calibration"]["mean_ece"] = math.nan  # type: ignore[index]
        report = evaluate_candidate(evidence, CONFIG)
        self.assertEqual(report["gates"]["calibration"]["status"], "FAIL")

    def test_calibration_boundaries_are_inclusive_except_improvement(self) -> None:
        evidence = passing_evidence()
        evidence["calibration"] = {
            "mean_ece": 0.10,
            "max_fold_horizon_ece": 0.20,
            "mean_relative_brier_improvement": 0.000001,
        }
        report = evaluate_candidate(evidence, CONFIG)
        self.assertEqual(report["gates"]["calibration"]["status"], "PASS")
        evidence["calibration"]["mean_relative_brier_improvement"] = 0.0  # type: ignore[index]
        report = evaluate_candidate(evidence, CONFIG)
        self.assertEqual(report["gates"]["calibration"]["status"], "FAIL")

    def test_base_portfolio_requires_exact_registered_cost_and_positive_edge(self) -> None:
        evidence = passing_evidence()
        evidence["portfolio"]["base_cost_bps_per_1x_turnover"] = 2.1  # type: ignore[index]
        report = evaluate_candidate(evidence, CONFIG)
        self.assertEqual(report["gates"]["after_cost_portfolio"]["status"], "FAIL")
        evidence = passing_evidence()
        evidence["portfolio"]["combined_excess_total_return"] = 0.0  # type: ignore[index]
        report = evaluate_candidate(evidence, CONFIG)
        self.assertEqual(report["gates"]["after_cost_portfolio"]["status"], "FAIL")

    def test_cross_year_requires_two_positive_folds(self) -> None:
        evidence = passing_evidence()
        evidence["cross_year"]["fold_excess_total_returns"] = [0.03, -0.01, -0.02]  # type: ignore[index]
        report = evaluate_candidate(evidence, CONFIG)
        self.assertEqual(report["gates"]["cross_year_robustness"]["status"], "FAIL")

    def test_cross_year_seventy_percent_share_passes_but_more_fails(self) -> None:
        evidence = passing_evidence()
        evidence["cross_year"]["fold_excess_total_returns"] = [0.07, 0.03, -0.01]  # type: ignore[index]
        report = evaluate_candidate(evidence, CONFIG)
        self.assertEqual(report["gates"]["cross_year_robustness"]["status"], "PASS")
        evidence["cross_year"]["fold_excess_total_returns"] = [0.071, 0.029, -0.01]  # type: ignore[index]
        report = evaluate_candidate(evidence, CONFIG)
        self.assertEqual(report["gates"]["cross_year_robustness"]["status"], "FAIL")

    def test_sharpe_must_match_or_exceed_both_references(self) -> None:
        evidence = passing_evidence()
        evidence["risk"]["candidate_sharpe"] = 0.99  # type: ignore[index]
        report = evaluate_candidate(evidence, CONFIG)
        self.assertEqual(report["gates"]["sharpe"]["status"], "FAIL")

    def test_drawdown_within_two_percentage_points_passes(self) -> None:
        evidence = passing_evidence()
        evidence["risk"]["candidate_max_drawdown"] = 0.21  # type: ignore[index]
        evidence["risk"]["benchmark_max_drawdown"] = 0.19  # type: ignore[index]
        evidence["risk"]["v2_1_max_drawdown"] = 0.18  # type: ignore[index]
        report = evaluate_candidate(evidence, CONFIG)
        self.assertEqual(report["gates"]["maximum_drawdown"]["status"], "PASS")

    def test_drawdown_three_point_deterioration_needs_return_and_calmar_exception(self) -> None:
        evidence = passing_evidence()
        evidence["risk"]["candidate_max_drawdown"] = 0.22  # type: ignore[index]
        evidence["risk"]["benchmark_max_drawdown"] = 0.19  # type: ignore[index]
        evidence["risk"]["v2_1_max_drawdown"] = 0.18  # type: ignore[index]
        evidence["portfolio"]["annualized_excess_return"] = 0.03  # type: ignore[index]
        evidence["risk"]["candidate_calmar"] = 0.90  # type: ignore[index]
        evidence["risk"]["benchmark_calmar"] = 0.90  # type: ignore[index]
        evidence["risk"]["v2_1_calmar"] = 0.85  # type: ignore[index]
        report = evaluate_candidate(evidence, CONFIG)
        self.assertEqual(report["gates"]["maximum_drawdown"]["status"], "PASS")
        evidence["risk"]["candidate_calmar"] = 0.89  # type: ignore[index]
        report = evaluate_candidate(evidence, CONFIG)
        self.assertEqual(report["gates"]["maximum_drawdown"]["status"], "FAIL")

    def test_drawdown_worse_than_five_points_always_fails(self) -> None:
        evidence = passing_evidence()
        evidence["risk"]["candidate_max_drawdown"] = 0.251  # type: ignore[index]
        evidence["risk"]["benchmark_max_drawdown"] = 0.20  # type: ignore[index]
        evidence["risk"]["v2_1_max_drawdown"] = 0.19  # type: ignore[index]
        evidence["portfolio"]["annualized_excess_return"] = 0.50  # type: ignore[index]
        evidence["risk"]["candidate_calmar"] = 10.0  # type: ignore[index]
        report = evaluate_candidate(evidence, CONFIG)
        self.assertEqual(report["gates"]["maximum_drawdown"]["status"], "FAIL")

    def test_cost_robustness_requires_positive_5bps_and_nonnegative_10bps(self) -> None:
        evidence = passing_evidence()
        evidence["cost_robustness"] = {
            "stress_5bps_excess_total_return": 0.00001,
            "stress_10bps_annualized_excess_return": 0.0,
        }
        report = evaluate_candidate(evidence, CONFIG)
        self.assertEqual(report["gates"]["cost_robustness"]["status"], "PASS")
        evidence["cost_robustness"]["stress_5bps_excess_total_return"] = 0.0  # type: ignore[index]
        report = evaluate_candidate(evidence, CONFIG)
        self.assertEqual(report["gates"]["cost_robustness"]["status"], "FAIL")

    def test_parameter_robustness_requires_three_positive_and_no_hard_drawdown_violation(self) -> None:
        evidence = passing_evidence()
        scenarios = copy.deepcopy(evidence["parameter_robustness"]["scenarios"])  # type: ignore[index]
        scenarios[2]["combined_excess_total_return"] = -0.01
        evidence["parameter_robustness"]["scenarios"] = scenarios  # type: ignore[index]
        report = evaluate_candidate(evidence, CONFIG)
        self.assertEqual(report["gates"]["parameter_robustness"]["status"], "FAIL")

        evidence = passing_evidence()
        scenarios = copy.deepcopy(evidence["parameter_robustness"]["scenarios"])  # type: ignore[index]
        scenarios[0]["max_drawdown"] = 0.241
        evidence["parameter_robustness"]["scenarios"] = scenarios  # type: ignore[index]
        report = evaluate_candidate(evidence, CONFIG)
        self.assertEqual(report["gates"]["parameter_robustness"]["status"], "FAIL")

    def test_data_quality_failure_blocks_promotion(self) -> None:
        evidence = passing_evidence()
        evidence["data_quality"]["sample_hashes_match"] = False  # type: ignore[index]
        report = evaluate_candidate(evidence, CONFIG)
        self.assertEqual(report["gates"]["data_quality"]["status"], "FAIL")
        self.assertFalse(report["promotion_ready"])

    def test_gate_version_is_immutable(self) -> None:
        config = copy.deepcopy(CONFIG)
        config["gate_version"] = "v3-champion-gates-002"
        with self.assertRaises(ValueError):
            evaluate_candidate(passing_evidence(), config)


if __name__ == "__main__":
    unittest.main()
