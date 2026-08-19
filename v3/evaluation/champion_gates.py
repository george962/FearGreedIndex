#!/usr/bin/env python3
"""Deterministic fail-closed champion acceptance gates for V3-018.

The gate engine is intentionally separate from model fitting, decision policy,
sizing, and execution. It consumes immutable evaluation evidence and decides
only whether a candidate is eligible for V3-019 champion selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "v3" / "evaluation" / "champion_gates_v1.json"
DEFAULT_CURRENT_EVIDENCE = ROOT / "v3" / "reports" / "champion_candidate_evidence.json"
DEFAULT_REPORT = ROOT / "v3" / "reports" / "champion_gate_assessment.json"

GATE_ORDER = (
    "evidence_completeness",
    "prediction_prerequisite",
    "calibration",
    "after_cost_portfolio",
    "cross_year_robustness",
    "sharpe",
    "maximum_drawdown",
    "cost_robustness",
    "parameter_robustness",
    "data_quality",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_CURRENT_EVIDENCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _finite_numbers(values: Iterable[Any]) -> bool:
    return all(_finite(value) for value in values)


def _gate(status: str, *, reason: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    if status not in {"PASS", "FAIL", "BLOCKED"}:
        raise ValueError(f"Unsupported gate status: {status}")
    return {
        "status": status,
        "reason": reason,
        "details": details or {},
    }


def _required_sections_present(evidence: dict[str, Any]) -> tuple[bool, list[str]]:
    required = (
        "candidate_id",
        "as_of",
        "lineage",
        "prediction",
        "calibration",
        "portfolio",
        "cross_year",
        "risk",
        "cost_robustness",
        "parameter_robustness",
        "data_quality",
    )
    missing = [key for key in required if key not in evidence]
    return not missing, missing


def _evaluate_evidence_completeness(evidence: dict[str, Any]) -> dict[str, Any]:
    sections_ok, missing_sections = _required_sections_present(evidence)
    lineage = evidence.get("lineage")
    if not isinstance(lineage, dict):
        lineage = {}
    required_lineage = (
        "feature_version",
        "model_version",
        "label_version",
        "training_version",
        "policy_version",
        "sizing_version",
        "evidence_hashes",
    )
    missing_lineage = [
        key
        for key in required_lineage
        if key not in lineage or lineage.get(key) in (None, "", [], {})
    ]
    hashes = lineage.get("evidence_hashes")
    hashes_valid = (
        isinstance(hashes, dict)
        and len(hashes) > 0
        and all(isinstance(value, str) and len(value) == 64 for value in hashes.values())
    )
    complete_flag = evidence.get("evidence_complete") is True
    passed = sections_ok and not missing_lineage and hashes_valid and complete_flag
    return _gate(
        "PASS" if passed else "FAIL",
        reason=(
            "Required champion evidence and lineage are complete."
            if passed
            else "Required champion evidence or lineage is missing/incomplete."
        ),
        details={
            "evidence_complete_flag": complete_flag,
            "missing_sections": missing_sections,
            "missing_lineage": missing_lineage,
            "evidence_hashes_valid": hashes_valid,
        },
    )


def _evaluate_prediction(evidence: dict[str, Any]) -> dict[str, Any]:
    prediction = evidence.get("prediction")
    if not isinstance(prediction, dict):
        return _gate("FAIL", reason="Prediction prerequisite evidence is missing.")
    ready = prediction.get("absolute_prediction_gate_pass") is True
    return _gate(
        "PASS" if ready else "FAIL",
        reason=(
            "Candidate passes the pre-existing absolute prediction-readiness gate."
            if ready
            else "Candidate does not pass the pre-existing absolute prediction-readiness gate."
        ),
        details={
            "absolute_prediction_gate_pass": ready,
            "source_gate_version": prediction.get("source_gate_version"),
        },
    )


def _blocked_after_prediction(gate_name: str) -> dict[str, Any]:
    return _gate(
        "BLOCKED",
        reason=(
            f"{gate_name} is not eligible for promotion evaluation because the "
            "prediction prerequisite failed; V3-017 extra sizing remains locked."
        ),
    )


def _evaluate_calibration(evidence: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    calibration = evidence.get("calibration")
    limits = config["calibration"]
    if not isinstance(calibration, dict):
        return _gate("FAIL", reason="Calibration evidence is missing.")
    values = (
        calibration.get("mean_ece"),
        calibration.get("max_fold_horizon_ece"),
        calibration.get("mean_relative_brier_improvement"),
    )
    if not _finite_numbers(values):
        return _gate("FAIL", reason="Calibration evidence contains missing or non-finite values.")
    mean_ece, max_ece, brier_improvement = map(float, values)
    passed = (
        mean_ece <= float(limits["mean_ece_max"])
        and max_ece <= float(limits["max_fold_horizon_ece"])
        and brier_improvement > float(limits["mean_relative_brier_improvement_min_exclusive"])
    )
    return _gate(
        "PASS" if passed else "FAIL",
        reason="Calibration thresholds pass." if passed else "Calibration thresholds fail.",
        details={
            "mean_ece": mean_ece,
            "max_fold_horizon_ece": max_ece,
            "mean_relative_brier_improvement": brier_improvement,
        },
    )


def _evaluate_after_cost_portfolio(evidence: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    portfolio = evidence.get("portfolio")
    limits = config["after_cost_portfolio"]
    if not isinstance(portfolio, dict):
        return _gate("FAIL", reason="After-cost portfolio evidence is missing.")
    values = (
        portfolio.get("base_cost_bps_per_1x_turnover"),
        portfolio.get("combined_excess_total_return"),
        portfolio.get("annualized_excess_return"),
    )
    if not _finite_numbers(values):
        return _gate("FAIL", reason="After-cost portfolio evidence is missing or non-finite.")
    cost, total_excess, annual_excess = map(float, values)
    passed = (
        math.isclose(cost, float(limits["base_cost_bps_per_1x_turnover"]), rel_tol=0.0, abs_tol=1e-12)
        and total_excess > float(limits["combined_excess_total_return_min_exclusive"])
        and annual_excess > float(limits["annualized_excess_return_min_exclusive"])
    )
    return _gate(
        "PASS" if passed else "FAIL",
        reason="Base-cost portfolio edge passes." if passed else "Base-cost portfolio edge fails.",
        details={
            "base_cost_bps_per_1x_turnover": cost,
            "combined_excess_total_return": total_excess,
            "annualized_excess_return": annual_excess,
        },
    )


def _evaluate_cross_year(evidence: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    cross_year = evidence.get("cross_year")
    limits = config["cross_year_robustness"]
    if not isinstance(cross_year, dict):
        return _gate("FAIL", reason="Cross-year evidence is missing.")
    fold_excess = cross_year.get("fold_excess_total_returns")
    if not isinstance(fold_excess, list) or len(fold_excess) != int(limits["expected_folds"]):
        return _gate("FAIL", reason="Cross-year evidence has the wrong number of folds.")
    if not _finite_numbers(fold_excess):
        return _gate("FAIL", reason="Cross-year fold evidence contains missing/non-finite values.")
    values = [float(value) for value in fold_excess]
    positive = [value for value in values if value > 0.0]
    positive_count = len(positive)
    if positive:
        positive_sum = sum(positive)
        max_share = max(positive) / positive_sum if positive_sum > 0.0 else math.inf
    else:
        max_share = math.inf
    passed = (
        positive_count >= int(limits["required_positive_folds"])
        and max_share <= float(limits["max_positive_excess_contribution_share"])
    )
    return _gate(
        "PASS" if passed else "FAIL",
        reason="Cross-year robustness passes." if passed else "Cross-year robustness fails.",
        details={
            "fold_excess_total_returns": values,
            "positive_fold_count": positive_count,
            "max_positive_excess_contribution_share": max_share,
        },
    )


def _evaluate_sharpe(evidence: dict[str, Any]) -> dict[str, Any]:
    risk = evidence.get("risk")
    if not isinstance(risk, dict):
        return _gate("FAIL", reason="Risk evidence is missing.")
    values = (
        risk.get("candidate_sharpe"),
        risk.get("benchmark_sharpe"),
        risk.get("v2_1_sharpe"),
    )
    if not _finite_numbers(values):
        return _gate("FAIL", reason="Sharpe evidence contains missing/non-finite values.")
    candidate, benchmark, v2_1 = map(float, values)
    passed = candidate >= benchmark and candidate >= v2_1
    return _gate(
        "PASS" if passed else "FAIL",
        reason="Sharpe is at least both references." if passed else "Sharpe is below a required reference.",
        details={
            "candidate_sharpe": candidate,
            "benchmark_sharpe": benchmark,
            "v2_1_sharpe": v2_1,
        },
    )


def _drawdown_ceiling(evidence: dict[str, Any], config: dict[str, Any]) -> float | None:
    risk = evidence.get("risk")
    if not isinstance(risk, dict):
        return None
    refs = (risk.get("benchmark_max_drawdown"), risk.get("v2_1_max_drawdown"))
    if not _finite_numbers(refs):
        return None
    worse_reference = max(float(refs[0]), float(refs[1]))
    hard_pp = float(config["parameter_robustness"]["hard_drawdown_ceiling_deterioration_pp"])
    return worse_reference + hard_pp / 100.0


def _evaluate_drawdown(evidence: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    risk = evidence.get("risk")
    limits = config["maximum_drawdown"]
    portfolio = evidence.get("portfolio")
    if not isinstance(risk, dict) or not isinstance(portfolio, dict):
        return _gate("FAIL", reason="Drawdown or return-benefit evidence is missing.")
    values = (
        risk.get("candidate_max_drawdown"),
        risk.get("benchmark_max_drawdown"),
        risk.get("v2_1_max_drawdown"),
        risk.get("candidate_calmar"),
        risk.get("benchmark_calmar"),
        risk.get("v2_1_calmar"),
        portfolio.get("annualized_excess_return"),
    )
    if not _finite_numbers(values):
        return _gate("FAIL", reason="Drawdown evidence contains missing/non-finite values.")
    (
        candidate_dd,
        benchmark_dd,
        v2_1_dd,
        candidate_calmar,
        benchmark_calmar,
        v2_1_calmar,
        annual_excess,
    ) = map(float, values)
    if min(candidate_dd, benchmark_dd, v2_1_dd) < 0.0:
        return _gate("FAIL", reason="Maximum drawdown must be supplied as non-negative magnitudes.")
    worse_reference = max(benchmark_dd, v2_1_dd)
    deterioration_pp = (candidate_dd - worse_reference) * 100.0
    simple_limit = float(limits["no_exception_deterioration_pp_max"])
    conditional_limit = float(limits["conditional_deterioration_pp_max"])
    min_benefit = float(limits["conditional_min_annualized_return_benefit_pp"])
    if deterioration_pp <= simple_limit:
        passed = True
        route = "within_no_exception_limit"
    elif deterioration_pp <= conditional_limit:
        passed = (
            annual_excess * 100.0 >= min_benefit
            and candidate_calmar >= benchmark_calmar
            and candidate_calmar >= v2_1_calmar
        )
        route = "conditional_exception"
    else:
        passed = False
        route = "hard_failure_above_conditional_limit"
    return _gate(
        "PASS" if passed else "FAIL",
        reason="Maximum-drawdown gate passes." if passed else "Maximum-drawdown gate fails.",
        details={
            "candidate_max_drawdown": candidate_dd,
            "worse_reference_max_drawdown": worse_reference,
            "deterioration_pp": deterioration_pp,
            "route": route,
            "candidate_calmar": candidate_calmar,
            "benchmark_calmar": benchmark_calmar,
            "v2_1_calmar": v2_1_calmar,
            "annualized_excess_return": annual_excess,
        },
    )


def _evaluate_cost_robustness(evidence: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    cost = evidence.get("cost_robustness")
    limits = config["cost_robustness"]
    if not isinstance(cost, dict):
        return _gate("FAIL", reason="Cost-robustness evidence is missing.")
    values = (
        cost.get("stress_5bps_excess_total_return"),
        cost.get("stress_10bps_annualized_excess_return"),
    )
    if not _finite_numbers(values):
        return _gate("FAIL", reason="Cost-robustness evidence is missing or non-finite.")
    excess_5, annual_10 = map(float, values)
    passed = (
        excess_5 > float(limits["stress_5bps_excess_total_return_min_exclusive"])
        and annual_10 >= float(limits["stress_10bps_annualized_excess_return_min_inclusive"])
    )
    return _gate(
        "PASS" if passed else "FAIL",
        reason="Cost stress passes." if passed else "Cost stress fails.",
        details={
            "stress_5bps_excess_total_return": excess_5,
            "stress_10bps_annualized_excess_return": annual_10,
        },
    )


def _evaluate_parameter_robustness(evidence: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    parameter = evidence.get("parameter_robustness")
    limits = config["parameter_robustness"]
    if not isinstance(parameter, dict):
        return _gate("FAIL", reason="Parameter-robustness evidence is missing.")
    scenarios = parameter.get("scenarios")
    expected = int(limits["expected_scenarios"])
    if not isinstance(scenarios, list) or len(scenarios) != expected:
        return _gate("FAIL", reason="Parameter robustness has the wrong scenario count.")
    ceiling = _drawdown_ceiling(evidence, config)
    if ceiling is None:
        return _gate("FAIL", reason="Cannot compute parameter-robustness drawdown ceiling.")

    positive_count = 0
    hard_violations: list[str] = []
    normalized: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            return _gate("FAIL", reason="Parameter scenario is not an object.")
        name = str(scenario.get("name", f"scenario_{index + 1}"))
        excess = scenario.get("combined_excess_total_return")
        drawdown = scenario.get("max_drawdown")
        if not _finite_numbers((excess, drawdown)):
            return _gate("FAIL", reason="Parameter scenario contains missing/non-finite values.")
        excess_value = float(excess)
        drawdown_value = float(drawdown)
        if drawdown_value < 0.0:
            return _gate("FAIL", reason="Parameter-scenario drawdown must be a non-negative magnitude.")
        if excess_value > float(limits["positive_excess_total_return_min_exclusive"]):
            positive_count += 1
        if drawdown_value > ceiling:
            hard_violations.append(name)
        normalized.append(
            {
                "name": name,
                "combined_excess_total_return": excess_value,
                "max_drawdown": drawdown_value,
            }
        )

    passed = (
        positive_count >= int(limits["required_positive_scenarios"])
        and not hard_violations
    )
    return _gate(
        "PASS" if passed else "FAIL",
        reason="Parameter perturbations pass." if passed else "Parameter perturbations fail.",
        details={
            "positive_scenario_count": positive_count,
            "hard_drawdown_ceiling": ceiling,
            "hard_drawdown_violations": hard_violations,
            "scenarios": normalized,
        },
    )


def _evaluate_data_quality(evidence: dict[str, Any]) -> dict[str, Any]:
    quality = evidence.get("data_quality")
    if not isinstance(quality, dict):
        return _gate("FAIL", reason="Data-quality evidence is missing.")
    required = (
        "point_in_time_pass",
        "leakage_gate_pass",
        "sample_hashes_match",
        "frozen_v2_1_reproducible",
    )
    passed = all(quality.get(key) is True for key in required)
    return _gate(
        "PASS" if passed else "FAIL",
        reason="Data-quality and leakage gates pass." if passed else "A data-quality/leakage gate failed or is missing.",
        details={key: quality.get(key) for key in required},
    )


def evaluate_candidate(evidence: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if config.get("gate_version") != "v3-champion-gates-001":
        raise ValueError("Unsupported or unfrozen champion gate version")
    if config.get("fail_closed") is not True:
        raise ValueError("Champion gate config must be fail-closed")

    gates: dict[str, dict[str, Any]] = {}
    gates["evidence_completeness"] = _evaluate_evidence_completeness(evidence)
    gates["prediction_prerequisite"] = _evaluate_prediction(evidence)

    prediction_pass = gates["prediction_prerequisite"]["status"] == "PASS"
    if not prediction_pass:
        for name in (
            "calibration",
            "after_cost_portfolio",
            "cross_year_robustness",
            "sharpe",
            "maximum_drawdown",
            "cost_robustness",
            "parameter_robustness",
        ):
            gates[name] = _blocked_after_prediction(name)
    else:
        gates["calibration"] = _evaluate_calibration(evidence, config)
        gates["after_cost_portfolio"] = _evaluate_after_cost_portfolio(evidence, config)
        gates["cross_year_robustness"] = _evaluate_cross_year(evidence, config)
        gates["sharpe"] = _evaluate_sharpe(evidence)
        gates["maximum_drawdown"] = _evaluate_drawdown(evidence, config)
        gates["cost_robustness"] = _evaluate_cost_robustness(evidence, config)
        gates["parameter_robustness"] = _evaluate_parameter_robustness(evidence, config)

    gates["data_quality"] = _evaluate_data_quality(evidence)

    ordered = {name: gates[name] for name in GATE_ORDER}
    promotion_ready = all(gate["status"] == "PASS" for gate in ordered.values())
    failures = [name for name, gate in ordered.items() if gate["status"] != "PASS"]
    return {
        "gate_version": config["gate_version"],
        "candidate_id": evidence.get("candidate_id"),
        "as_of": evidence.get("as_of"),
        "status": "PROMOTION_READY" if promotion_ready else "NOT_PROMOTION_READY",
        "promotion_ready": promotion_ready,
        "champion_selected": False,
        "v3_019_eligible": promotion_ready,
        "failed_or_blocked_gates": failures,
        "gates": ordered,
        "note": (
            "V3-018 determines promotion eligibility only. V3-019 must still select exactly one passing candidate before any champion manifest or extra sizing can activate."
        ),
    }


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    evidence = load_json(args.evidence)
    report = evaluate_candidate(evidence, config)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
