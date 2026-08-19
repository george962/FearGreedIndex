#!/usr/bin/env python3
"""Deterministic V3-016 prediction-to-action research policy.

This module does not train models, size positions, sell, underweight, or place
orders. It maps standardized prediction outputs to a research action while
preserving lineage and explicit reason codes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = ROOT / "v3" / "policy" / "policy_v1.json"
ACTIONS = (
    "STRONG ADD",
    "ADD MODESTLY",
    "BASELINE",
    "WAIT FOR BETTER ENTRY",
)


@dataclass(frozen=True)
class PredictionInput:
    decision_date: str
    expected_return_20d: float
    probability_up_20d: float
    predicted_drawdown_20d: float
    probability_further_5pct_decline_20d: float
    uncertainty_score: float
    calibration_quality: float
    experiment_id: str
    model_name: str
    model_version: str
    feature_set_version: str
    label_version: str
    training_cutoff: str
    prediction_sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("status") != "research_only":
        raise ValueError("V3-016 policy must remain research_only")
    return policy


def _validate_probability(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")


def validate_prediction(item: PredictionInput) -> None:
    for name in ("expected_return_20d", "predicted_drawdown_20d"):
        value = float(getattr(item, name))
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    for name in (
        "probability_up_20d",
        "probability_further_5pct_decline_20d",
        "uncertainty_score",
        "calibration_quality",
    ):
        _validate_probability(name, float(getattr(item, name)))
    for name in (
        "decision_date",
        "experiment_id",
        "model_name",
        "model_version",
        "feature_set_version",
        "label_version",
        "training_cutoff",
        "prediction_sha256",
    ):
        if not str(getattr(item, name)).strip():
            raise ValueError(f"{name} is required")


def _strong_add(item: PredictionInput, settings: dict[str, Any]) -> bool:
    return (
        item.expected_return_20d >= settings["minimum_expected_return_20d"]
        and item.probability_up_20d >= settings["minimum_probability_up_20d"]
        and item.predicted_drawdown_20d >= settings["minimum_predicted_drawdown_20d"]
        and item.probability_further_5pct_decline_20d
        <= settings["maximum_probability_further_5pct_decline_20d"]
        and item.calibration_quality >= settings["minimum_calibration_quality"]
        and item.uncertainty_score <= settings["maximum_uncertainty_score"]
    )


def _add_modestly(item: PredictionInput, settings: dict[str, Any]) -> bool:
    return (
        item.expected_return_20d >= settings["minimum_expected_return_20d"]
        and item.probability_up_20d >= settings["minimum_probability_up_20d"]
        and item.predicted_drawdown_20d >= settings["minimum_predicted_drawdown_20d"]
        and item.probability_further_5pct_decline_20d
        <= settings["maximum_probability_further_5pct_decline_20d"]
    )


def _wait_reasons(item: PredictionInput, settings: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if item.expected_return_20d <= settings["maximum_expected_return_20d"]:
        reasons.append("EXPECTED_RETURN_WEAK")
    if item.probability_up_20d <= settings["maximum_probability_up_20d"]:
        reasons.append("PROBABILITY_UP_WEAK")
    if item.predicted_drawdown_20d <= settings["maximum_predicted_drawdown_20d"]:
        reasons.append("PREDICTED_DRAWDOWN_HIGH")
    if (
        item.probability_further_5pct_decline_20d
        >= settings["minimum_probability_further_5pct_decline_20d"]
    ):
        reasons.append("FURTHER_DECLINE_RISK_HIGH")
    return reasons


def decide(
    item: PredictionInput,
    *,
    policy: dict[str, Any] | None = None,
    policy_path: Path = DEFAULT_POLICY_PATH,
) -> dict[str, Any]:
    validate_prediction(item)
    active = load_policy(policy_path) if policy is None else policy
    confidence = active["confidence_gate"]

    low_confidence_reasons: list[str] = []
    if item.calibration_quality < confidence["minimum_calibration_quality"]:
        low_confidence_reasons.append("CALIBRATION_QUALITY_LOW")
    if item.uncertainty_score > confidence["maximum_uncertainty_score"]:
        low_confidence_reasons.append("UNCERTAINTY_HIGH")

    if low_confidence_reasons:
        action = "BASELINE"
        reasons = low_confidence_reasons
    else:
        wait_reasons = _wait_reasons(item, active["wait_for_better_entry"])
        if wait_reasons:
            action = "WAIT FOR BETTER ENTRY"
            reasons = wait_reasons
        elif _strong_add(item, active["strong_add"]):
            action = "STRONG ADD"
            reasons = ["RETURN_UPSIDE_STRONG", "DOWNSIDE_RISK_ACCEPTABLE", "CONFIDENCE_HIGH"]
        elif _add_modestly(item, active["add_modestly"]):
            action = "ADD MODESTLY"
            reasons = ["RETURN_UPSIDE_POSITIVE", "DOWNSIDE_RISK_ACCEPTABLE"]
        else:
            action = "BASELINE"
            reasons = ["NO_ACTION_THRESHOLD_MET"]

    if action not in ACTIONS:
        raise AssertionError(f"Unexpected policy action: {action}")

    result = {
        "policy_version": active["policy_version"],
        "policy_status": active["status"],
        "policy_sha256": _sha256(policy_path) if policy is None else None,
        "action": action,
        "reason_codes": reasons,
        "action_semantics": active["action_semantics"][action],
        "decision_date": item.decision_date,
        "model_lineage": {
            "experiment_id": item.experiment_id,
            "model_name": item.model_name,
            "model_version": item.model_version,
            "feature_set_version": item.feature_set_version,
            "label_version": item.label_version,
            "training_cutoff": item.training_cutoff,
            "prediction_sha256": item.prediction_sha256,
        },
        "prediction_inputs": {
            "expected_return_20d": item.expected_return_20d,
            "probability_up_20d": item.probability_up_20d,
            "predicted_drawdown_20d": item.predicted_drawdown_20d,
            "probability_further_5pct_decline_20d": item.probability_further_5pct_decline_20d,
            "uncertainty_score": item.uncertainty_score,
            "calibration_quality": item.calibration_quality,
        },
        "sizing_defined": False,
        "sell_or_underweight_allowed": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = decide(PredictionInput(**payload), policy_path=args.policy)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
