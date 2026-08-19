#!/usr/bin/env python3
"""Build the current candidate's V3-018 evidence from immutable repaired reports.

The current retained research candidate is intentionally NOT allowed to invent
portfolio evidence: if the existing absolute prediction gate has not passed,
V3-017 remains at 1.00x and downstream sizing-dependent champion evidence stays
blocked/incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "v3" / "reports" / "champion_candidate_evidence.json"
TREASURY_REPORT = ROOT / "v3" / "reports" / "treasury_ablation.json"
INTEGRITY_REPORT = ROOT / "v3" / "reports" / "integrity_rebuild_summary.json"
POLICY_MANIFEST = ROOT / "v3" / "reports" / "decision_policy_manifest.json"
SIZING_MANIFEST = ROOT / "v3" / "reports" / "sizing_policy_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_evidence() -> dict[str, Any]:
    treasury = load_json(TREASURY_REPORT)
    integrity = load_json(INTEGRITY_REPORT)
    policy = load_json(POLICY_MANIFEST)
    sizing = load_json(SIZING_MANIFEST)

    candidate_id = str(treasury["best_ranked_full_candidate_in_ablation_tournament"])
    promotion_ready = treasury.get("promotion_ready_experiments_from_absolute_gates")
    if not isinstance(promotion_ready, list):
        raise ValueError("Treasury promotion-ready experiment list is missing")
    prediction_ready = candidate_id in promotion_ready

    decisions = integrity.get("decisions")
    treasury_integrity = decisions.get("treasury") if isinstance(decisions, dict) else None
    data_quality_pass = (
        integrity.get("status") == "PASS"
        and integrity.get("v2_1_reproducible") is True
        and isinstance(treasury_integrity, dict)
        and treasury_integrity.get("sample_hashes_match") is True
    )

    sizing_activation = sizing.get("current_candidate_activation")
    if prediction_ready and sizing_activation == "BLOCKED":
        incomplete_reason = (
            "Prediction is marked ready but sizing remains blocked; champion portfolio evidence must be generated only after resolving that contract."
        )
    elif not prediction_ready:
        incomplete_reason = (
            "The strongest retained candidate fails the existing absolute prediction-readiness prerequisite. V3-017 therefore remains at 1.00x and champion portfolio/cost/parameter evidence is intentionally not generated."
        )
    else:
        incomplete_reason = "Champion portfolio evidence has not yet been supplied."

    evidence = {
        "candidate_id": candidate_id,
        "as_of": treasury["ablation_as_of"],
        "evidence_complete": False,
        "incomplete_reason": incomplete_reason,
        "lineage": {
            "feature_version": treasury["candidate_feature_version"],
            "model_version": candidate_id,
            "label_version": "v3-labels-001",
            "training_version": "v3-evaluator-001",
            "policy_version": policy.get("policy_version"),
            "sizing_version": sizing.get("sizing_version"),
            "evidence_hashes": {
                "treasury_ablation": sha256(TREASURY_REPORT),
                "integrity_rebuild": sha256(INTEGRITY_REPORT),
                "decision_policy_manifest": sha256(POLICY_MANIFEST),
                "sizing_policy_manifest": sha256(SIZING_MANIFEST),
            },
        },
        "prediction": {
            "absolute_prediction_gate_pass": prediction_ready,
            "source_gate_version": "v3-tournament-001",
            "promotion_ready_experiments": promotion_ready,
        },
        "calibration": {},
        "portfolio": {},
        "cross_year": {},
        "risk": {},
        "cost_robustness": {},
        "parameter_robustness": {},
        "data_quality": {
            "point_in_time_pass": data_quality_pass,
            "leakage_gate_pass": data_quality_pass,
            "sample_hashes_match": data_quality_pass,
            "frozen_v2_1_reproducible": integrity.get("v2_1_reproducible") is True,
        },
        "sizing_activation": sizing_activation,
        "current_multiplier": 1.00,
    }
    return evidence


def main() -> int:
    args = parse_args()
    evidence = build_evidence()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
