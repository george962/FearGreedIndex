#!/usr/bin/env python3
"""Verify frozen EXP-006 evidence and immutable experiment contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "v3" / "experiments" / "EXP-006" / "manifest.json"
EVALUATION = ROOT / "v3" / "reports" / "exp006_evaluation.json"
METRICS = ROOT / "v3" / "reports" / "exp006_metrics.csv"
STATES = ROOT / "v3" / "reports" / "exp006_state_distribution.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    for path in (MANIFEST, EVALUATION, METRICS, STATES):
        require(path.exists(), f"Missing EXP-006 artifact: {path}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    metrics = pd.read_csv(METRICS)
    states = pd.read_csv(STATES)

    require(manifest.get("experiment_id") == "EXP-006", "EXP-006 id drift")
    require(manifest.get("status") == "completed_negative_experiment", "EXP-006 status drift")
    require(manifest.get("feature_set_version") == "v3-features-004-treasury", "EXP-006 feature version drift")
    require(manifest.get("feature_count") == 53, "EXP-006 feature count drift")
    require(manifest.get("tuning_after_result_allowed_under_same_version") is False, "EXP-006 post-result tuning must remain disabled")

    target = manifest.get("target", {})
    require(target.get("return_minimum") == 0.02, "EXP-006 target return threshold drift")
    require(target.get("drawdown_strictly_greater_than") == -0.05, "EXP-006 drawdown threshold drift")
    require(target.get("ordinal_bad_return_maximum") == -0.02, "EXP-006 BAD return threshold drift")
    require(target.get("ordinal_excellent_return_minimum") == 0.05, "EXP-006 EXCELLENT threshold drift")

    gate = manifest.get("viability_gate", {})
    require(gate.get("minimum_positive_relative_brier_folds") == 2, "EXP-006 Brier-fold gate drift")
    require(gate.get("minimum_mean_roc_auc_exclusive") == 0.52, "EXP-006 mean-AUC gate drift")
    require(gate.get("minimum_positive_auc_folds") == 2, "EXP-006 AUC-fold gate drift")
    require(gate.get("require_identical_test_sample_hashes") is True, "EXP-006 sample-hash gate drift")

    evidence = manifest.get("evidence", {})
    require(evidence.get("evaluation_sha256") == sha256(EVALUATION), "EXP-006 evaluation hash mismatch")
    require(evidence.get("metrics_sha256") == sha256(METRICS), "EXP-006 metrics hash mismatch")
    require(evidence.get("state_distribution_sha256") == sha256(STATES), "EXP-006 state-distribution hash mismatch")

    require(evaluation.get("experiment_viability_pass") is False, "EXP-006 unexpectedly passed")
    require(evaluation.get("decision") == "DO_NOT_ADVANCE_OPPORTUNITY_TARGET_UNDER_EXP_006", "EXP-006 decision drift")
    require(evaluation.get("sample_hashes_match") is True, "EXP-006 sample hashes do not match")
    require(evaluation.get("champion_selected") is False, "EXP-006 must not select a champion")
    require(evaluation.get("v3_019_eligible") is False, "EXP-006 must not unlock V3-019")
    require(float(evaluation.get("current_sizing_multiplier")) == 1.0, "EXP-006 sizing changed")
    require(evaluation.get("champion_gate_version") == "v3-champion-gates-001", "EXP-006 champion gate drift")

    require(set(metrics["model_name"]) == {"opportunity_logistic_l2_v1", "opportunity_random_forest_v1"}, "EXP-006 model set drift")
    require(set(metrics["fold"].astype(str)) == {"2024", "2025", "2026_ytd"}, "EXP-006 fold set drift")
    require(metrics.groupby("fold")["sample_sha256"].nunique().eq(1).all(), "EXP-006 model sample hashes differ")
    require((metrics["relative_brier_improvement"] <= 0.0).all(), "EXP-006 frozen Brier conclusion changed")

    rf = metrics.loc[metrics["model_name"].eq("opportunity_random_forest_v1")].set_index("fold")
    require(float(rf.loc["2024", "roc_auc"]) > 0.5, "EXP-006 2024 RF AUC relationship changed")
    require(float(rf.loc["2025", "roc_auc"]) < 0.5, "EXP-006 2025 RF reversal changed")
    require(float(rf.loc["2026_ytd", "roc_auc"]) < 0.5, "EXP-006 2026 RF reversal changed")

    require(set(states["state"]) == {"BAD", "NORMAL", "GOOD", "EXCELLENT"}, "EXP-006 state set drift")
    print("EXP-006 immutable evidence and experiment contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
