#!/usr/bin/env python3
"""Verify immutable EXP-005 evidence, manifest, and model contract."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from v3.models.extra_trees_calibrated import (
    CALIBRATION_FRACTION,
    CALIBRATOR_PARAMS,
    EXPERIMENT_ID,
    MINIMUM_CALIBRATION_ROWS,
    MINIMUM_TRAINING_ROWS,
    MODEL_NAME,
    TREE_PARAMS,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "v3" / "experiments" / "EXP-005" / "manifest.json"
EVALUATION = ROOT / "v3" / "reports" / "exp005_evaluation.json"
METRICS = ROOT / "v3" / "reports" / "exp005_metrics.csv"
TOURNAMENT = ROOT / "v3" / "reports" / "exp005_tournament.csv"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exp005_tournament_row() -> dict[str, str]:
    with TOURNAMENT.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row.get("experiment_id") == EXPERIMENT_ID]
    _require(len(matches) == 1, "Tournament must contain exactly one EXP-005 row")
    return matches[0]


def main() -> int:
    for path in (MANIFEST, EVALUATION, METRICS, TOURNAMENT):
        _require(path.exists(), f"Missing immutable EXP-005 artifact: {path}")

    manifest = load_json(MANIFEST)
    evaluation = load_json(EVALUATION)

    _require(manifest.get("immutable") is True, "EXP-005 manifest must be immutable")
    _require(manifest.get("experiment_id") == EXPERIMENT_ID, "EXP-005 id mismatch")
    _require(manifest.get("model_name") == MODEL_NAME, "EXP-005 model name mismatch")
    _require(manifest.get("feature_set_version") == "v3-features-004-treasury", "Unexpected EXP-005 feature version")
    _require(manifest.get("label_version") == "v3-labels-001", "Unexpected EXP-005 label version")
    _require(manifest.get("gate_version") == "v3-champion-gates-001", "Unexpected champion gate version")
    _require(manifest.get("prediction_prerequisite_pass") is False, "EXP-005 must remain prediction-gate rejected")
    _require(manifest.get("promotion_status") == "NOT_PROMOTION_READY", "EXP-005 promotion status changed")
    _require(float(manifest.get("sizing_multiplier")) == 1.0, "EXP-005 sizing must remain 1.00x")
    _require(manifest.get("tuning_after_result_allowed_under_same_version") is False, "EXP-005 post-result tuning must remain disabled")

    classification = manifest.get("classification")
    regression = manifest.get("regression")
    _require(isinstance(classification, dict), "Missing EXP-005 classification manifest")
    _require(isinstance(regression, dict), "Missing EXP-005 regression manifest")
    _require(classification.get("model", {}).get("n_estimators") == TREE_PARAMS["n_estimators"], "Tree count drift")
    _require(classification.get("model", {}).get("max_depth") == TREE_PARAMS["max_depth"], "Tree depth drift")
    _require(classification.get("model", {}).get("min_samples_leaf") == TREE_PARAMS["min_samples_leaf"], "Tree leaf-size drift")
    _require(classification.get("model", {}).get("max_features") == TREE_PARAMS["max_features"], "Tree feature-sampling drift")
    _require(classification.get("model", {}).get("bootstrap") == TREE_PARAMS["bootstrap"], "Tree bootstrap drift")
    _require(classification.get("model", {}).get("random_state") == TREE_PARAMS["random_state"], "Tree seed drift")
    _require(classification.get("model", {}).get("n_jobs") == TREE_PARAMS["n_jobs"], "Tree thread-count drift")
    _require(regression.get("n_estimators") == TREE_PARAMS["n_estimators"], "Regression tree count drift")
    _require(regression.get("max_depth") == TREE_PARAMS["max_depth"], "Regression tree depth drift")
    _require(regression.get("min_samples_leaf") == TREE_PARAMS["min_samples_leaf"], "Regression leaf-size drift")
    _require(regression.get("max_features") == TREE_PARAMS["max_features"], "Regression feature-sampling drift")
    _require(regression.get("bootstrap") == TREE_PARAMS["bootstrap"], "Regression bootstrap drift")
    _require(regression.get("random_state") == TREE_PARAMS["random_state"], "Regression seed drift")
    _require(regression.get("n_jobs") == TREE_PARAMS["n_jobs"], "Regression thread-count drift")
    _require(float(classification.get("base_fit_fraction")) == 1.0 - CALIBRATION_FRACTION, "Base-fit fraction drift")
    _require(float(classification.get("calibration_fraction")) == CALIBRATION_FRACTION, "Calibration fraction drift")
    _require(int(classification.get("minimum_base_fit_rows")) == MINIMUM_TRAINING_ROWS, "Base-fit minimum drift")
    _require(int(classification.get("calibration_minimum_rows")) == MINIMUM_CALIBRATION_ROWS, "Calibration minimum drift")
    _require(classification.get("calibrator", {}).get("C") == CALIBRATOR_PARAMS["C"], "Calibrator C drift")
    _require(classification.get("calibrator", {}).get("penalty") == CALIBRATOR_PARAMS["penalty"], "Calibrator penalty drift")
    _require(classification.get("calibrator", {}).get("solver") == CALIBRATOR_PARAMS["solver"], "Calibrator solver drift")
    _require(classification.get("calibrator", {}).get("fit_intercept") == CALIBRATOR_PARAMS["fit_intercept"], "Calibrator intercept drift")
    _require(classification.get("calibrator", {}).get("max_iter") == CALIBRATOR_PARAMS["max_iter"], "Calibrator iteration drift")
    _require(classification.get("refit_after_calibration") is False, "EXP-005 must not refit after calibration")

    evidence_hashes = manifest.get("evidence")
    _require(isinstance(evidence_hashes, dict), "Missing EXP-005 evidence hashes")
    _require(evidence_hashes.get("evaluation_sha256") == sha256(EVALUATION), "EXP-005 evaluation hash mismatch")
    _require(evidence_hashes.get("metrics_sha256") == sha256(METRICS), "EXP-005 metrics hash mismatch")
    _require(evidence_hashes.get("tournament_sha256") == sha256(TOURNAMENT), "EXP-005 tournament hash mismatch")

    for key in ("experiment_id", "model_name", "feature_version", "frozen_dataset_sha256"):
        manifest_key = "feature_set_version" if key == "feature_version" else key
        _require(evaluation.get(key) == manifest.get(manifest_key), f"EXP-005 evaluation/manifest mismatch: {key}")
    _require(evaluation.get("status") == "EXP_005_EVALUATION_COMPLETE", "EXP-005 evaluation is not complete")
    _require(evaluation.get("absolute_prediction_prerequisite_pass") is False, "EXP-005 prediction prerequisite unexpectedly passed")
    _require(evaluation.get("sample_hashes_match") is True, "EXP-005 sample hashes do not match")
    _require(evaluation.get("champion_selected") is False, "EXP-005 must not select a champion")
    _require(float(evaluation.get("current_sizing_multiplier")) == 1.0, "EXP-005 current sizing changed")
    _require(evaluation.get("v3_018_gate_version") == "v3-champion-gates-001", "EXP-005 gate version drift")

    tournament = _exp005_tournament_row()
    _require(tournament.get("promotion_ready") == "False", "Tournament unexpectedly marks EXP-005 promotion-ready")
    _require(tournament.get("status") == "NOT_PROMOTION_READY", "Tournament EXP-005 status drift")
    _require(float(tournament.get("overall_full_candidate_rank", "nan")) == 1.0, "Recorded EXP-005 comparison rank changed")

    print("EXP-005 immutable evidence and model contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
