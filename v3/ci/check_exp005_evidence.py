#!/usr/bin/env python3
"""Verify immutable EXP-005 evidence, model contract, and semantic reproduction.

Committed evidence bytes are hashed by the manifest. A fresh rerun is allowed
only tiny floating-point serialization drift while IDs, sample hashes, discrete
decisions, signs encoded by promotion reasons, and model constants stay exact.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import subprocess
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

ABS_TOLERANCE = 1e-5
REL_TOLERANCE = 1e-9


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def committed_bytes(path: Path) -> bytes:
    result = subprocess.run(
        ["git", "show", f"HEAD:{_relative(path)}"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ValueError(
            f"Could not read committed EXP-005 artifact {_relative(path)}: "
            + result.stderr.decode("utf-8", errors="replace")
        )
    return result.stdout


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _numeric_equal(left: float, right: float) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=REL_TOLERANCE,
        abs_tol=ABS_TOLERANCE,
    )


def _compare_json(current: Any, committed: Any, path: str = "root") -> None:
    if isinstance(current, bool) or isinstance(committed, bool):
        _require(current is committed, f"Semantic drift at {path}: {committed!r} -> {current!r}")
        return
    if isinstance(current, (int, float)) and isinstance(committed, (int, float)):
        _require(
            _numeric_equal(float(current), float(committed)),
            f"Numeric drift beyond tolerance at {path}: {committed!r} -> {current!r}",
        )
        return
    if isinstance(current, dict) and isinstance(committed, dict):
        _require(set(current) == set(committed), f"JSON keys changed at {path}")
        for key in sorted(current):
            _compare_json(current[key], committed[key], f"{path}.{key}")
        return
    if isinstance(current, list) and isinstance(committed, list):
        _require(len(current) == len(committed), f"List length changed at {path}")
        for index, (left, right) in enumerate(zip(current, committed)):
            _compare_json(left, right, f"{path}[{index}]")
        return
    _require(current == committed, f"Semantic drift at {path}: {committed!r} -> {current!r}")


def _csv_rows(payload: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(payload)))


def _try_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _compare_csv(path: Path, key_columns: tuple[str, ...]) -> None:
    committed = _csv_rows(committed_bytes(path).decode("utf-8"))
    current = _csv_rows(path.read_text(encoding="utf-8"))
    _require(len(current) == len(committed), f"Row count changed in {_relative(path)}")

    def key(row: dict[str, str]) -> tuple[str, ...]:
        return tuple(row.get(column, "") for column in key_columns)

    committed_map = {key(row): row for row in committed}
    current_map = {key(row): row for row in current}
    _require(len(committed_map) == len(committed), f"Duplicate committed row keys in {_relative(path)}")
    _require(len(current_map) == len(current), f"Duplicate rerun row keys in {_relative(path)}")
    _require(set(current_map) == set(committed_map), f"Row identities changed in {_relative(path)}")

    for row_key in sorted(current_map):
        left = current_map[row_key]
        right = committed_map[row_key]
        _require(set(left) == set(right), f"Columns changed for {row_key} in {_relative(path)}")
        for column in left:
            current_value = left[column]
            committed_value = right[column]
            if current_value == committed_value:
                continue
            current_number = _try_float(current_value)
            committed_number = _try_float(committed_value)
            if current_number is not None and committed_number is not None:
                _require(
                    _numeric_equal(current_number, committed_number),
                    f"Numeric drift beyond tolerance in {_relative(path)} {row_key} {column}: "
                    f"{committed_value} -> {current_value}",
                )
                continue
            raise ValueError(
                f"Discrete drift in {_relative(path)} {row_key} {column}: "
                f"{committed_value!r} -> {current_value!r}"
            )


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
    current_evaluation = load_json(EVALUATION)
    committed_evaluation = json.loads(committed_bytes(EVALUATION).decode("utf-8"))
    _require(isinstance(committed_evaluation, dict), "Committed EXP-005 evaluation is not an object")

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
    model_manifest = classification.get("model", {})
    for key, expected in TREE_PARAMS.items():
        _require(model_manifest.get(key) == expected, f"Classification tree parameter drift: {key}")
        _require(regression.get(key) == expected, f"Regression tree parameter drift: {key}")
    _require(float(classification.get("base_fit_fraction")) == 1.0 - CALIBRATION_FRACTION, "Base-fit fraction drift")
    _require(float(classification.get("calibration_fraction")) == CALIBRATION_FRACTION, "Calibration fraction drift")
    _require(int(classification.get("minimum_base_fit_rows")) == MINIMUM_TRAINING_ROWS, "Base-fit minimum drift")
    _require(int(classification.get("calibration_minimum_rows")) == MINIMUM_CALIBRATION_ROWS, "Calibration minimum drift")
    calibrator_manifest = classification.get("calibrator", {})
    for key, expected in CALIBRATOR_PARAMS.items():
        _require(calibrator_manifest.get(key) == expected, f"Calibrator parameter drift: {key}")
    _require(classification.get("refit_after_calibration") is False, "EXP-005 must not refit after calibration")

    evidence_hashes = manifest.get("evidence")
    _require(isinstance(evidence_hashes, dict), "Missing EXP-005 evidence hashes")
    _require(
        evidence_hashes.get("evaluation_sha256") == sha256_bytes(committed_bytes(EVALUATION)),
        "Committed EXP-005 evaluation hash mismatch",
    )
    _require(
        evidence_hashes.get("metrics_sha256") == sha256_bytes(committed_bytes(METRICS)),
        "Committed EXP-005 metrics hash mismatch",
    )
    _require(
        evidence_hashes.get("tournament_sha256") == sha256_bytes(committed_bytes(TOURNAMENT)),
        "Committed EXP-005 tournament hash mismatch",
    )

    _compare_json(current_evaluation, committed_evaluation, "evaluation")
    _compare_csv(
        METRICS,
        ("experiment_id", "model_name", "fold", "target_type", "target", "horizon", "sample_sha256"),
    )
    _compare_csv(TOURNAMENT, ("experiment_id",))

    for key in ("experiment_id", "model_name", "feature_version", "frozen_dataset_sha256"):
        manifest_key = "feature_set_version" if key == "feature_version" else key
        _require(current_evaluation.get(key) == manifest.get(manifest_key), f"EXP-005 evaluation/manifest mismatch: {key}")
    _require(current_evaluation.get("status") == "EXP_005_EVALUATION_COMPLETE", "EXP-005 evaluation is not complete")
    _require(current_evaluation.get("absolute_prediction_prerequisite_pass") is False, "EXP-005 prediction prerequisite unexpectedly passed")
    _require(current_evaluation.get("sample_hashes_match") is True, "EXP-005 sample hashes do not match")
    _require(current_evaluation.get("champion_selected") is False, "EXP-005 must not select a champion")
    _require(float(current_evaluation.get("current_sizing_multiplier")) == 1.0, "EXP-005 current sizing changed")
    _require(current_evaluation.get("v3_018_gate_version") == "v3-champion-gates-001", "EXP-005 gate version drift")

    tournament = _exp005_tournament_row()
    _require(tournament.get("promotion_ready") == "False", "Tournament unexpectedly marks EXP-005 promotion-ready")
    _require(tournament.get("status") == "NOT_PROMOTION_READY", "Tournament EXP-005 status drift")
    _require(float(tournament.get("overall_full_candidate_rank", "nan")) == 1.0, "Recorded EXP-005 comparison rank changed")

    print(
        "EXP-005 immutable evidence/model contract and semantic reproduction: PASS "
        f"(abs_tol={ABS_TOLERANCE:g})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
