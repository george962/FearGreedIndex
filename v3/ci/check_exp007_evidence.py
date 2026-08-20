#!/usr/bin/env python3
"""Verify frozen EXP-007 evidence and immutable rate-regime contract."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "v3" / "experiments" / "EXP-007" / "manifest.json"
EVALUATION = ROOT / "v3" / "reports" / "exp007_evaluation.json"
METRICS = ROOT / "v3" / "reports" / "exp007_metrics.csv"
DIAGNOSTICS = ROOT / "v3" / "reports" / "exp007_regime_diagnostics.csv"
FLOAT_TOLERANCE = 1e-10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_close(actual: object, expected: object, message: str) -> None:
    require(abs(float(actual) - float(expected)) <= FLOAT_TOLERANCE, message)


def head_bytes(path: Path) -> bytes:
    relative = path.relative_to(ROOT).as_posix()
    return subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def compare_value(current: Any, frozen: Any, label: str) -> None:
    if isinstance(frozen, bool) or isinstance(frozen, int):
        require(current == frozen, f"{label} drift")
    elif isinstance(frozen, float):
        require(
            np.isclose(float(current), float(frozen), rtol=0.0, atol=FLOAT_TOLERANCE, equal_nan=True),
            f"{label} numeric drift",
        )
    else:
        require(current == frozen, f"{label} drift")


def compare_evaluation(current: dict[str, Any], frozen: dict[str, Any]) -> None:
    # Future rows appended after 2026-08-18 may change only the generated
    # whole-dataset hash. Every frozen fold sample, metric, regime result, and
    # decision remains immutable.
    require(set(current) == set(frozen), "EXP-007 evaluation key-set drift")
    for key, frozen_value in frozen.items():
        if key == "dataset_sha256":
            continue
        current_value = current[key]
        if isinstance(frozen_value, dict):
            require(set(current_value) == set(frozen_value), f"EXP-007 {key} key-set drift")
            for inner_key, inner_value in frozen_value.items():
                compare_value(current_value[inner_key], inner_value, f"EXP-007 {key}.{inner_key}")
        elif isinstance(frozen_value, list):
            require(current_value == frozen_value, f"EXP-007 {key} drift")
        else:
            compare_value(current_value, frozen_value, f"EXP-007 {key}")


def compare_frame(current: pd.DataFrame, frozen: pd.DataFrame, keys: list[str], label: str) -> None:
    require(list(current.columns) == list(frozen.columns), f"{label} column drift")
    current = current.sort_values(keys, kind="stable").reset_index(drop=True)
    frozen = frozen.sort_values(keys, kind="stable").reset_index(drop=True)
    require(len(current) == len(frozen), f"{label} row-count drift")
    for column in frozen.columns:
        if pd.api.types.is_numeric_dtype(frozen[column]):
            left = pd.to_numeric(current[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(frozen[column], errors="coerce").to_numpy(float)
            require(
                np.allclose(left, right, rtol=0.0, atol=FLOAT_TOLERANCE, equal_nan=True),
                f"{label} numeric drift in {column}",
            )
        else:
            left = current[column].fillna("<NA>").astype(str).tolist()
            right = frozen[column].fillna("<NA>").astype(str).tolist()
            require(left == right, f"{label} discrete drift in {column}")


def main() -> int:
    for path in (MANIFEST, EVALUATION, METRICS, DIAGNOSTICS):
        require(path.exists(), f"Missing EXP-007 artifact: {path}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    metrics = pd.read_csv(METRICS)
    diagnostics = pd.read_csv(DIAGNOSTICS)

    frozen_eval_bytes = head_bytes(EVALUATION)
    frozen_metrics_bytes = head_bytes(METRICS)
    frozen_diag_bytes = head_bytes(DIAGNOSTICS)
    frozen_evaluation = json.loads(frozen_eval_bytes.decode("utf-8"))
    frozen_metrics = pd.read_csv(io.BytesIO(frozen_metrics_bytes))
    frozen_diagnostics = pd.read_csv(io.BytesIO(frozen_diag_bytes))

    require(manifest.get("experiment_id") == "EXP-007", "EXP-007 id drift")
    require(manifest.get("status") == "completed_negative_experiment", "EXP-007 status drift")
    require(manifest.get("target_source_experiment") == "EXP-006", "EXP-007 target lineage drift")
    require(manifest.get("feature_set_version") == "v3-features-004-treasury", "EXP-007 feature drift")
    require(manifest.get("tuning_after_result_allowed_under_same_version") is False, "EXP-007 post-result tuning must remain disabled")

    regime = manifest.get("regime", {})
    require(regime.get("source_feature") == "treasury_10y_change_20", "EXP-007 regime feature drift")
    require(regime.get("rising_rule") == "> 0", "EXP-007 rising rule drift")
    require(regime.get("falling_or_flat_rule") == "<= 0", "EXP-007 falling rule drift")
    require(regime.get("minimum_training_rows_per_regime") == 100, "EXP-007 support minimum drift")

    gate = manifest.get("viability_gate", {})
    require(gate.get("minimum_mean_roc_auc_exclusive") == 0.52, "EXP-007 mean AUC gate drift")
    require(gate.get("minimum_fold_roc_auc_inclusive") == 0.45, "EXP-007 minimum-fold AUC gate drift")
    require(gate.get("minimum_positive_auc_folds") == 2, "EXP-007 AUC-fold gate drift")
    require(gate.get("minimum_positive_relative_brier_folds") == 2, "EXP-007 Brier-fold gate drift")

    evidence = manifest.get("evidence", {})
    require(evidence.get("evaluation_sha256") == sha256_bytes(frozen_eval_bytes), "EXP-007 committed evaluation hash mismatch")
    require(evidence.get("metrics_sha256") == sha256_bytes(frozen_metrics_bytes), "EXP-007 committed metrics hash mismatch")
    require(evidence.get("regime_diagnostics_sha256") == sha256_bytes(frozen_diag_bytes), "EXP-007 committed diagnostics hash mismatch")

    compare_evaluation(evaluation, frozen_evaluation)
    compare_frame(metrics, frozen_metrics, ["fold"], "EXP-007 metrics")
    compare_frame(diagnostics, frozen_diagnostics, ["fold", "regime"], "EXP-007 diagnostics")

    require(evaluation.get("experiment_viability_pass") is False, "EXP-007 unexpectedly passed")
    require(evaluation.get("decision") == "DO_NOT_ADVANCE_RATE_REGIME_UNDER_EXP_007", "EXP-007 decision drift")
    require(evaluation.get("training_prevalence_ordering_stable") is False, "EXP-007 ordering unexpectedly stable")
    require(evaluation.get("training_prevalence_ordering_by_fold") == ["FALLING_HIGHER", "FALLING_HIGHER", "RISING_HIGHER"], "EXP-007 ordering evidence drift")
    require(evaluation.get("champion_selected") is False, "EXP-007 must not select champion")
    require(evaluation.get("v3_019_eligible") is False, "EXP-007 must not unlock V3-019")
    require(float(evaluation.get("current_sizing_multiplier")) == 1.0, "EXP-007 sizing changed")

    viability = evaluation.get("viability", {})
    require(viability.get("full_mature_test_regime_coverage") is True, "EXP-007 test coverage changed")
    require(viability.get("regime_training_support_pass") is True, "EXP-007 support result changed")
    require(viability.get("sample_hashes_match_exp006") is True, "EXP-007 sample lineage changed")
    require(float(viability.get("mean_relative_brier_improvement")) < 0.0, "EXP-007 Brier conclusion changed")
    require(float(viability.get("mean_roc_auc")) < 0.52, "EXP-007 AUC conclusion changed")
    require(float(viability.get("minimum_fold_roc_auc")) < 0.45, "EXP-007 reversal gate conclusion changed")

    result = manifest.get("result", {})
    require(result.get("experiment_viability_pass") == evaluation.get("experiment_viability_pass"), "EXP-007 manifest pass result contradicts evaluation")
    require(result.get("decision") == evaluation.get("decision"), "EXP-007 manifest decision contradicts evaluation")
    require(result.get("sample_hashes_match_exp006") == viability.get("sample_hashes_match_exp006"), "EXP-007 manifest sample lineage contradicts evaluation")
    require(result.get("full_mature_test_regime_coverage") == viability.get("full_mature_test_regime_coverage"), "EXP-007 manifest coverage contradicts evaluation")
    require(result.get("regime_training_support_pass") == viability.get("regime_training_support_pass"), "EXP-007 manifest support result contradicts evaluation")
    require(result.get("training_prevalence_ordering_stable") == evaluation.get("training_prevalence_ordering_stable"), "EXP-007 manifest ordering stability contradicts evaluation")
    require(result.get("training_prevalence_ordering_by_fold") == evaluation.get("training_prevalence_ordering_by_fold"), "EXP-007 manifest ordering sequence contradicts evaluation")
    require(result.get("champion_selected") == evaluation.get("champion_selected"), "EXP-007 manifest champion result contradicts evaluation")
    require(result.get("v3_019_eligible") == evaluation.get("v3_019_eligible"), "EXP-007 manifest V3-019 result contradicts evaluation")
    require_close(result.get("sizing_multiplier"), evaluation.get("current_sizing_multiplier"), "EXP-007 manifest sizing contradicts evaluation")
    require_close(result.get("mean_relative_brier_improvement"), viability.get("mean_relative_brier_improvement"), "EXP-007 manifest mean Brier contradicts evaluation")
    require(int(result.get("positive_relative_brier_folds")) == int(viability.get("positive_relative_brier_folds")), "EXP-007 manifest Brier fold count contradicts evaluation")
    require_close(result.get("mean_roc_auc"), viability.get("mean_roc_auc"), "EXP-007 manifest mean AUC contradicts evaluation")
    require(int(result.get("positive_auc_folds")) == int(viability.get("positive_auc_folds")), "EXP-007 manifest AUC fold count contradicts evaluation")
    require_close(result.get("minimum_fold_roc_auc"), viability.get("minimum_fold_roc_auc"), "EXP-007 manifest minimum AUC contradicts evaluation")

    require(set(metrics["fold"].astype(str)) == {"2024", "2025", "2026_ytd"}, "EXP-007 fold set drift")
    require(metrics["sample_sha256"].nunique() == 3, "EXP-007 sample hash count drift")
    require(set(diagnostics["regime"]) == {"RATES_RISING", "RATES_FALLING_OR_FLAT"}, "EXP-007 diagnostic regime set drift")
    require((diagnostics["training_rows"] >= 100).all(), "EXP-007 preserved support below minimum")

    print("EXP-007 immutable evidence + semantic reproduction: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
