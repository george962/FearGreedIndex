#!/usr/bin/env python3
"""Verify frozen EXP-007 evidence and immutable rate-regime contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "v3" / "experiments" / "EXP-007" / "manifest.json"
EVALUATION = ROOT / "v3" / "reports" / "exp007_evaluation.json"
METRICS = ROOT / "v3" / "reports" / "exp007_metrics.csv"
DIAGNOSTICS = ROOT / "v3" / "reports" / "exp007_regime_diagnostics.csv"
FLOAT_TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_close(actual: object, expected: object, message: str) -> None:
    require(abs(float(actual) - float(expected)) <= FLOAT_TOLERANCE, message)


def main() -> int:
    for path in (MANIFEST, EVALUATION, METRICS, DIAGNOSTICS):
        require(path.exists(), f"Missing EXP-007 artifact: {path}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    metrics = pd.read_csv(METRICS)
    diagnostics = pd.read_csv(DIAGNOSTICS)

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
    require(evidence.get("evaluation_sha256") == sha256(EVALUATION), "EXP-007 evaluation hash mismatch")
    require(evidence.get("metrics_sha256") == sha256(METRICS), "EXP-007 metrics hash mismatch")
    require(evidence.get("regime_diagnostics_sha256") == sha256(DIAGNOSTICS), "EXP-007 diagnostics hash mismatch")

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

    print("EXP-007 immutable evidence, manifest result, and regime contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
