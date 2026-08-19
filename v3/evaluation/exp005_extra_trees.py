#!/usr/bin/env python3
"""Evaluate pre-registered EXP-005 on the frozen retained Treasury dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v3.evaluation.tournament import summarize_experiments
from v3.evaluation.treasury_ablation import (
    ABLATION_AS_OF,
    FROZEN_TREASURY_DATASET,
    TREASURY_REGISTRY,
    prepare_frozen_datasets,
)
from v3.evaluation.walk_forward import (
    evaluate_prediction_frame,
    run_common_evaluation,
    validate_common_samples,
)
from v3.models.common import (
    load_feature_registry,
    load_model_dataset,
    validate_feature_columns,
)
from v3.models.extra_trees_calibrated import (
    EXPERIMENT_ID,
    MODEL_NAME,
    predict_fold,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.json"
EXISTING_PREDICTIONS = ROOT / "v3" / "reports" / "exp005_existing_predictions.parquet"
EXISTING_METRICS = ROOT / "v3" / "reports" / "exp005_existing_metrics.csv"
EXISTING_SUMMARY = ROOT / "v3" / "reports" / "exp005_existing_summary.json"
EXP005_PREDICTIONS = ROOT / "v3" / "reports" / "exp005_predictions.parquet"
EXP005_METRICS = ROOT / "v3" / "reports" / "exp005_metrics.csv"
TOURNAMENT_OUTPUT = ROOT / "v3" / "reports" / "exp005_tournament.csv"
REPORT_OUTPUT = ROOT / "v3" / "reports" / "exp005_evaluation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT)
    return parser.parse_args()


def generate_exp005_predictions(
    frame: pd.DataFrame,
    features: list[str],
    folds: list[dict[str, Any]],
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for fold in folds:
        predicted, _ = predict_fold(frame, features, fold)
        predicted = predicted.copy()
        predicted["experiment_id"] = EXPERIMENT_ID
        predicted["model_name"] = MODEL_NAME
        predicted["model_family"] = "extra_trees_calibrated"
        outputs.append(predicted)
    if not outputs:
        raise ValueError("EXP-005 produced no fold predictions")
    combined = pd.concat(outputs, ignore_index=True, sort=False)
    if combined["decision_date"].duplicated().any():
        raise ValueError("EXP-005 validation folds overlap on decision_date")
    return combined.sort_values("decision_date").reset_index(drop=True)


def build_tournament(
    existing_metrics: pd.DataFrame,
    exp005_metrics: pd.DataFrame,
) -> pd.DataFrame:
    existing = existing_metrics.copy()
    existing["experiment_id"] = "UST-" + existing["experiment_id"].astype(str)
    existing["model_name"] = existing["model_name"].astype(str) + " [+Treasury]"
    return summarize_experiments(
        pd.concat([existing, exp005_metrics], ignore_index=True, sort=False)
    )


def run_exp005(report_output: Path = REPORT_OUTPUT) -> dict[str, Any]:
    prepared = prepare_frozen_datasets()
    if prepared["as_of"] != ABLATION_AS_OF.date().isoformat():
        raise ValueError("EXP-005 frozen cutoff differs from the registered cutoff")

    existing_report = run_common_evaluation(
        dataset_path=FROZEN_TREASURY_DATASET,
        registry_path=TREASURY_REGISTRY,
        predictions_output=EXISTING_PREDICTIONS,
        metrics_output=EXISTING_METRICS,
        summary_output=EXISTING_SUMMARY,
    )
    frame = load_model_dataset(FROZEN_TREASURY_DATASET)
    feature_version, features = load_feature_registry(TREASURY_REGISTRY)
    validate_feature_columns(frame, features)
    if feature_version != "v3-features-004-treasury":
        raise ValueError(f"EXP-005 requires retained Treasury features, got {feature_version}")

    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    folds = config.get("validation", {}).get("folds", [])
    if not folds:
        raise ValueError("config.json contains no validation folds")

    predictions = generate_exp005_predictions(frame, features, folds)
    exp005_metrics = evaluate_prediction_frame(frame, predictions)
    existing_metrics = pd.read_csv(EXISTING_METRICS)
    validate_common_samples(
        pd.concat([existing_metrics, exp005_metrics], ignore_index=True, sort=False)
    )

    EXP005_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(EXP005_PREDICTIONS, index=False, engine="pyarrow")
    exp005_metrics.to_csv(EXP005_METRICS, index=False)

    tournament = build_tournament(existing_metrics, exp005_metrics)
    tournament.to_csv(TOURNAMENT_OUTPUT, index=False)
    row = tournament.loc[tournament["experiment_id"] == EXPERIMENT_ID]
    if len(row) != 1:
        raise ValueError("EXP-005 tournament row is missing or duplicated")
    candidate = row.iloc[0]
    absolute_prediction_ready = bool(candidate["promotion_ready"])

    report: dict[str, Any] = {
        "status": "EXP_005_EVALUATION_COMPLETE",
        "experiment_id": EXPERIMENT_ID,
        "model_name": MODEL_NAME,
        "as_of": prepared["as_of"],
        "feature_version": feature_version,
        "feature_count": len(features),
        "frozen_dataset_sha256": prepared["candidate_dataset_sha256"],
        "existing_evaluator_status": existing_report["status"],
        "sample_hashes_match": True,
        "absolute_prediction_prerequisite_pass": absolute_prediction_ready,
        "promotion_gate_reason": str(candidate["promotion_gate_reason"]),
        "classification_mean_brier": float(candidate["classification_mean_brier"]),
        "classification_mean_ece": float(candidate["classification_mean_ece"]),
        "classification_mean_relative_brier_improvement": float(
            candidate["classification_mean_relative_brier_improvement"]
        ),
        "classification_positive_relative_brier_folds": int(
            candidate["classification_positive_relative_brier_folds"]
        ),
        "return_mean_spearman": float(candidate["return_mean_spearman"]),
        "return_positive_spearman_folds": int(
            candidate["return_positive_spearman_folds"]
        ),
        "drawdown_mean_spearman": float(candidate["drawdown_mean_spearman"]),
        "drawdown_positive_spearman_folds": int(
            candidate["drawdown_positive_spearman_folds"]
        ),
        "overall_full_candidate_rank": float(candidate["overall_full_candidate_rank"]),
        "champion_selected": False,
        "v3_018_gate_version": "v3-champion-gates-001",
        "v3_018_evaluation_status": (
            "ELIGIBLE_FOR_FULL_V3_018_ASSESSMENT"
            if absolute_prediction_ready
            else "NOT_ELIGIBLE_PREDICTION_PREREQUISITE_FAILED"
        ),
        "current_sizing_multiplier": 1.00,
        "note": (
            "EXP-005 was pre-registered in issue #44. This result may not be followed by parameter tuning under the same experiment/version."
        ),
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    args = parse_args()
    report = run_exp005(args.report_output)
    return 0 if report["status"] == "EXP_005_EVALUATION_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
