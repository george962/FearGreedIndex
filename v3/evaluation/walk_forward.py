#!/usr/bin/env python3
"""Common chronological evaluator for the initial v3 model experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from v3.evaluation.metrics import classification_metrics, regression_metrics
from v3.models.common import (
    DEFAULT_FEATURE_REGISTRY,
    DEFAULT_MODEL_DATASET,
    HORIZONS,
    load_feature_registry,
    load_model_dataset,
    validate_feature_columns,
)
from v3.models.gradient_boosting import predict_fold as predict_gradient_boosting
from v3.models.logistic_baseline import predict_fold as predict_logistic
from v3.models.random_forest import predict_fold as predict_random_forest
from v3.models.return_regression import predict_fold as predict_ridge

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.json"
DEFAULT_PREDICTIONS = ROOT / "v3" / "reports" / "common_walk_forward_predictions.parquet"
DEFAULT_METRICS = ROOT / "v3" / "reports" / "common_walk_forward_metrics.csv"
DEFAULT_SUMMARY = ROOT / "v3" / "reports" / "common_walk_forward_summary.json"

ModelPredictor = Callable[
    [pd.DataFrame, list[str], dict[str, Any], int],
    tuple[pd.DataFrame, Any],
]

MODEL_REGISTRY: tuple[dict[str, Any], ...] = (
    {
        "experiment_id": "EXP-001",
        "model_name": "logistic_l2_v1",
        "model_family": "logistic_classification",
        "predictor": predict_logistic,
    },
    {
        "experiment_id": "EXP-002",
        "model_name": "ridge_return_v1",
        "model_family": "ridge_return_regression",
        "predictor": predict_ridge,
    },
    {
        "experiment_id": "EXP-003",
        "model_name": "hist_gradient_boosting_v1",
        "model_family": "histogram_gradient_boosting",
        "predictor": predict_gradient_boosting,
    },
    {
        "experiment_id": "EXP-004",
        "model_name": "random_forest_v1",
        "model_family": "random_forest",
        "predictor": predict_random_forest,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_MODEL_DATASET)
    parser.add_argument("--registry", type=Path, default=DEFAULT_FEATURE_REGISTRY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--minimum-training-rows", type=int, default=100)
    return parser.parse_args()


def _date_sample_hash(dates: pd.Series) -> str:
    normalized = pd.to_datetime(dates, errors="raise").dt.strftime("%Y-%m-%d")
    payload = "\n".join(normalized.tolist()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prediction_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("predicted_")]


def generate_predictions(
    frame: pd.DataFrame,
    features: list[str],
    folds: list[dict[str, Any]],
    minimum_training_rows: int = 100,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for spec in MODEL_REGISTRY:
        predictor: ModelPredictor = spec["predictor"]
        for fold in folds:
            predicted, _ = predictor(
                frame,
                features,
                fold,
                minimum_training_rows=minimum_training_rows,
            )
            predicted = predicted.copy()
            predicted["experiment_id"] = spec["experiment_id"]
            predicted["model_name"] = spec["model_name"]
            predicted["model_family"] = spec["model_family"]
            outputs.append(predicted)
    if not outputs:
        raise ValueError("No model predictions were generated")
    combined = pd.concat(outputs, ignore_index=True, sort=False)
    return combined.sort_values(["experiment_id", "decision_date"]).reset_index(drop=True)


def evaluate_prediction_frame(
    model_dataset: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        raise ValueError("Prediction frame is empty")

    for (experiment_id, model_name, fold), group in predictions.groupby(
        ["experiment_id", "model_name", "fold"], sort=True
    ):
        for horizon in HORIZONS:
            p_col = f"predicted_p_up_{horizon}d"
            if p_col in group and group[p_col].notna().any():
                target = f"forward_positive_{horizon}d"
                joined = group[["decision_date", p_col]].dropna(subset=[p_col]).merge(
                    model_dataset[["decision_date", target]],
                    on="decision_date",
                    how="left",
                    validate="one_to_one",
                )
                realized = joined.loc[joined[target].notna()].sort_values("decision_date")
                metrics = classification_metrics(
                    realized[target].astype(int),
                    realized[p_col].astype(float),
                )
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "model_name": model_name,
                        "fold": fold,
                        "target_type": "classification",
                        "target": f"p_up_{horizon}d",
                        "horizon": horizon,
                        "sample_sha256": _date_sample_hash(realized["decision_date"]),
                        **metrics,
                    }
                )

            r_col = f"predicted_return_{horizon}d"
            if r_col in group and group[r_col].notna().any():
                target = f"forward_return_{horizon}d"
                joined = group[["decision_date", r_col]].dropna(subset=[r_col]).merge(
                    model_dataset[["decision_date", target]],
                    on="decision_date",
                    how="left",
                    validate="one_to_one",
                )
                realized = joined.loc[joined[target].notna()].sort_values("decision_date")
                metrics = regression_metrics(realized[target], realized[r_col])
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "model_name": model_name,
                        "fold": fold,
                        "target_type": "return_regression",
                        "target": f"return_{horizon}d",
                        "horizon": horizon,
                        "sample_sha256": _date_sample_hash(realized["decision_date"]),
                        **metrics,
                    }
                )

        dd_col = "predicted_drawdown_20d"
        if dd_col in group and group[dd_col].notna().any():
            target = "max_drawdown_20d"
            joined = group[["decision_date", dd_col]].dropna(subset=[dd_col]).merge(
                model_dataset[["decision_date", target]],
                on="decision_date",
                how="left",
                validate="one_to_one",
            )
            realized = joined.loc[joined[target].notna()].sort_values("decision_date")
            metrics = regression_metrics(realized[target], realized[dd_col])
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "model_name": model_name,
                    "fold": fold,
                    "target_type": "drawdown_regression",
                    "target": "drawdown_20d",
                    "horizon": 20,
                    "sample_sha256": _date_sample_hash(realized["decision_date"]),
                    **metrics,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No evaluation metrics were produced")
    validate_common_samples(result)
    return result.sort_values(
        ["target_type", "horizon", "fold", "experiment_id"]
    ).reset_index(drop=True)


def validate_common_samples(metrics: pd.DataFrame) -> None:
    """Require identical realized dates across comparable model lanes."""

    required = {"fold", "target_type", "target", "horizon", "sample_sha256"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Metrics missing sample-contract columns: {sorted(missing)}")

    mismatches: list[str] = []
    grouped = metrics.groupby(["fold", "target_type", "target", "horizon"], sort=True)
    for key, group in grouped:
        if len(group) > 1 and group["sample_sha256"].nunique() != 1:
            mismatches.append(str(key))
    if mismatches:
        raise ValueError(
            "Comparable models were not evaluated on identical realized dates: "
            + ", ".join(mismatches)
        )


def aggregate_summary(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (experiment_id, model_name, target_type), group in metrics.groupby(
        ["experiment_id", "model_name", "target_type"], sort=True
    ):
        row: dict[str, Any] = {
            "experiment_id": experiment_id,
            "model_name": model_name,
            "target_type": target_type,
            "fold_target_cells": int(len(group)),
        }
        if target_type == "classification":
            for column in (
                "brier_score",
                "log_loss",
                "expected_calibration_error",
                "maximum_calibration_error",
                "relative_brier_improvement",
            ):
                row[f"mean_{column}"] = float(pd.to_numeric(group[column]).mean())
        else:
            for column in ("mae", "rmse", "spearman_rank_correlation"):
                row[f"mean_{column}"] = float(pd.to_numeric(group[column]).mean())
        rows.append(row)
    return rows


def run_common_evaluation(
    dataset_path: Path = DEFAULT_MODEL_DATASET,
    registry_path: Path = DEFAULT_FEATURE_REGISTRY,
    config_path: Path = DEFAULT_CONFIG,
    predictions_output: Path = DEFAULT_PREDICTIONS,
    metrics_output: Path = DEFAULT_METRICS,
    summary_output: Path = DEFAULT_SUMMARY,
    minimum_training_rows: int = 100,
) -> dict[str, Any]:
    frame = load_model_dataset(dataset_path)
    feature_version, features = load_feature_registry(registry_path)
    validate_feature_columns(frame, features)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    folds = config.get("validation", {}).get("folds", [])
    if not folds:
        raise ValueError("config.json has no validation folds")

    predictions = generate_predictions(
        frame,
        features,
        folds,
        minimum_training_rows=minimum_training_rows,
    )
    metrics = evaluate_prediction_frame(frame, predictions)

    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_output, index=False, engine="pyarrow")
    metrics.to_csv(metrics_output, index=False)

    report = {
        "status": "COMMON_EVALUATION_COMPLETE",
        "feature_set_version": feature_version,
        "experiments": [spec["experiment_id"] for spec in MODEL_REGISTRY],
        "prediction_rows": int(len(predictions)),
        "metric_rows": int(len(metrics)),
        "aggregate_metrics": aggregate_summary(metrics),
        "sample_contract": "Comparable models in each fold/target lane use identical realized decision dates, verified by SHA-256.",
        "trading_evaluation_status": "DEFERRED_UNTIL_V3_016_DECISION_POLICY",
        "backtest_utility_status": "AVAILABLE_FOR_POLICY_EVALUATION",
        "note": "V3-009 evaluates prediction quality only; it does not select a champion.",
    }
    summary_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {predictions_output} ({len(predictions)} model-date rows)")
    print(f"Wrote {metrics_output} ({len(metrics)} fold-target rows)")
    print(f"Wrote {summary_output}")
    return report


def main() -> int:
    args = parse_args()
    report = run_common_evaluation(
        dataset_path=args.dataset,
        registry_path=args.registry,
        config_path=args.config,
        predictions_output=args.predictions_output,
        metrics_output=args.metrics_output,
        summary_output=args.summary_output,
        minimum_training_rows=args.minimum_training_rows,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
