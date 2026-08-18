#!/usr/bin/env python3
"""Regularized chronological return-regression baseline for v3.

The model predicts executable-entry forward returns at 5, 20, and 60 sessions.
Every fold uses only labels whose outcome-known date has matured by the training
cutoff. No threshold selection, model tournament logic, or sizing lives here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from v3.models.common import (
    DEFAULT_FEATURE_REGISTRY,
    DEFAULT_MODEL_DATASET,
    HORIZONS,
    eligible_training_mask,
    fold_test_mask,
    load_feature_registry,
    load_model_dataset,
    validate_feature_columns,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.json"
DEFAULT_PREDICTIONS = ROOT / "v3" / "reports" / "return_regression_predictions.parquet"
DEFAULT_SUMMARY = ROOT / "v3" / "reports" / "return_regression_summary.json"
MODEL_NAME = "ridge_return_v1"
RIDGE_ALPHA = 1.0
MINIMUM_TRAINING_ROWS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_MODEL_DATASET)
    parser.add_argument("--registry", type=Path, default=DEFAULT_FEATURE_REGISTRY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--minimum-training-rows", type=int, default=MINIMUM_TRAINING_ROWS)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


def fit_predict_return(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> np.ndarray:
    if len(x_train) != len(y_train):
        raise ValueError("x_train and y_train length mismatch")
    if len(x_train) == 0:
        raise ValueError("No training rows")

    target = pd.to_numeric(y_train, errors="raise").astype(float)
    if not np.isfinite(target.to_numpy()).all():
        raise ValueError("Training target contains non-finite values")

    pipeline = make_pipeline()
    pipeline.fit(x_train, target)
    prediction = np.asarray(pipeline.predict(x_test), dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("Regression produced non-finite predictions")
    return prediction


def predict_fold(
    frame: pd.DataFrame,
    features: list[str],
    fold: dict[str, Any],
    minimum_training_rows: int = MINIMUM_TRAINING_ROWS,
) -> tuple[pd.DataFrame, dict[int, int]]:
    features = validate_feature_columns(frame, features)
    test = frame.loc[fold_test_mask(frame, fold)].copy()
    if test.empty:
        raise ValueError(f"Fold {fold['name']} contains no test rows")

    output = pd.DataFrame(
        {
            "decision_date": test["decision_date"].to_numpy(),
            "fold": str(fold["name"]),
            "model_name": MODEL_NAME,
            "training_cutoff": pd.Timestamp(fold["train_end"]).normalize(),
        },
        index=test.index,
    )
    training_rows: dict[int, int] = {}

    for horizon in HORIZONS:
        target_column = f"forward_return_{horizon}d"
        train_mask = eligible_training_mask(
            frame,
            horizon,
            fold["train_end"],
            target_column=target_column,
        )
        train = frame.loc[train_mask]
        training_rows[horizon] = int(len(train))
        if len(train) < minimum_training_rows:
            raise ValueError(
                f"Fold {fold['name']} horizon {horizon}d has only {len(train)} "
                f"eligible training rows; need {minimum_training_rows}"
            )

        output[f"predicted_return_{horizon}d"] = fit_predict_return(
            train[features],
            train[target_column],
            test[features],
        )

    return output.reset_index(drop=True), training_rows


def summarize_fold(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    fold: dict[str, Any],
    training_rows: dict[int, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        prediction_column = f"predicted_return_{horizon}d"
        target_column = f"forward_return_{horizon}d"
        joined = predictions[["decision_date", prediction_column]].merge(
            frame[["decision_date", target_column]],
            on="decision_date",
            how="left",
            validate="one_to_one",
        )
        realized = joined.loc[joined[target_column].notna()].copy()
        prediction = pd.to_numeric(joined[prediction_column], errors="raise").to_numpy(float)
        summary: dict[str, Any] = {
            "fold": str(fold["name"]),
            "horizon": horizon,
            "training_cutoff": str(fold["train_end"]),
            "training_rows": int(training_rows[horizon]),
            "prediction_rows": int(len(joined)),
            "realized_rows": int(len(realized)),
            "mean_predicted_return": float(np.mean(prediction)),
            "minimum_predicted_return": float(np.min(prediction)),
            "maximum_predicted_return": float(np.max(prediction)),
        }
        if len(realized):
            actual = pd.to_numeric(realized[target_column], errors="raise").to_numpy(float)
            predicted = pd.to_numeric(realized[prediction_column], errors="raise").to_numpy(float)
            error = predicted - actual
            summary.update(
                {
                    "mean_actual_return": float(np.mean(actual)),
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                }
            )
        else:
            summary.update({"mean_actual_return": None, "mae": None, "rmse": None})
        rows.append(summary)
    return rows


def run_baseline(
    dataset_path: Path = DEFAULT_MODEL_DATASET,
    registry_path: Path = DEFAULT_FEATURE_REGISTRY,
    config_path: Path = DEFAULT_CONFIG,
    predictions_output: Path = DEFAULT_PREDICTIONS,
    summary_output: Path = DEFAULT_SUMMARY,
    minimum_training_rows: int = MINIMUM_TRAINING_ROWS,
) -> dict[str, Any]:
    frame = load_model_dataset(dataset_path)
    feature_version, features = load_feature_registry(registry_path)
    validate_feature_columns(frame, features)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    folds = config.get("validation", {}).get("folds", [])
    if not folds:
        raise ValueError("config.json has no validation folds")

    prediction_frames: list[pd.DataFrame] = []
    evidence: list[dict[str, Any]] = []
    for fold in folds:
        predicted, training_rows = predict_fold(
            frame,
            features,
            fold,
            minimum_training_rows=minimum_training_rows,
        )
        prediction_frames.append(predicted)
        evidence.extend(summarize_fold(frame, predicted, fold, training_rows))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.sort_values(["decision_date", "fold"]).reset_index(drop=True)
    if predictions["decision_date"].duplicated().any():
        raise ValueError("Validation folds overlap on decision_date")

    for horizon in HORIZONS:
        column = f"predicted_return_{horizon}d"
        values = pd.to_numeric(predictions[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite predictions in {column}")

    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_output, index=False, engine="pyarrow")

    report = {
        "model_name": MODEL_NAME,
        "model_type": "regularized_return_regression",
        "feature_set_version": feature_version,
        "feature_count": len(features),
        "dataset_sha256": _sha256(dataset_path),
        "prediction_rows": int(len(predictions)),
        "hyperparameters": {
            "model": "Ridge",
            "alpha": RIDGE_ALPHA,
            "imputer": "training_median",
            "scaler": "training_standard_scaler",
        },
        "fold_horizon_evidence": evidence,
        "status": "BASELINE_GENERATED",
        "note": "Provisional V3-006 evidence only; model selection belongs to later tournament tasks.",
    }
    summary_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {predictions_output} ({len(predictions)} chronological predictions)")
    print(f"Wrote {summary_output}")
    return report


def main() -> int:
    args = parse_args()
    report = run_baseline(
        dataset_path=args.dataset,
        registry_path=args.registry,
        config_path=args.config,
        predictions_output=args.predictions_output,
        summary_output=args.summary_output,
        minimum_training_rows=args.minimum_training_rows,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
