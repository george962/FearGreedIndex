#!/usr/bin/env python3
"""Regularized chronological logistic-classification baseline for v3.

This module intentionally contains prediction logic only. It does not choose a
champion, tune thresholds, or size positions. Each fold trains only on labels
whose outcome-known date is at or before that fold's training cutoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
DEFAULT_PREDICTIONS = ROOT / "v3" / "reports" / "logistic_baseline_predictions.parquet"
DEFAULT_SUMMARY = ROOT / "v3" / "reports" / "logistic_baseline_summary.json"
MODEL_NAME = "logistic_l2_v1"
RANDOM_SEED = 42
LOGISTIC_C = 1.0
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
            (
                "model",
                LogisticRegression(
                    C=LOGISTIC_C,
                    solver="liblinear",
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def fit_predict_probability(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> np.ndarray:
    """Fit on training rows only and return P(up) for test rows."""

    if len(x_train) != len(y_train):
        raise ValueError("x_train and y_train length mismatch")
    if len(x_train) == 0:
        raise ValueError("No training rows")

    y = pd.to_numeric(y_train, errors="raise").astype(int)
    unique = sorted(y.unique().tolist())
    if not set(unique).issubset({0, 1}):
        raise ValueError(f"Classification target is not binary: {unique}")

    if len(unique) == 1:
        return np.full(len(x_test), float(unique[0]), dtype=float)

    pipeline = make_pipeline()
    pipeline.fit(x_train, y)
    probabilities = pipeline.predict_proba(x_test)
    classes = pipeline.named_steps["model"].classes_.tolist()
    positive_index = classes.index(1)
    result = probabilities[:, positive_index].astype(float)
    return np.clip(result, 0.0, 1.0)


def predict_fold(
    frame: pd.DataFrame,
    features: list[str],
    fold: dict[str, Any],
    minimum_training_rows: int = MINIMUM_TRAINING_ROWS,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Fit one frozen model per horizon at the fold cutoff and predict the fold."""

    features = validate_feature_columns(frame, features)
    test_mask = fold_test_mask(frame, fold)
    test = frame.loc[test_mask].copy()
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
        train_mask = eligible_training_mask(frame, horizon, fold["train_end"])
        train = frame.loc[train_mask]
        training_rows[horizon] = int(len(train))
        if len(train) < minimum_training_rows:
            raise ValueError(
                f"Fold {fold['name']} horizon {horizon}d has only {len(train)} "
                f"eligible training rows; need {minimum_training_rows}"
            )

        probabilities = fit_predict_probability(
            train[features],
            train[f"forward_positive_{horizon}d"],
            test[features],
        )
        output[f"predicted_p_up_{horizon}d"] = probabilities

    return output.reset_index(drop=True), training_rows


def _binary_log_loss(y_true: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(probability.astype(float), 1e-9, 1.0 - 1e-9)
    y = y_true.astype(float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def summarize_fold(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    fold: dict[str, Any],
    training_rows: dict[int, int],
) -> list[dict[str, Any]]:
    source = frame
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        probability_column = f"predicted_p_up_{horizon}d"
        target_column = f"forward_positive_{horizon}d"
        known_column = f"_forward_{horizon}d_known_date"
        joined = predictions[["decision_date", probability_column]].merge(
            source[["decision_date", target_column, known_column]],
            on="decision_date",
            how="left",
            validate="one_to_one",
        )
        realized = joined.loc[joined[target_column].notna()].copy()
        probability = pd.to_numeric(joined[probability_column], errors="raise").to_numpy(float)

        summary: dict[str, Any] = {
            "fold": str(fold["name"]),
            "horizon": horizon,
            "training_cutoff": str(fold["train_end"]),
            "training_rows": int(training_rows[horizon]),
            "prediction_rows": int(len(joined)),
            "realized_rows": int(len(realized)),
            "mean_predicted_probability": float(np.mean(probability)),
            "minimum_predicted_probability": float(np.min(probability)),
            "maximum_predicted_probability": float(np.max(probability)),
        }
        if len(realized):
            y = realized[target_column].astype(int).to_numpy(float)
            p = pd.to_numeric(
                realized[probability_column], errors="raise"
            ).to_numpy(float)
            summary.update(
                {
                    "actual_up_rate": float(np.mean(y)),
                    "brier_score": float(np.mean((p - y) ** 2)),
                    "log_loss": _binary_log_loss(y, p),
                }
            )
        else:
            summary.update(
                {
                    "actual_up_rate": None,
                    "brier_score": None,
                    "log_loss": None,
                }
            )
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
    fold_summaries: list[dict[str, Any]] = []
    for fold in folds:
        predicted, training_rows = predict_fold(
            frame,
            features,
            fold,
            minimum_training_rows=minimum_training_rows,
        )
        prediction_frames.append(predicted)
        fold_summaries.extend(
            summarize_fold(frame, predicted, fold, training_rows)
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.sort_values(["decision_date", "fold"]).reset_index(drop=True)
    if predictions["decision_date"].duplicated().any():
        raise ValueError("Validation folds overlap on decision_date")

    for horizon in HORIZONS:
        column = f"predicted_p_up_{horizon}d"
        if predictions[column].isna().any():
            raise ValueError(f"Missing probabilities in {column}")
        if not predictions[column].between(0.0, 1.0).all():
            raise ValueError(f"Out-of-range probabilities in {column}")

    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_output, index=False, engine="pyarrow")

    report = {
        "model_name": MODEL_NAME,
        "model_type": "regularized_logistic_classification",
        "feature_set_version": feature_version,
        "feature_count": len(features),
        "dataset_sha256": _sha256(dataset_path),
        "prediction_rows": int(len(predictions)),
        "hyperparameters": {
            "penalty": "l2",
            "C": LOGISTIC_C,
            "solver": "liblinear",
            "max_iter": 2000,
            "random_seed": RANDOM_SEED,
            "imputer": "training_median",
            "scaler": "training_standard_scaler",
        },
        "fold_horizon_evidence": fold_summaries,
        "status": "BASELINE_GENERATED",
        "note": "Provisional V3-005 evidence only; champion selection belongs to later tournament tasks.",
    }
    summary_output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"Wrote {predictions_output} ({len(predictions)} chronological predictions)"
    )
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
