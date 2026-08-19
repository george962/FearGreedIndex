#!/usr/bin/env python3
"""EXP-005 chronologically calibrated ExtraTrees prediction candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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
DEFAULT_PREDICTIONS = ROOT / "v3" / "reports" / "exp005_predictions.parquet"
DEFAULT_SUMMARY = ROOT / "v3" / "reports" / "exp005_model_summary.json"
MODEL_NAME = "extra_trees_calibrated_v1"
EXPERIMENT_ID = "EXP-005"
RANDOM_SEED = 42
MINIMUM_TRAINING_ROWS = 100
MINIMUM_CALIBRATION_ROWS = 80
CALIBRATION_FRACTION = 0.20

TREE_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "bootstrap": False,
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
}
CALIBRATOR_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "fit_intercept": True,
    "max_iter": 1000,
}


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


def classifier_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(**TREE_PARAMS)),
        ]
    )


def regressor_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(**TREE_PARAMS)),
        ]
    )


def chronological_calibration_split(
    train: pd.DataFrame,
    *,
    minimum_fit_rows: int = MINIMUM_TRAINING_ROWS,
    minimum_calibration_rows: int = MINIMUM_CALIBRATION_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the earliest 80% for base fitting and latest 20% for calibration."""
    if "decision_date" not in train:
        raise ValueError("Classification training frame lacks decision_date")
    ordered = train.sort_values("decision_date").copy()
    if ordered["decision_date"].duplicated().any():
        raise ValueError("Classification training frame has duplicate decision dates")
    split_index = int(math.floor(len(ordered) * (1.0 - CALIBRATION_FRACTION)))
    fit = ordered.iloc[:split_index].copy()
    calibration = ordered.iloc[split_index:].copy()
    if len(fit) < minimum_fit_rows:
        raise ValueError(
            f"EXP-005 base-fit segment has {len(fit)} rows; need {minimum_fit_rows}"
        )
    if len(calibration) < minimum_calibration_rows:
        raise ValueError(
            f"EXP-005 calibration segment has {len(calibration)} rows; need {minimum_calibration_rows}"
        )
    if fit["decision_date"].max() >= calibration["decision_date"].min():
        raise ValueError("EXP-005 chronological fit/calibration split overlaps")
    return fit, calibration


def _binary_target(series: pd.Series, *, label: str) -> pd.Series:
    y = pd.to_numeric(series, errors="raise").astype(int)
    unique = sorted(y.unique().tolist())
    if unique != [0, 1]:
        raise ValueError(f"{label} must contain both binary classes; got {unique}")
    return y


def _positive_probability(pipeline: Pipeline, x: pd.DataFrame) -> np.ndarray:
    classes = pipeline.named_steps["model"].classes_.tolist()
    if 1 not in classes:
        raise ValueError("ExtraTrees classifier does not expose positive class")
    positive_index = classes.index(1)
    return np.asarray(pipeline.predict_proba(x)[:, positive_index], dtype=float)


def fit_calibrated_probability(
    train: pd.DataFrame,
    features: list[str],
    target_column: str,
    x_test: pd.DataFrame,
    *,
    minimum_fit_rows: int = MINIMUM_TRAINING_ROWS,
    minimum_calibration_rows: int = MINIMUM_CALIBRATION_ROWS,
) -> tuple[np.ndarray, dict[str, int]]:
    fit, calibration = chronological_calibration_split(
        train,
        minimum_fit_rows=minimum_fit_rows,
        minimum_calibration_rows=minimum_calibration_rows,
    )
    y_fit = _binary_target(fit[target_column], label="EXP-005 base-fit target")
    y_calibration = _binary_target(
        calibration[target_column], label="EXP-005 calibration target"
    )

    base = classifier_pipeline()
    base.fit(fit[features], y_fit)
    calibration_probability = _positive_probability(base, calibration[features])
    test_probability = _positive_probability(base, x_test)

    calibrator = LogisticRegression(**CALIBRATOR_PARAMS)
    calibrator.fit(calibration_probability.reshape(-1, 1), y_calibration)
    calibrated = calibrator.predict_proba(test_probability.reshape(-1, 1))[:, 1]
    calibrated = np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)
    if not np.isfinite(calibrated).all():
        raise ValueError("EXP-005 produced non-finite calibrated probabilities")
    return calibrated, {
        "eligible_rows": int(len(train)),
        "base_fit_rows": int(len(fit)),
        "calibration_rows": int(len(calibration)),
    }


def fit_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> np.ndarray:
    y = pd.to_numeric(y_train, errors="raise").astype(float)
    if not np.isfinite(y.to_numpy()).all():
        raise ValueError("EXP-005 regression target contains non-finite values")
    pipeline = regressor_pipeline()
    pipeline.fit(x_train, y)
    prediction = np.asarray(pipeline.predict(x_test), dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("EXP-005 produced non-finite regression predictions")
    return prediction


def _training_frame(
    frame: pd.DataFrame,
    horizon: int,
    cutoff: str,
    target_column: str,
    minimum_training_rows: int,
) -> pd.DataFrame:
    train = frame.loc[
        eligible_training_mask(
            frame,
            horizon,
            cutoff,
            target_column=target_column,
        )
    ].copy()
    if len(train) < minimum_training_rows:
        raise ValueError(
            f"Horizon {horizon}d target {target_column} has only {len(train)} "
            f"eligible rows; need {minimum_training_rows}"
        )
    return train


def predict_fold(
    frame: pd.DataFrame,
    features: list[str],
    fold: dict[str, Any],
    minimum_training_rows: int = MINIMUM_TRAINING_ROWS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    training_rows: dict[str, Any] = {}

    for horizon in HORIZONS:
        class_target = f"forward_positive_{horizon}d"
        train_class = _training_frame(
            frame, horizon, fold["train_end"], class_target, minimum_training_rows
        )
        probability, class_rows = fit_calibrated_probability(
            train_class,
            features,
            class_target,
            test[features],
            minimum_fit_rows=minimum_training_rows,
            minimum_calibration_rows=MINIMUM_CALIBRATION_ROWS,
        )
        output[f"predicted_p_up_{horizon}d"] = probability
        training_rows[f"p_up_{horizon}d"] = class_rows

        return_target = f"forward_return_{horizon}d"
        train_return = _training_frame(
            frame, horizon, fold["train_end"], return_target, minimum_training_rows
        )
        output[f"predicted_return_{horizon}d"] = fit_regression(
            train_return[features], train_return[return_target], test[features]
        )
        training_rows[f"return_{horizon}d"] = int(len(train_return))

    drawdown_target = "max_drawdown_20d"
    train_drawdown = _training_frame(
        frame, 20, fold["train_end"], drawdown_target, minimum_training_rows
    )
    output["predicted_drawdown_20d"] = fit_regression(
        train_drawdown[features], train_drawdown[drawdown_target], test[features]
    )
    training_rows["drawdown_20d"] = int(len(train_drawdown))
    return output.reset_index(drop=True), training_rows


def run_candidate(
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

    outputs: list[pd.DataFrame] = []
    training_evidence: dict[str, Any] = {}
    for fold in folds:
        predicted, rows = predict_fold(
            frame,
            features,
            fold,
            minimum_training_rows=minimum_training_rows,
        )
        predicted["experiment_id"] = EXPERIMENT_ID
        predicted["model_name"] = MODEL_NAME
        predicted["model_family"] = "extra_trees_calibrated"
        outputs.append(predicted)
        training_evidence[str(fold["name"])] = rows

    predictions = pd.concat(outputs, ignore_index=True).sort_values("decision_date")
    if predictions["decision_date"].duplicated().any():
        raise ValueError("EXP-005 validation folds overlap on decision_date")
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_output, index=False, engine="pyarrow")

    report = {
        "experiment_id": EXPERIMENT_ID,
        "model_name": MODEL_NAME,
        "model_type": "extra_trees_with_chronological_sigmoid_calibration",
        "feature_set_version": feature_version,
        "feature_count": len(features),
        "dataset_sha256": _sha256(dataset_path),
        "prediction_rows": int(len(predictions)),
        "tree_hyperparameters": TREE_PARAMS,
        "calibrator_hyperparameters": CALIBRATOR_PARAMS,
        "calibration_fraction": CALIBRATION_FRACTION,
        "minimum_calibration_rows": MINIMUM_CALIBRATION_ROWS,
        "training_evidence": training_evidence,
        "status": "EXP_005_PREDICTIONS_GENERATED",
        "note": "Pre-registered in issue #44. No post-result tuning, feature changes, policy changes, or sizing changes are permitted in EXP-005.",
    }
    summary_output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> int:
    args = parse_args()
    report = run_candidate(
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
