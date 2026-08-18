#!/usr/bin/env python3
"""Fixed random-forest robustness benchmark for v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
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
DEFAULT_PREDICTIONS = ROOT / "v3" / "reports" / "random_forest_predictions.parquet"
DEFAULT_SUMMARY = ROOT / "v3" / "reports" / "random_forest_summary.json"
MODEL_NAME = "random_forest_v1"
RANDOM_SEED = 42
MINIMUM_TRAINING_ROWS = 100
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
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
            ("model", RandomForestClassifier(**MODEL_PARAMS)),
        ]
    )


def regressor_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(**MODEL_PARAMS)),
        ]
    )


def fit_probability(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> np.ndarray:
    y = pd.to_numeric(y_train, errors="raise").astype(int)
    unique = sorted(y.unique().tolist())
    if not set(unique).issubset({0, 1}):
        raise ValueError(f"Classification target is not binary: {unique}")
    if len(unique) == 1:
        return np.full(len(x_test), float(unique[0]), dtype=float)
    pipeline = classifier_pipeline()
    pipeline.fit(x_train, y)
    classes = pipeline.named_steps["model"].classes_.tolist()
    positive_index = classes.index(1)
    probability = pipeline.predict_proba(x_test)[:, positive_index]
    return np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)


def fit_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
) -> np.ndarray:
    y = pd.to_numeric(y_train, errors="raise").astype(float)
    if not np.isfinite(y.to_numpy()).all():
        raise ValueError("Regression target contains non-finite values")
    pipeline = regressor_pipeline()
    pipeline.fit(x_train, y)
    prediction = np.asarray(pipeline.predict(x_test), dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("Random forest produced non-finite predictions")
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
    ]
    if len(train) < minimum_training_rows:
        raise ValueError(
            f"Horizon {horizon}d target {target_column} has only {len(train)} "
            f"eligible training rows; need {minimum_training_rows}"
        )
    return train


def predict_fold(
    frame: pd.DataFrame,
    features: list[str],
    fold: dict[str, Any],
    minimum_training_rows: int = MINIMUM_TRAINING_ROWS,
) -> tuple[pd.DataFrame, dict[str, int]]:
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
    training_rows: dict[str, int] = {}

    for horizon in HORIZONS:
        class_target = f"forward_positive_{horizon}d"
        train_class = _training_frame(
            frame, horizon, fold["train_end"], class_target, minimum_training_rows
        )
        training_rows[f"p_up_{horizon}d"] = int(len(train_class))
        output[f"predicted_p_up_{horizon}d"] = fit_probability(
            train_class[features], train_class[class_target], test[features]
        )

        return_target = f"forward_return_{horizon}d"
        train_return = _training_frame(
            frame, horizon, fold["train_end"], return_target, minimum_training_rows
        )
        training_rows[f"return_{horizon}d"] = int(len(train_return))
        output[f"predicted_return_{horizon}d"] = fit_regression(
            train_return[features], train_return[return_target], test[features]
        )

    drawdown_target = "max_drawdown_20d"
    train_drawdown = _training_frame(
        frame, 20, fold["train_end"], drawdown_target, minimum_training_rows
    )
    training_rows["drawdown_20d"] = int(len(train_drawdown))
    output["predicted_drawdown_20d"] = fit_regression(
        train_drawdown[features], train_drawdown[drawdown_target], test[features]
    )
    return output.reset_index(drop=True), training_rows


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    clipped = np.clip(p.astype(float), 1e-9, 1.0 - 1e-9)
    target = y.astype(float)
    return float(-np.mean(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped)))


def summarize_fold(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    fold: dict[str, Any],
    training_rows: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        p_col = f"predicted_p_up_{horizon}d"
        class_target = f"forward_positive_{horizon}d"
        class_joined = predictions[["decision_date", p_col]].merge(
            frame[["decision_date", class_target]],
            on="decision_date",
            how="left",
            validate="one_to_one",
        )
        realized_class = class_joined.loc[class_joined[class_target].notna()].copy()
        if len(realized_class):
            y = realized_class[class_target].astype(int).to_numpy(float)
            p = pd.to_numeric(realized_class[p_col], errors="raise").to_numpy(float)
            rows.append(
                {
                    "fold": str(fold["name"]),
                    "target": f"p_up_{horizon}d",
                    "horizon": horizon,
                    "training_rows": training_rows[f"p_up_{horizon}d"],
                    "realized_rows": int(len(realized_class)),
                    "actual_up_rate": float(np.mean(y)),
                    "mean_prediction": float(np.mean(p)),
                    "brier_score": float(np.mean((p - y) ** 2)),
                    "log_loss": _log_loss(y, p),
                }
            )

        r_col = f"predicted_return_{horizon}d"
        return_target = f"forward_return_{horizon}d"
        return_joined = predictions[["decision_date", r_col]].merge(
            frame[["decision_date", return_target]],
            on="decision_date",
            how="left",
            validate="one_to_one",
        )
        realized_return = return_joined.loc[return_joined[return_target].notna()].copy()
        if len(realized_return):
            actual = pd.to_numeric(realized_return[return_target], errors="raise").to_numpy(float)
            pred = pd.to_numeric(realized_return[r_col], errors="raise").to_numpy(float)
            error = pred - actual
            rows.append(
                {
                    "fold": str(fold["name"]),
                    "target": f"return_{horizon}d",
                    "horizon": horizon,
                    "training_rows": training_rows[f"return_{horizon}d"],
                    "realized_rows": int(len(realized_return)),
                    "mean_actual": float(np.mean(actual)),
                    "mean_prediction": float(np.mean(pred)),
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                }
            )

    dd_joined = predictions[["decision_date", "predicted_drawdown_20d"]].merge(
        frame[["decision_date", "max_drawdown_20d"]],
        on="decision_date",
        how="left",
        validate="one_to_one",
    )
    realized_dd = dd_joined.loc[dd_joined["max_drawdown_20d"].notna()].copy()
    if len(realized_dd):
        actual = pd.to_numeric(realized_dd["max_drawdown_20d"], errors="raise").to_numpy(float)
        pred = pd.to_numeric(realized_dd["predicted_drawdown_20d"], errors="raise").to_numpy(float)
        error = pred - actual
        rows.append(
            {
                "fold": str(fold["name"]),
                "target": "drawdown_20d",
                "horizon": 20,
                "training_rows": training_rows["drawdown_20d"],
                "realized_rows": int(len(realized_dd)),
                "mean_actual": float(np.mean(actual)),
                "mean_prediction": float(np.mean(pred)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
            }
        )
    return rows


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
        p_col = f"predicted_p_up_{horizon}d"
        if not predictions[p_col].between(0.0, 1.0).all():
            raise ValueError(f"Out-of-range probabilities in {p_col}")
        r_col = f"predicted_return_{horizon}d"
        if not np.isfinite(pd.to_numeric(predictions[r_col]).to_numpy(float)).all():
            raise ValueError(f"Non-finite return predictions in {r_col}")
    if not np.isfinite(pd.to_numeric(predictions["predicted_drawdown_20d"]).to_numpy(float)).all():
        raise ValueError("Non-finite drawdown predictions")

    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_output, index=False, engine="pyarrow")
    report = {
        "model_name": MODEL_NAME,
        "model_type": "random_forest",
        "feature_set_version": feature_version,
        "feature_count": len(features),
        "dataset_sha256": _sha256(dataset_path),
        "prediction_rows": int(len(predictions)),
        "hyperparameters": MODEL_PARAMS,
        "fold_target_evidence": evidence,
        "status": "BENCHMARK_GENERATED",
        "note": "V3-008 fixed nonlinear robustness benchmark; no champion selection or post-hoc tuning.",
    }
    summary_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {predictions_output} ({len(predictions)} chronological predictions)")
    print(f"Wrote {summary_output}")
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
