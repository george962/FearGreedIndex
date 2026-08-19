#!/usr/bin/env python3
"""Run the pre-registered EXP-006 opportunity-target experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from v3.models.common import fold_test_mask, load_feature_registry, validate_feature_columns

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config.json"
DATASET = ROOT / "v3" / "data" / "model_dataset_treasury.parquet"
REGISTRY = ROOT / "v3" / "reports" / "feature_registry_treasury.json"
MANIFEST = ROOT / "v3" / "experiments" / "EXP-006" / "manifest.json"
PREDICTIONS = ROOT / "v3" / "reports" / "exp006_predictions.parquet"
METRICS = ROOT / "v3" / "reports" / "exp006_metrics.csv"
STATE_DISTRIBUTION = ROOT / "v3" / "reports" / "exp006_state_distribution.csv"
EVALUATION = ROOT / "v3" / "reports" / "exp006_evaluation.json"

EXPERIMENT_ID = "EXP-006"
AS_OF = pd.Timestamp("2026-08-18")
RETURN_GOOD = 0.02
RETURN_EXCELLENT = 0.05
RETURN_BAD = -0.02
DRAWDOWN_BAD = -0.05
MINIMUM_TRAINING_ROWS = 100

LOGISTIC_NAME = "opportunity_logistic_l2_v1"
RF_NAME = "opportunity_random_forest_v1"

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": 1,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_opportunity_targets(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "forward_return_20d",
        "max_drawdown_20d",
        "_forward_20d_known_date",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"EXP-006 dataset missing target inputs: {missing}")

    result = frame.copy()
    result["_forward_20d_known_date"] = pd.to_datetime(
        result["_forward_20d_known_date"], errors="coerce"
    ).dt.normalize()
    forward_return = pd.to_numeric(result["forward_return_20d"], errors="coerce")
    drawdown = pd.to_numeric(result["max_drawdown_20d"], errors="coerce")
    mature = forward_return.notna() & drawdown.notna() & result["_forward_20d_known_date"].notna()

    favorable = pd.Series(pd.NA, index=result.index, dtype="boolean")
    favorable.loc[mature] = (
        forward_return.loc[mature].ge(RETURN_GOOD)
        & drawdown.loc[mature].gt(DRAWDOWN_BAD)
    )
    result["favorable_entry_20d"] = favorable

    state = pd.Series(pd.NA, index=result.index, dtype="string")
    bad = mature & (forward_return.le(RETURN_BAD) | drawdown.le(DRAWDOWN_BAD))
    excellent = mature & ~bad & forward_return.ge(RETURN_EXCELLENT)
    good = mature & ~bad & ~excellent & forward_return.ge(RETURN_GOOD)
    normal = mature & ~bad & ~excellent & ~good
    state.loc[bad] = "BAD"
    state.loc[excellent] = "EXCELLENT"
    state.loc[good] = "GOOD"
    state.loc[normal] = "NORMAL"
    result["opportunity_state_20d"] = state

    if result.loc[mature, "opportunity_state_20d"].isna().any():
        raise ValueError("EXP-006 mature row did not receive exactly one opportunity state")
    if result.loc[~mature, "opportunity_state_20d"].notna().any():
        raise ValueError("EXP-006 assigned opportunity state before target maturity")
    return result


def eligible_training_mask(frame: pd.DataFrame, cutoff: str | pd.Timestamp) -> pd.Series:
    cutoff_ts = pd.Timestamp(cutoff).normalize()
    return (
        frame["decision_date"].le(cutoff_ts)
        & frame["_forward_20d_known_date"].notna()
        & frame["_forward_20d_known_date"].le(cutoff_ts)
        & frame["favorable_entry_20d"].notna()
    )


def realized_test_mask(frame: pd.DataFrame, fold: dict[str, Any]) -> pd.Series:
    return (
        fold_test_mask(frame, fold)
        & frame["_forward_20d_known_date"].notna()
        & frame["_forward_20d_known_date"].le(AS_OF)
        & frame["favorable_entry_20d"].notna()
    )


def logistic_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def random_forest_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(**RF_PARAMS)),
        ]
    )


def sample_hash(dates: pd.Series) -> str:
    values = pd.to_datetime(dates, errors="raise").dt.strftime("%Y-%m-%d").tolist()
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (p >= edges[index]) & (p <= edges[index + 1])
        else:
            mask = (p >= edges[index]) & (p < edges[index + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        error += (count / total) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return float(error)


def _positive_probability(pipeline: Pipeline, x: pd.DataFrame) -> np.ndarray:
    classes = pipeline.named_steps["model"].classes_.tolist()
    if 1 not in classes:
        raise ValueError("EXP-006 classifier training sample lacks positive class")
    index = classes.index(1)
    return np.clip(np.asarray(pipeline.predict_proba(x)[:, index], dtype=float), 0.0, 1.0)


def evaluate_model(
    model_name: str,
    pipeline_factory: Any,
    frame: pd.DataFrame,
    features: list[str],
    folds: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []

    for fold in folds:
        train = frame.loc[eligible_training_mask(frame, fold["train_end"])].copy()
        test = frame.loc[realized_test_mask(frame, fold)].copy()
        if len(train) < MINIMUM_TRAINING_ROWS:
            raise ValueError(f"EXP-006 fold {fold['name']} has only {len(train)} training rows")
        if test.empty:
            raise ValueError(f"EXP-006 fold {fold['name']} has no mature test rows")

        y_train = train["favorable_entry_20d"].astype(int)
        y_test = test["favorable_entry_20d"].astype(int).to_numpy()
        if y_train.nunique() != 2 or len(np.unique(y_test)) != 2:
            raise ValueError(f"EXP-006 fold {fold['name']} must contain both classes")

        pipeline = pipeline_factory()
        pipeline.fit(train[features], y_train)
        probability = _positive_probability(pipeline, test[features])
        training_prevalence = float(y_train.mean())
        baseline_probability = np.full(len(test), training_prevalence, dtype=float)
        brier = float(np.mean((probability - y_test) ** 2))
        baseline_brier = float(np.mean((baseline_probability - y_test) ** 2))
        relative_brier = (
            float((baseline_brier - brier) / baseline_brier)
            if baseline_brier > 0
            else float("nan")
        )
        auc = float(roc_auc_score(y_test, probability))
        predicted_class = (probability >= 0.5).astype(int)
        date_hash = sample_hash(test["decision_date"])

        metric_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "model_name": model_name,
                "fold": str(fold["name"]),
                "training_rows": int(len(train)),
                "test_rows": int(len(test)),
                "sample_sha256": date_hash,
                "training_prevalence": training_prevalence,
                "test_prevalence": float(np.mean(y_test)),
                "mean_prediction": float(np.mean(probability)),
                "brier_score": brier,
                "baseline_brier_score": baseline_brier,
                "relative_brier_improvement": relative_brier,
                "log_loss": float(log_loss(y_test, probability, labels=[0, 1])),
                "expected_calibration_error": expected_calibration_error(y_test, probability),
                "roc_auc": auc,
                "precision_at_0_5": float(precision_score(y_test, predicted_class, zero_division=0)),
                "recall_at_0_5": float(recall_score(y_test, predicted_class, zero_division=0)),
            }
        )

        prediction_rows.append(
            pd.DataFrame(
                {
                    "decision_date": test["decision_date"].to_numpy(),
                    "fold": str(fold["name"]),
                    "experiment_id": EXPERIMENT_ID,
                    "model_name": model_name,
                    "predicted_favorable_entry_20d": probability,
                    "favorable_entry_20d": y_test,
                    "opportunity_state_20d": test["opportunity_state_20d"].to_numpy(),
                    "forward_return_20d": test["forward_return_20d"].to_numpy(),
                    "max_drawdown_20d": test["max_drawdown_20d"].to_numpy(),
                    "sample_sha256": date_hash,
                }
            )
        )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    summary = summarize_model(metrics)
    return metrics, predictions, summary


def summarize_model(metrics: pd.DataFrame) -> dict[str, Any]:
    mean_relative = float(metrics["relative_brier_improvement"].mean())
    positive_relative_folds = int(metrics["relative_brier_improvement"].gt(0.0).sum())
    mean_auc = float(metrics["roc_auc"].mean())
    positive_auc_folds = int(metrics["roc_auc"].gt(0.5).sum())
    gate_pass = bool(
        mean_relative > 0.0
        and positive_relative_folds >= 2
        and mean_auc > 0.52
        and positive_auc_folds >= 2
    )
    return {
        "mean_relative_brier_improvement": mean_relative,
        "positive_relative_brier_folds": positive_relative_folds,
        "mean_roc_auc": mean_auc,
        "positive_auc_folds": positive_auc_folds,
        "mean_brier_score": float(metrics["brier_score"].mean()),
        "mean_ece": float(metrics["expected_calibration_error"].mean()),
        "viability_gate_pass": gate_pass,
    }


def state_distribution(frame: pd.DataFrame, folds: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        test = frame.loc[realized_test_mask(frame, fold)].copy()
        counts = test["opportunity_state_20d"].value_counts()
        total = int(len(test))
        for state in ("BAD", "NORMAL", "GOOD", "EXCELLENT"):
            count = int(counts.get(state, 0))
            rows.append(
                {
                    "fold": str(fold["name"]),
                    "state": state,
                    "count": count,
                    "share": float(count / total) if total else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def run() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    folds = config.get("validation", {}).get("folds", [])
    if len(folds) != 3:
        raise ValueError("EXP-006 requires the frozen three chronological folds")

    frame = pd.read_parquet(DATASET, engine="pyarrow").copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
    frame = frame.sort_values("decision_date").reset_index(drop=True)
    frame = add_opportunity_targets(frame)
    feature_version, features = load_feature_registry(REGISTRY)
    validate_feature_columns(frame, features)
    if feature_version != manifest.get("feature_set_version"):
        raise ValueError("EXP-006 feature registry version does not match manifest")
    if len(features) != int(manifest.get("feature_count", -1)):
        raise ValueError("EXP-006 feature count does not match manifest")

    logistic_metrics, logistic_predictions, logistic_summary = evaluate_model(
        LOGISTIC_NAME, logistic_pipeline, frame, features, folds
    )
    rf_metrics, rf_predictions, rf_summary = evaluate_model(
        RF_NAME, random_forest_pipeline, frame, features, folds
    )
    metrics = pd.concat([logistic_metrics, rf_metrics], ignore_index=True)
    predictions = pd.concat([logistic_predictions, rf_predictions], ignore_index=True)

    hash_counts = metrics.groupby("fold")["sample_sha256"].nunique()
    sample_hashes_match = bool(hash_counts.eq(1).all())
    if not sample_hashes_match:
        raise ValueError("EXP-006 candidate models used different test samples")

    states = state_distribution(frame, folds)
    experiment_pass = bool(
        sample_hashes_match
        and (logistic_summary["viability_gate_pass"] or rf_summary["viability_gate_pass"])
    )
    decision = (
        "OPPORTUNITY_TARGET_WORTH_FURTHER_RESEARCH"
        if experiment_pass
        else "DO_NOT_ADVANCE_OPPORTUNITY_TARGET_UNDER_EXP_006"
    )

    report = {
        "experiment_id": EXPERIMENT_ID,
        "as_of": AS_OF.strftime("%Y-%m-%d"),
        "status": "EXP_006_EVALUATION_COMPLETE",
        "feature_version": feature_version,
        "feature_count": len(features),
        "dataset_sha256": _sha256(DATASET),
        "target": {
            "primary": "favorable_entry_20d",
            "return_minimum": RETURN_GOOD,
            "drawdown_strictly_greater_than": DRAWDOWN_BAD,
        },
        "sample_hashes_match": sample_hashes_match,
        "models": {
            LOGISTIC_NAME: logistic_summary,
            RF_NAME: rf_summary,
        },
        "experiment_viability_pass": experiment_pass,
        "decision": decision,
        "champion_selected": False,
        "v3_019_eligible": False,
        "current_sizing_multiplier": 1.0,
        "champion_gate_version": "v3-champion-gates-001",
        "note": "EXP-006 tests target formulation only. Passing does not imply champion promotion; failing may not be followed by threshold/model tuning under EXP-006.",
    }

    EVALUATION.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS, index=False)
    states.to_csv(STATE_DISTRIBUTION, index=False)
    predictions.to_parquet(PREDICTIONS, index=False, engine="pyarrow")
    EVALUATION.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    run()
