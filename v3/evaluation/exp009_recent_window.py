#!/usr/bin/env python3
"""Run pre-registered EXP-009 fixed recent-window adaptation test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from v3.evaluation.exp006_opportunity import (
    RF_NAME,
    _positive_probability,
    add_opportunity_targets,
    eligible_training_mask,
    expected_calibration_error,
    random_forest_pipeline,
    realized_test_mask,
    sample_hash,
)
from v3.models.common import load_feature_registry, validate_feature_columns

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config.json"
DATASET = ROOT / "v3" / "data" / "model_dataset_treasury.parquet"
REGISTRY = ROOT / "v3" / "reports" / "feature_registry_treasury.json"
MANIFEST = ROOT / "v3" / "experiments" / "EXP-009" / "manifest.json"
EXP006_METRICS = ROOT / "v3" / "reports" / "exp006_metrics.csv"
PREDICTIONS = ROOT / "v3" / "reports" / "exp009_predictions.parquet"
METRICS = ROOT / "v3" / "reports" / "exp009_metrics.csv"
EVALUATION = ROOT / "v3" / "reports" / "exp009_evaluation.json"

EXPERIMENT_ID = "EXP-009"
WINDOW_ROWS = 504
MODEL_NAME = "opportunity_random_forest_recent_504_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recent_training_frame(
    frame: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    window_rows: int = WINDOW_ROWS,
) -> pd.DataFrame:
    eligible = frame.loc[eligible_training_mask(frame, cutoff)].copy()
    eligible = eligible.sort_values("decision_date").reset_index(drop=True)
    if len(eligible) < window_rows:
        raise ValueError(
            f"EXP-009 cutoff {pd.Timestamp(cutoff).date()} has only {len(eligible)} "
            f"eligible rows; need exactly {window_rows}"
        )
    recent = eligible.tail(window_rows).reset_index(drop=True)
    if len(recent) != window_rows:
        raise ValueError("EXP-009 recent training window size drift")
    if recent["favorable_entry_20d"].astype(int).nunique() != 2:
        raise ValueError("EXP-009 recent training window lacks both target classes")
    return recent


def load_full_history_rf_metrics(path: Path = EXP006_METRICS) -> dict[str, dict[str, Any]]:
    metrics = pd.read_csv(path)
    required = {"model_name", "fold", "sample_sha256", "brier_score", "roc_auc"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"EXP-006 metrics missing comparison columns: {missing}")
    rf = metrics.loc[metrics["model_name"].astype(str).eq(RF_NAME)].copy()
    if len(rf) != 3:
        raise ValueError(f"Expected 3 frozen EXP-006 RF rows, found {len(rf)}")
    result: dict[str, dict[str, Any]] = {}
    for _, row in rf.iterrows():
        fold = str(row["fold"])
        if fold in result:
            raise ValueError(f"Duplicate EXP-006 RF fold: {fold}")
        result[fold] = {
            "sample_sha256": str(row["sample_sha256"]),
            "brier_score": float(row["brier_score"]),
            "roc_auc": float(row["roc_auc"]),
        }
    return result


def summarize_viability(metrics: pd.DataFrame) -> dict[str, Any]:
    mean_relative = float(metrics["relative_brier_improvement"].mean())
    positive_relative = int(metrics["relative_brier_improvement"].gt(0.0).sum())
    mean_auc = float(metrics["roc_auc"].mean())
    positive_auc = int(metrics["roc_auc"].gt(0.5).sum())
    minimum_auc = float(metrics["roc_auc"].min())
    mean_brier_delta = float(metrics["brier_improvement_vs_full_history"].mean())
    mean_auc_delta = float(metrics["auc_improvement_vs_full_history"].mean())
    brier_better_folds = int(metrics["brier_improvement_vs_full_history"].gt(0.0).sum())
    auc_better_folds = int(metrics["auc_improvement_vs_full_history"].gt(0.0).sum())
    exact_window_rows = bool(metrics["training_rows"].eq(WINDOW_ROWS).all())
    sample_hashes_match = bool(metrics["sample_hash_matches_exp006"].all())

    gate_pass = bool(
        exact_window_rows
        and sample_hashes_match
        and mean_relative > 0.0
        and positive_relative >= 2
        and mean_auc > 0.52
        and positive_auc >= 2
        and minimum_auc >= 0.45
        and mean_brier_delta > 0.0
        and mean_auc_delta > 0.0
        and brier_better_folds >= 2
        and auc_better_folds >= 2
    )
    return {
        "exact_504_training_rows_each_fold": exact_window_rows,
        "sample_hashes_match_exp006": sample_hashes_match,
        "mean_relative_brier_improvement": mean_relative,
        "positive_relative_brier_folds": positive_relative,
        "mean_roc_auc": mean_auc,
        "positive_auc_folds": positive_auc,
        "minimum_fold_roc_auc": minimum_auc,
        "mean_brier_improvement_vs_full_history": mean_brier_delta,
        "mean_auc_improvement_vs_full_history": mean_auc_delta,
        "brier_better_than_full_history_folds": brier_better_folds,
        "auc_better_than_full_history_folds": auc_better_folds,
        "viability_gate_pass": gate_pass,
    }


def run() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    folds = config.get("validation", {}).get("folds", [])
    if len(folds) != 3:
        raise ValueError("EXP-009 requires the frozen three chronological folds")
    if int(manifest.get("adaptation", {}).get("window_eligible_rows", -1)) != WINDOW_ROWS:
        raise ValueError("EXP-009 manifest window drift")

    frame = pd.read_parquet(DATASET, engine="pyarrow").copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
    frame = frame.sort_values("decision_date").reset_index(drop=True)
    frame = add_opportunity_targets(frame)
    feature_version, features = load_feature_registry(REGISTRY)
    validate_feature_columns(frame, features)
    if feature_version != "v3-features-004-treasury":
        raise ValueError("EXP-009 feature version drift")
    if len(features) != 53:
        raise ValueError("EXP-009 feature count drift")

    full_history = load_full_history_rf_metrics()
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []

    for fold in folds:
        fold_name = str(fold["name"])
        train = recent_training_frame(frame, fold["train_end"])
        test = frame.loc[realized_test_mask(frame, fold)].copy()
        if test.empty:
            raise ValueError(f"EXP-009 fold {fold_name} has no mature test rows")
        y_train = train["favorable_entry_20d"].astype(int)
        y_test = test["favorable_entry_20d"].astype(int).to_numpy()
        if len(np.unique(y_test)) != 2:
            raise ValueError(f"EXP-009 fold {fold_name} test target lacks both classes")

        pipeline = random_forest_pipeline()
        pipeline.fit(train[features], y_train)
        probability = _positive_probability(pipeline, test[features])
        recent_prevalence = float(y_train.mean())
        reference = np.full(len(test), recent_prevalence, dtype=float)
        brier = float(np.mean((probability - y_test) ** 2))
        baseline_brier = float(np.mean((reference - y_test) ** 2))
        relative_brier = (
            float((baseline_brier - brier) / baseline_brier)
            if baseline_brier > 0.0
            else float("nan")
        )
        auc = float(roc_auc_score(y_test, probability))
        date_hash = sample_hash(test["decision_date"])
        frozen = full_history.get(fold_name)
        if frozen is None:
            raise ValueError(f"Missing frozen EXP-006 RF comparison for fold {fold_name}")
        hash_match = bool(date_hash == frozen["sample_sha256"])
        if not hash_match:
            raise ValueError(
                f"EXP-009 fold {fold_name} sample hash mismatch: {date_hash} != {frozen['sample_sha256']}"
            )

        brier_delta = float(frozen["brier_score"] - brier)
        auc_delta = float(auc - frozen["roc_auc"])
        metric_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "model_name": MODEL_NAME,
                "fold": fold_name,
                "training_rows": int(len(train)),
                "training_start_date": train["decision_date"].iloc[0].strftime("%Y-%m-%d"),
                "training_end_date": train["decision_date"].iloc[-1].strftime("%Y-%m-%d"),
                "test_rows": int(len(test)),
                "sample_sha256": date_hash,
                "sample_hash_matches_exp006": hash_match,
                "training_prevalence": recent_prevalence,
                "test_prevalence": float(np.mean(y_test)),
                "mean_prediction": float(np.mean(probability)),
                "brier_score": brier,
                "baseline_brier_score": baseline_brier,
                "relative_brier_improvement": relative_brier,
                "log_loss": float(log_loss(y_test, probability, labels=[0, 1])),
                "expected_calibration_error": expected_calibration_error(y_test, probability),
                "roc_auc": auc,
                "full_history_brier_score": float(frozen["brier_score"]),
                "full_history_roc_auc": float(frozen["roc_auc"]),
                "brier_improvement_vs_full_history": brier_delta,
                "auc_improvement_vs_full_history": auc_delta,
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "decision_date": test["decision_date"].to_numpy(),
                    "fold": fold_name,
                    "experiment_id": EXPERIMENT_ID,
                    "model_name": MODEL_NAME,
                    "predicted_favorable_entry_20d": probability,
                    "recent_window_base_rate": reference,
                    "favorable_entry_20d": y_test,
                    "sample_sha256": date_hash,
                }
            )
        )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    viability = summarize_viability(metrics)
    experiment_pass = bool(viability["viability_gate_pass"])
    decision = (
        "RECENT_WINDOW_WORTH_FURTHER_RESEARCH"
        if experiment_pass
        else "DO_NOT_ADVANCE_RECENT_WINDOW_UNDER_EXP_009"
    )
    report = {
        "experiment_id": EXPERIMENT_ID,
        "as_of": "2026-08-18",
        "status": "EXP_009_EVALUATION_COMPLETE",
        "feature_version": feature_version,
        "feature_count": len(features),
        "dataset_sha256": _sha256(DATASET),
        "target_source_experiment": "EXP-006",
        "model_source_experiment": "EXP-006",
        "model_name": MODEL_NAME,
        "adaptation": {"window_eligible_rows": WINDOW_ROWS, "selection": "most_recent_eligible_rows"},
        "viability": viability,
        "experiment_viability_pass": experiment_pass,
        "decision": decision,
        "champion_selected": False,
        "v3_019_eligible": False,
        "current_sizing_multiplier": 1.0,
        "champion_gate_version": "v3-champion-gates-001",
        "note": "EXP-009 changes only training recency. Window length and RF parameters may not be retuned after result inspection under this experiment ID.",
    }

    EVALUATION.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS, index=False)
    predictions.to_parquet(PREDICTIONS, index=False, engine="pyarrow")
    EVALUATION.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    run()
