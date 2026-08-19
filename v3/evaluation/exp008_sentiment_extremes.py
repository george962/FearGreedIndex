#!/usr/bin/env python3
"""Run pre-registered EXP-008 fixed Fear & Greed sentiment-state test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from v3.evaluation.exp006_opportunity import (
    add_opportunity_targets,
    eligible_training_mask,
    expected_calibration_error,
    realized_test_mask,
    sample_hash,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config.json"
DATASET = ROOT / "v3" / "data" / "model_dataset_treasury.parquet"
MANIFEST = ROOT / "v3" / "experiments" / "EXP-008" / "manifest.json"
EXP006_METRICS = ROOT / "v3" / "reports" / "exp006_metrics.csv"
PREDICTIONS = ROOT / "v3" / "reports" / "exp008_predictions.parquet"
METRICS = ROOT / "v3" / "reports" / "exp008_metrics.csv"
DIAGNOSTICS = ROOT / "v3" / "reports" / "exp008_sentiment_diagnostics.csv"
EVALUATION = ROOT / "v3" / "reports" / "exp008_evaluation.json"

EXPERIMENT_ID = "EXP-008"
STATE_FEATURE = "fear_greed"
EXTREME_FEAR = "EXTREME_FEAR"
NEUTRAL_RANGE = "NEUTRAL_RANGE"
EXTREME_GREED = "EXTREME_GREED"
STATES = (EXTREME_FEAR, NEUTRAL_RANGE, EXTREME_GREED)
FEAR_MAX = 25.0
GREED_MIN = 75.0
MINIMUM_STATE_TRAINING_ROWS = 50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_sentiment_state(frame: pd.DataFrame) -> pd.DataFrame:
    if STATE_FEATURE not in frame.columns:
        raise ValueError(f"EXP-008 dataset missing state feature: {STATE_FEATURE}")
    result = frame.copy()
    value = pd.to_numeric(result[STATE_FEATURE], errors="coerce")
    state = pd.Series(pd.NA, index=result.index, dtype="string")
    observed = value.notna()
    state.loc[observed & value.le(FEAR_MAX)] = EXTREME_FEAR
    state.loc[observed & value.gt(FEAR_MAX) & value.lt(GREED_MIN)] = NEUTRAL_RANGE
    state.loc[observed & value.ge(GREED_MIN)] = EXTREME_GREED
    result["exp008_sentiment_state"] = state
    return result


def load_exp006_sample_hashes(path: Path = EXP006_METRICS) -> dict[str, str]:
    metrics = pd.read_csv(path)
    required = {"fold", "sample_sha256"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"EXP-006 metrics missing sample columns: {missing}")
    result: dict[str, str] = {}
    metrics = metrics.copy()
    metrics["fold"] = metrics["fold"].astype(str)
    for fold, group in metrics.groupby("fold", sort=False):
        hashes = sorted(group["sample_sha256"].dropna().astype(str).unique().tolist())
        if len(hashes) != 1:
            raise ValueError(f"EXP-006 fold {fold} does not have one unique sample hash")
        result[str(fold)] = hashes[0]
    return result


def state_training_evidence(train: pd.DataFrame) -> tuple[dict[str, dict[str, float | int | bool]], bool]:
    evidence: dict[str, dict[str, float | int | bool]] = {}
    support_pass = True
    for state in STATES:
        group = train.loc[train["exp008_sentiment_state"].eq(state)].copy()
        target = group["favorable_entry_20d"].dropna().astype(int)
        rows = int(len(group))
        both_classes = bool(target.nunique() == 2)
        enough_rows = bool(rows >= MINIMUM_STATE_TRAINING_ROWS)
        if rows == 0:
            prevalence = float("nan")
        else:
            prevalence = float(target.mean())
        support_pass = support_pass and enough_rows and both_classes and np.isfinite(prevalence)
        evidence[state] = {
            "training_rows": rows,
            "training_prevalence": prevalence,
            "minimum_rows_pass": enough_rows,
            "both_target_classes": both_classes,
        }
    return evidence, bool(support_pass)


def ordered_as_hypothesized(prevalence: dict[str, float]) -> bool:
    return bool(
        prevalence[EXTREME_FEAR] > prevalence[NEUTRAL_RANGE]
        and prevalence[NEUTRAL_RANGE] > prevalence[EXTREME_GREED]
    )


def summarize_viability(
    metrics: pd.DataFrame,
    *,
    full_coverage: bool,
    sample_hashes_match: bool,
    support_pass: bool,
    ordered_folds: int,
) -> dict[str, Any]:
    mean_relative = float(metrics["relative_brier_improvement"].mean())
    positive_relative = int(metrics["relative_brier_improvement"].gt(0.0).sum())
    mean_auc = float(metrics["roc_auc"].mean())
    positive_auc = int(metrics["roc_auc"].gt(0.5).sum())
    minimum_auc = float(metrics["roc_auc"].min())
    gate_pass = bool(
        full_coverage
        and sample_hashes_match
        and support_pass
        and mean_relative > 0.0
        and positive_relative >= 2
        and mean_auc > 0.52
        and positive_auc >= 2
        and minimum_auc >= 0.45
        and ordered_folds >= 2
    )
    return {
        "full_mature_test_state_coverage": bool(full_coverage),
        "sample_hashes_match_exp006": bool(sample_hashes_match),
        "state_training_support_pass": bool(support_pass),
        "mean_relative_brier_improvement": mean_relative,
        "positive_relative_brier_folds": positive_relative,
        "mean_roc_auc": mean_auc,
        "positive_auc_folds": positive_auc,
        "minimum_fold_roc_auc": minimum_auc,
        "hypothesized_prevalence_ordering_folds": int(ordered_folds),
        "viability_gate_pass": gate_pass,
    }


def run() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    folds = config.get("validation", {}).get("folds", [])
    if len(folds) != 3:
        raise ValueError("EXP-008 requires the frozen three chronological folds")

    frame = pd.read_parquet(DATASET, engine="pyarrow").copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
    frame = frame.sort_values("decision_date").reset_index(drop=True)
    frame = add_opportunity_targets(frame)
    frame = add_sentiment_state(frame)

    if manifest.get("feature_set_version") != "v3-features-004-treasury":
        raise ValueError("EXP-008 manifest feature version drift")
    state_manifest = manifest.get("sentiment_state", {})
    if state_manifest.get("source_feature") != STATE_FEATURE:
        raise ValueError("EXP-008 manifest state feature drift")
    if float(state_manifest.get("extreme_fear_max", float("nan"))) != FEAR_MAX:
        raise ValueError("EXP-008 extreme-fear threshold drift")
    if float(state_manifest.get("extreme_greed_min", float("nan"))) != GREED_MIN:
        raise ValueError("EXP-008 extreme-greed threshold drift")

    expected_hashes = load_exp006_sample_hashes()
    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    full_coverage = True
    sample_hashes_match = True
    all_support_pass = True
    ordered_folds = 0
    ordering_by_fold: list[str] = []

    for fold in folds:
        fold_name = str(fold["name"])
        train = frame.loc[eligible_training_mask(frame, fold["train_end"])].copy()
        test = frame.loc[realized_test_mask(frame, fold)].copy()
        if test.empty:
            raise ValueError(f"EXP-008 fold {fold_name} has no mature test rows")

        missing_test_state = int(test["exp008_sentiment_state"].isna().sum())
        if missing_test_state:
            full_coverage = False
            raise ValueError(
                f"EXP-008 fold {fold_name} has {missing_test_state} mature test rows with missing state"
            )

        support, fold_support_pass = state_training_evidence(train)
        all_support_pass = all_support_pass and fold_support_pass
        y_train = train["favorable_entry_20d"].astype(int)
        global_prevalence = float(y_train.mean())
        state_prevalence = {
            state: float(support[state]["training_prevalence"])
            for state in STATES
        }

        order_pass = all(np.isfinite(state_prevalence[state]) for state in STATES) and ordered_as_hypothesized(state_prevalence)
        if order_pass:
            ordered_folds += 1
            ordering_label = "EXTREME_FEAR_GT_NEUTRAL_GT_EXTREME_GREED"
        else:
            ordering_label = "NOT_HYPOTHESIZED_ORDER"
        ordering_by_fold.append(ordering_label)

        y_test = test["favorable_entry_20d"].astype(int).to_numpy()
        if len(np.unique(y_test)) != 2:
            raise ValueError(f"EXP-008 fold {fold_name} test target lacks both classes")
        probability = test["exp008_sentiment_state"].map(state_prevalence).astype(float).to_numpy()
        if not np.isfinite(probability).all():
            raise ValueError(f"EXP-008 fold {fold_name} produced non-finite state probabilities")
        reference = np.full(len(test), global_prevalence, dtype=float)
        brier = float(np.mean((probability - y_test) ** 2))
        reference_brier = float(np.mean((reference - y_test) ** 2))
        relative_brier = (
            float((reference_brier - brier) / reference_brier)
            if reference_brier > 0.0
            else float("nan")
        )
        auc = float(roc_auc_score(y_test, probability))
        date_hash = sample_hash(test["decision_date"])
        expected_hash = expected_hashes.get(fold_name)
        hash_match = bool(expected_hash is not None and date_hash == expected_hash)
        sample_hashes_match = sample_hashes_match and hash_match
        if not hash_match:
            raise ValueError(
                f"EXP-008 fold {fold_name} sample hash does not match EXP-006: "
                f"{date_hash} != {expected_hash}"
            )

        metric_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "fold": fold_name,
                "training_rows": int(len(train)),
                "test_rows": int(len(test)),
                "sample_sha256": date_hash,
                "global_training_prevalence": global_prevalence,
                "mean_state_prediction": float(np.mean(probability)),
                "test_prevalence": float(np.mean(y_test)),
                "brier_score": brier,
                "global_baseline_brier_score": reference_brier,
                "relative_brier_improvement": relative_brier,
                "log_loss": float(log_loss(y_test, probability, labels=[0, 1])),
                "expected_calibration_error": expected_calibration_error(y_test, probability),
                "roc_auc": auc,
                "state_support_pass": bool(fold_support_pass),
                "training_prevalence_ordering": ordering_label,
            }
        )

        for state in STATES:
            test_group = test.loc[test["exp008_sentiment_state"].eq(state)]
            diagnostic_rows.append(
                {
                    "fold": fold_name,
                    "sentiment_state": state,
                    "training_rows": int(support[state]["training_rows"]),
                    "training_prevalence": state_prevalence[state],
                    "minimum_rows_pass": bool(support[state]["minimum_rows_pass"]),
                    "both_target_classes": bool(support[state]["both_target_classes"]),
                    "test_rows": int(len(test_group)),
                    "test_favorable_rate": (
                        float(test_group["favorable_entry_20d"].astype(int).mean())
                        if len(test_group)
                        else float("nan")
                    ),
                    "global_training_prevalence": global_prevalence,
                    "training_prevalence_ordering": ordering_label,
                }
            )

        prediction_rows.append(
            pd.DataFrame(
                {
                    "decision_date": test["decision_date"].to_numpy(),
                    "fold": fold_name,
                    "experiment_id": EXPERIMENT_ID,
                    "sentiment_state": test["exp008_sentiment_state"].to_numpy(),
                    "predicted_favorable_entry_20d": probability,
                    "global_reference_probability": reference,
                    "favorable_entry_20d": y_test,
                    "sample_sha256": date_hash,
                }
            )
        )

    metrics = pd.DataFrame(metric_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    viability = summarize_viability(
        metrics,
        full_coverage=full_coverage,
        sample_hashes_match=sample_hashes_match,
        support_pass=all_support_pass,
        ordered_folds=ordered_folds,
    )
    experiment_pass = bool(viability["viability_gate_pass"])
    decision = (
        "SENTIMENT_EXTREMES_WORTH_FURTHER_RESEARCH"
        if experiment_pass
        else "DO_NOT_ADVANCE_SENTIMENT_EXTREMES_UNDER_EXP_008"
    )

    report = {
        "experiment_id": EXPERIMENT_ID,
        "as_of": "2026-08-18",
        "status": "EXP_008_EVALUATION_COMPLETE",
        "feature_version": "v3-features-004-treasury",
        "feature_count": 53,
        "dataset_sha256": _sha256(DATASET),
        "target_source_experiment": "EXP-006",
        "sentiment_state_feature": STATE_FEATURE,
        "sentiment_state_definition": {
            EXTREME_FEAR: "fear_greed <= 25",
            NEUTRAL_RANGE: "25 < fear_greed < 75",
            EXTREME_GREED: "fear_greed >= 75",
        },
        "viability": viability,
        "training_prevalence_ordering_by_fold": ordering_by_fold,
        "experiment_viability_pass": experiment_pass,
        "decision": decision,
        "champion_selected": False,
        "v3_019_eligible": False,
        "current_sizing_multiplier": 1.0,
        "champion_gate_version": "v3-champion-gates-001",
        "note": "EXP-008 tests one fixed Fear & Greed state hypothesis with no ML model. Thresholds may not be retuned after result inspection.",
    }

    EVALUATION.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS, index=False)
    diagnostics.to_csv(DIAGNOSTICS, index=False)
    predictions.to_parquet(PREDICTIONS, index=False, engine="pyarrow")
    EVALUATION.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    run()
