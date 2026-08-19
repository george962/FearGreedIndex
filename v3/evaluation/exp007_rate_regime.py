#!/usr/bin/env python3
"""Run pre-registered EXP-007 fixed 10Y-rate regime test."""

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
MANIFEST = ROOT / "v3" / "experiments" / "EXP-007" / "manifest.json"
EXP006_METRICS = ROOT / "v3" / "reports" / "exp006_metrics.csv"
PREDICTIONS = ROOT / "v3" / "reports" / "exp007_predictions.parquet"
METRICS = ROOT / "v3" / "reports" / "exp007_metrics.csv"
DIAGNOSTICS = ROOT / "v3" / "reports" / "exp007_regime_diagnostics.csv"
EVALUATION = ROOT / "v3" / "reports" / "exp007_evaluation.json"

EXPERIMENT_ID = "EXP-007"
REGIME_FEATURE = "treasury_10y_change_20"
RISING = "RATES_RISING"
FALLING = "RATES_FALLING_OR_FLAT"
REGIMES = (RISING, FALLING)
MINIMUM_REGIME_TRAINING_ROWS = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_rate_regime(frame: pd.DataFrame) -> pd.DataFrame:
    if REGIME_FEATURE not in frame.columns:
        raise ValueError(f"EXP-007 dataset missing regime feature: {REGIME_FEATURE}")
    result = frame.copy()
    value = pd.to_numeric(result[REGIME_FEATURE], errors="coerce")
    regime = pd.Series(pd.NA, index=result.index, dtype="string")
    observed = value.notna()
    regime.loc[observed & value.gt(0.0)] = RISING
    regime.loc[observed & value.le(0.0)] = FALLING
    result["exp007_rate_regime"] = regime
    return result


def load_exp006_sample_hashes(path: Path = EXP006_METRICS) -> dict[str, str]:
    metrics = pd.read_csv(path)
    required = {"fold", "sample_sha256"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"EXP-006 metrics missing sample columns: {missing}")
    metrics = metrics.copy()
    metrics["fold"] = metrics["fold"].astype(str)
    result: dict[str, str] = {}
    for fold, group in metrics.groupby("fold", sort=False):
        hashes = sorted(group["sample_sha256"].dropna().astype(str).unique().tolist())
        if len(hashes) != 1:
            raise ValueError(f"EXP-006 fold {fold} does not have one unique sample hash")
        result[str(fold)] = hashes[0]
    return result


def validate_regime_training_support(train: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    evidence: dict[str, dict[str, float | int]] = {}
    for regime in REGIMES:
        group = train.loc[train["exp007_rate_regime"].eq(regime)].copy()
        if len(group) < MINIMUM_REGIME_TRAINING_ROWS:
            raise ValueError(
                f"EXP-007 regime {regime} has {len(group)} eligible training rows; "
                f"need {MINIMUM_REGIME_TRAINING_ROWS}"
            )
        target = group["favorable_entry_20d"].astype(int)
        if target.nunique() != 2:
            raise ValueError(f"EXP-007 regime {regime} training target lacks both classes")
        evidence[regime] = {
            "training_rows": int(len(group)),
            "training_prevalence": float(target.mean()),
        }
    return evidence


def summarize_viability(
    metrics: pd.DataFrame,
    *,
    full_coverage: bool,
    sample_hashes_match: bool,
    support_pass: bool,
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
    )
    return {
        "full_mature_test_regime_coverage": bool(full_coverage),
        "sample_hashes_match_exp006": bool(sample_hashes_match),
        "regime_training_support_pass": bool(support_pass),
        "mean_relative_brier_improvement": mean_relative,
        "positive_relative_brier_folds": positive_relative,
        "mean_roc_auc": mean_auc,
        "positive_auc_folds": positive_auc,
        "minimum_fold_roc_auc": minimum_auc,
        "viability_gate_pass": gate_pass,
    }


def run() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    folds = config.get("validation", {}).get("folds", [])
    if len(folds) != 3:
        raise ValueError("EXP-007 requires the frozen three chronological folds")

    frame = pd.read_parquet(DATASET, engine="pyarrow").copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
    frame = frame.sort_values("decision_date").reset_index(drop=True)
    frame = add_opportunity_targets(frame)
    frame = add_rate_regime(frame)

    if manifest.get("feature_set_version") != "v3-features-004-treasury":
        raise ValueError("EXP-007 manifest feature version drift")
    if manifest.get("regime", {}).get("source_feature") != REGIME_FEATURE:
        raise ValueError("EXP-007 manifest regime feature drift")

    expected_hashes = load_exp006_sample_hashes()
    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    full_coverage = True
    support_pass = True
    sample_hashes_match = True
    ordering: list[str] = []

    for fold in folds:
        fold_name = str(fold["name"])
        train = frame.loc[eligible_training_mask(frame, fold["train_end"])].copy()
        test = frame.loc[realized_test_mask(frame, fold)].copy()
        if test.empty:
            raise ValueError(f"EXP-007 fold {fold_name} has no mature test rows")

        missing_test_regime = int(test["exp007_rate_regime"].isna().sum())
        if missing_test_regime:
            full_coverage = False
            raise ValueError(
                f"EXP-007 fold {fold_name} has {missing_test_regime} mature test rows with missing regime"
            )

        support = validate_regime_training_support(train)
        y_train = train["favorable_entry_20d"].astype(int)
        global_prevalence = float(y_train.mean())

        regime_prevalence = {
            regime: float(support[regime]["training_prevalence"])
            for regime in REGIMES
        }
        if regime_prevalence[RISING] > regime_prevalence[FALLING]:
            ordering_label = "RISING_HIGHER"
        elif regime_prevalence[RISING] < regime_prevalence[FALLING]:
            ordering_label = "FALLING_HIGHER"
        else:
            ordering_label = "EQUAL"
        ordering.append(ordering_label)

        y_test = test["favorable_entry_20d"].astype(int).to_numpy()
        if len(np.unique(y_test)) != 2:
            raise ValueError(f"EXP-007 fold {fold_name} test target lacks both classes")
        probability = test["exp007_rate_regime"].map(regime_prevalence).astype(float).to_numpy()
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
                f"EXP-007 fold {fold_name} sample hash does not match EXP-006: "
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
                "mean_regime_prediction": float(np.mean(probability)),
                "test_prevalence": float(np.mean(y_test)),
                "brier_score": brier,
                "global_baseline_brier_score": reference_brier,
                "relative_brier_improvement": relative_brier,
                "log_loss": float(log_loss(y_test, probability, labels=[0, 1])),
                "expected_calibration_error": expected_calibration_error(y_test, probability),
                "roc_auc": auc,
                "training_prevalence_ordering": ordering_label,
            }
        )

        for regime in REGIMES:
            test_group = test.loc[test["exp007_rate_regime"].eq(regime)]
            diagnostic_rows.append(
                {
                    "fold": fold_name,
                    "regime": regime,
                    "training_rows": int(support[regime]["training_rows"]),
                    "training_prevalence": regime_prevalence[regime],
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
                    "rate_regime": test["exp007_rate_regime"].to_numpy(),
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
        support_pass=support_pass,
    )
    ordering_stable = len(set(ordering)) == 1
    experiment_pass = bool(viability["viability_gate_pass"])
    decision = (
        "RATE_REGIME_WORTH_FURTHER_RESEARCH"
        if experiment_pass
        else "DO_NOT_ADVANCE_RATE_REGIME_UNDER_EXP_007"
    )

    report = {
        "experiment_id": EXPERIMENT_ID,
        "as_of": "2026-08-18",
        "status": "EXP_007_EVALUATION_COMPLETE",
        "feature_version": "v3-features-004-treasury",
        "feature_count": 53,
        "dataset_sha256": _sha256(DATASET),
        "target_source_experiment": "EXP-006",
        "regime_feature": REGIME_FEATURE,
        "regime_definition": {
            RISING: "> 0",
            FALLING: "<= 0",
        },
        "viability": viability,
        "training_prevalence_ordering_by_fold": ordering,
        "training_prevalence_ordering_stable": ordering_stable,
        "experiment_viability_pass": experiment_pass,
        "decision": decision,
        "champion_selected": False,
        "v3_019_eligible": False,
        "current_sizing_multiplier": 1.0,
        "champion_gate_version": "v3-champion-gates-001",
        "note": "EXP-007 tests one fixed Treasury regime hypothesis with no ML model. It may not be retuned after result inspection.",
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
