#!/usr/bin/env python3
"""DIAG-001: diagnose target, covariate, and feature-target relationship drift.

This module is deliberately diagnostic-only. It fits no predictive model and
cannot change champion, policy, or sizing state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
from sklearn.metrics import roc_auc_score

from v3.evaluation.exp006_opportunity import (
    add_opportunity_targets,
    eligible_training_mask,
    realized_test_mask,
    sample_hash,
)
from v3.models.common import load_feature_registry, validate_feature_columns

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config.json"
DATASET = ROOT / "v3" / "data" / "model_dataset_treasury.parquet"
REGISTRY = ROOT / "v3" / "reports" / "feature_registry_treasury.json"
EXP006_METRICS = ROOT / "v3" / "reports" / "exp006_metrics.csv"
TARGET_DRIFT = ROOT / "v3" / "reports" / "diag001_target_drift.csv"
FEATURE_DRIFT = ROOT / "v3" / "reports" / "diag001_feature_drift.csv"
ASSOCIATION = ROOT / "v3" / "reports" / "diag001_feature_target_association.csv"
STABILITY = ROOT / "v3" / "reports" / "diag001_feature_stability.csv"
SUMMARY = ROOT / "v3" / "reports" / "diag001_summary.json"

DIAGNOSTIC_ID = "DIAG-001"
TARGET = "favorable_entry_20d"
MIN_ASSOCIATION_ROWS = 20
FOLD_ORDER = ("2024", "2025", "2026_ytd")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_categories(path: Path = REGISTRY) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["name"]): str(item.get("category", "unknown")) for item in payload["features"]}


def load_exp006_hashes(path: Path = EXP006_METRICS) -> dict[str, str]:
    metrics = pd.read_csv(path)
    required = {"fold", "sample_sha256"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"EXP-006 metrics missing columns: {missing}")
    result: dict[str, str] = {}
    metrics = metrics.copy()
    metrics["fold"] = metrics["fold"].astype(str)
    for fold, group in metrics.groupby("fold", sort=False):
        hashes = group["sample_sha256"].dropna().astype(str).unique().tolist()
        if len(hashes) != 1:
            raise ValueError(f"EXP-006 fold {fold} has {len(hashes)} sample hashes")
        result[str(fold)] = hashes[0]
    return result


def _numeric_observed(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def pooled_standardized_mean_difference(train: np.ndarray, test: np.ndarray) -> float:
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    train = train[np.isfinite(train)]
    test = test[np.isfinite(test)]
    if len(train) < 2 or len(test) < 2:
        return float("nan")
    train_var = float(np.var(train, ddof=1))
    test_var = float(np.var(test, ddof=1))
    denominator_df = len(train) + len(test) - 2
    if denominator_df <= 0:
        return float("nan")
    pooled_var = ((len(train) - 1) * train_var + (len(test) - 1) * test_var) / denominator_df
    if not np.isfinite(pooled_var) or pooled_var <= 0.0:
        return float("nan")
    return float((np.mean(test) - np.mean(train)) / np.sqrt(pooled_var))


def distribution_metrics(train_series: pd.Series, test_series: pd.Series) -> dict[str, float | int]:
    train = _numeric_observed(train_series)
    test = _numeric_observed(test_series)
    train_missing_rate = float(pd.to_numeric(train_series, errors="coerce").isna().mean())
    test_missing_rate = float(pd.to_numeric(test_series, errors="coerce").isna().mean())

    def quantiles(values: np.ndarray) -> tuple[float, float, float]:
        if len(values) == 0:
            return float("nan"), float("nan"), float("nan")
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        return float(q1), float(median), float(q3)

    train_q1, train_median, train_q3 = quantiles(train)
    test_q1, test_median, test_q3 = quantiles(test)
    train_iqr = train_q3 - train_q1 if np.isfinite(train_q1) and np.isfinite(train_q3) else float("nan")
    test_iqr = test_q3 - test_q1 if np.isfinite(test_q1) and np.isfinite(test_q3) else float("nan")

    if len(train) >= 2 and len(test) >= 2:
        ks = ks_2samp(train, test, alternative="two-sided", method="auto")
        ks_statistic = float(ks.statistic)
        ks_pvalue = float(ks.pvalue)
        wasserstein = float(wasserstein_distance(train, test))
    else:
        ks_statistic = float("nan")
        ks_pvalue = float("nan")
        wasserstein = float("nan")

    normalized_wasserstein = (
        float(wasserstein / abs(train_iqr))
        if np.isfinite(wasserstein) and np.isfinite(train_iqr) and abs(train_iqr) > 0.0
        else float("nan")
    )

    return {
        "train_rows": int(len(train_series)),
        "test_rows": int(len(test_series)),
        "train_observed_rows": int(len(train)),
        "test_observed_rows": int(len(test)),
        "train_missing_rate": train_missing_rate,
        "test_missing_rate": test_missing_rate,
        "missing_rate_delta": float(test_missing_rate - train_missing_rate),
        "train_mean": float(np.mean(train)) if len(train) else float("nan"),
        "test_mean": float(np.mean(test)) if len(test) else float("nan"),
        "train_q1": train_q1,
        "train_median": train_median,
        "train_q3": train_q3,
        "train_iqr": train_iqr,
        "test_q1": test_q1,
        "test_median": test_median,
        "test_q3": test_q3,
        "test_iqr": test_iqr,
        "standardized_mean_difference": pooled_standardized_mean_difference(train, test),
        "ks_statistic": ks_statistic,
        "ks_pvalue": ks_pvalue,
        "wasserstein_distance": wasserstein,
        "normalized_wasserstein_train_iqr": normalized_wasserstein,
    }


def association_sign(value: float) -> str:
    if not np.isfinite(value):
        return "UNAVAILABLE"
    if value > 0.0:
        return "POSITIVE"
    if value < 0.0:
        return "NEGATIVE"
    return "ZERO"


def association_metrics(frame: pd.DataFrame, feature: str) -> dict[str, float | int | str | bool]:
    values = pd.to_numeric(frame[feature], errors="coerce")
    target = pd.to_numeric(frame[TARGET], errors="coerce")
    observed = values.notna() & target.notna()
    x = values.loc[observed].astype(float)
    y = target.loc[observed].astype(int)
    observed_rows = int(len(x))
    usable = bool(
        observed_rows >= MIN_ASSOCIATION_ROWS
        and x.nunique(dropna=True) >= 2
        and y.nunique(dropna=True) == 2
    )
    if not usable:
        return {
            "observed_rows": observed_rows,
            "available": False,
            "spearman": float("nan"),
            "roc_auc_raw_upward": float("nan"),
            "sign": "UNAVAILABLE",
        }

    spearman = float(spearmanr(x.to_numpy(float), y.to_numpy(float)).statistic)
    auc = float(roc_auc_score(y.to_numpy(int), x.to_numpy(float)))
    if not np.isfinite(spearman) or not np.isfinite(auc):
        return {
            "observed_rows": observed_rows,
            "available": False,
            "spearman": float("nan"),
            "roc_auc_raw_upward": float("nan"),
            "sign": "UNAVAILABLE",
        }
    return {
        "observed_rows": observed_rows,
        "available": True,
        "spearman": spearman,
        "roc_auc_raw_upward": auc,
        "sign": association_sign(spearman),
    }


def sign_reversal(train_sign: str, test_sign: str) -> bool | None:
    usable = {"POSITIVE", "NEGATIVE"}
    if train_sign not in usable or test_sign not in usable:
        return None
    return bool(train_sign != test_sign)


def adjacent_sign_transitions(signs: list[str]) -> int:
    usable = [sign for sign in signs if sign in {"POSITIVE", "NEGATIVE"}]
    return int(sum(left != right for left, right in zip(usable, usable[1:])))


def _finite_aggregate(values: pd.Series, operation: str) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    numeric = numeric[np.isfinite(numeric)]
    if len(numeric) == 0:
        return float("nan")
    if operation == "mean":
        return float(np.mean(numeric))
    if operation == "max_abs":
        return float(np.max(np.abs(numeric)))
    if operation == "range":
        return float(np.max(numeric) - np.min(numeric))
    if operation == "std":
        return float(np.std(numeric, ddof=0))
    raise ValueError(f"Unsupported aggregate operation: {operation}")


def build_stability(
    features: list[str],
    categories: dict[str, str],
    feature_drift: pd.DataFrame,
    associations: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fold_rank = {name: index for index, name in enumerate(FOLD_ORDER)}
    for feature in features:
        assoc = associations.loc[associations["feature"].eq(feature)].copy()
        assoc["_rank"] = assoc["fold"].map(fold_rank)
        assoc = assoc.sort_values("_rank")
        drift = feature_drift.loc[feature_drift["feature"].eq(feature)].copy()
        test_signs = assoc["test_sign"].astype(str).tolist()
        reversals = int(assoc["train_test_sign_reversal"].eq(True).sum())
        rows.append(
            {
                "feature": feature,
                "category": categories.get(feature, "unknown"),
                "training_to_test_sign_reversals": reversals,
                "adjacent_test_fold_sign_transitions": adjacent_sign_transitions(test_signs),
                "available_test_association_folds": int(assoc["test_available"].eq(True).sum()),
                "test_spearman_range": _finite_aggregate(assoc["test_spearman"], "range"),
                "test_spearman_std": _finite_aggregate(assoc["test_spearman"], "std"),
                "mean_ks_statistic": _finite_aggregate(drift["ks_statistic"], "mean"),
                "max_abs_standardized_mean_difference": _finite_aggregate(
                    drift["standardized_mean_difference"], "max_abs"
                ),
                "max_abs_missing_rate_delta": _finite_aggregate(
                    drift["missing_rate_delta"], "max_abs"
                ),
            }
        )
    return pd.DataFrame(rows)


def run() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    folds = config.get("validation", {}).get("folds", [])
    if [str(fold["name"]) for fold in folds] != list(FOLD_ORDER):
        raise ValueError("DIAG-001 requires frozen fold order 2024/2025/2026_ytd")

    frame = pd.read_parquet(DATASET, engine="pyarrow").copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
    frame = frame.sort_values("decision_date").reset_index(drop=True)
    frame = add_opportunity_targets(frame)

    feature_version, features = load_feature_registry(REGISTRY)
    validate_feature_columns(frame, features)
    if feature_version != "v3-features-004-treasury" or len(features) != 53:
        raise ValueError("DIAG-001 requires the frozen 53-feature Treasury registry")
    categories = load_categories(REGISTRY)
    if set(categories) != set(features):
        raise ValueError("DIAG-001 feature category registry mismatch")

    expected_hashes = load_exp006_hashes()
    target_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []

    for fold in folds:
        fold_name = str(fold["name"])
        train = frame.loc[eligible_training_mask(frame, fold["train_end"])].copy()
        test = frame.loc[realized_test_mask(frame, fold)].copy()
        if train.empty or test.empty:
            raise ValueError(f"DIAG-001 fold {fold_name} has empty train/test data")
        actual_hash = sample_hash(test["decision_date"])
        expected_hash = expected_hashes.get(fold_name)
        if actual_hash != expected_hash:
            raise ValueError(
                f"DIAG-001 fold {fold_name} sample hash mismatch: {actual_hash} != {expected_hash}"
            )

        train_target = train[TARGET].astype(int)
        test_target = test[TARGET].astype(int)
        target_rows.append(
            {
                "diagnostic_id": DIAGNOSTIC_ID,
                "fold": fold_name,
                "training_rows": int(len(train)),
                "test_rows": int(len(test)),
                "training_prevalence": float(train_target.mean()),
                "test_prevalence": float(test_target.mean()),
                "prevalence_delta": float(test_target.mean() - train_target.mean()),
                "sample_sha256": actual_hash,
                "sample_hash_matches_exp006": True,
            }
        )

        for feature in features:
            distribution = distribution_metrics(train[feature], test[feature])
            drift_rows.append(
                {
                    "diagnostic_id": DIAGNOSTIC_ID,
                    "fold": fold_name,
                    "feature": feature,
                    "category": categories[feature],
                    **distribution,
                }
            )

            train_assoc = association_metrics(train, feature)
            test_assoc = association_metrics(test, feature)
            reversal = sign_reversal(str(train_assoc["sign"]), str(test_assoc["sign"]))
            train_spearman = float(train_assoc["spearman"])
            test_spearman = float(test_assoc["spearman"])
            association_rows.append(
                {
                    "diagnostic_id": DIAGNOSTIC_ID,
                    "fold": fold_name,
                    "feature": feature,
                    "category": categories[feature],
                    "train_observed_rows": int(train_assoc["observed_rows"]),
                    "test_observed_rows": int(test_assoc["observed_rows"]),
                    "train_available": bool(train_assoc["available"]),
                    "test_available": bool(test_assoc["available"]),
                    "train_spearman": train_spearman,
                    "test_spearman": test_spearman,
                    "train_roc_auc_raw_upward": float(train_assoc["roc_auc_raw_upward"]),
                    "test_roc_auc_raw_upward": float(test_assoc["roc_auc_raw_upward"]),
                    "train_sign": str(train_assoc["sign"]),
                    "test_sign": str(test_assoc["sign"]),
                    "train_test_sign_reversal": reversal,
                    "absolute_spearman_change": (
                        float(abs(test_spearman - train_spearman))
                        if np.isfinite(train_spearman) and np.isfinite(test_spearman)
                        else float("nan")
                    ),
                }
            )

    target_drift = pd.DataFrame(target_rows)
    feature_drift = pd.DataFrame(drift_rows)
    associations = pd.DataFrame(association_rows)
    stability = build_stability(features, categories, feature_drift, associations)

    if len(feature_drift) != 53 * 3 or len(associations) != 53 * 3 or len(stability) != 53:
        raise ValueError("DIAG-001 did not preserve all feature/fold rows")

    features_with_reversal = int(stability["training_to_test_sign_reversals"].gt(0).sum())
    features_with_test_transition = int(stability["adjacent_test_fold_sign_transitions"].gt(0).sum())
    report = {
        "diagnostic_id": DIAGNOSTIC_ID,
        "status": "DIAG_001_COMPLETE",
        "as_of": "2026-08-18",
        "feature_version": feature_version,
        "feature_count": len(features),
        "dataset_sha256": _sha256(DATASET),
        "target": TARGET,
        "folds": list(FOLD_ORDER),
        "sample_hashes_match_exp006": bool(target_drift["sample_hash_matches_exp006"].all()),
        "target_prevalence": {
            row["fold"]: {
                "training": row["training_prevalence"],
                "test": row["test_prevalence"],
                "delta": row["prevalence_delta"],
            }
            for row in target_rows
        },
        "feature_drift_rows": int(len(feature_drift)),
        "association_rows": int(len(associations)),
        "stability_rows": int(len(stability)),
        "features_with_any_training_to_test_sign_reversal": features_with_reversal,
        "features_with_any_adjacent_test_fold_sign_transition": features_with_test_transition,
        "minimum_association_rows": MIN_ASSOCIATION_ROWS,
        "model_fitted": False,
        "champion_selected": False,
        "v3_019_eligible": False,
        "current_sizing_multiplier": 1.0,
        "note": "Diagnostic only. No feature/model selection is permitted from DIAG-001 without a new pre-registered experiment.",
    }

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    target_drift.to_csv(TARGET_DRIFT, index=False)
    feature_drift.to_csv(FEATURE_DRIFT, index=False)
    associations.to_csv(ASSOCIATION, index=False)
    stability.to_csv(STABILITY, index=False)
    SUMMARY.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    run()
