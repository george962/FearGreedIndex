#!/usr/bin/env python3
"""Run the pre-registered STAB-001 past-only stability-selection experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from v3.evaluation.exp006_opportunity import (
    AS_OF,
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
MANIFEST = ROOT / "v3" / "methodology" / "STAB-001" / "manifest.json"
EXP006_METRICS = ROOT / "v3" / "reports" / "exp006_metrics.csv"
EVALUATION = ROOT / "v3" / "reports" / "stab001_evaluation.json"
METRICS = ROOT / "v3" / "reports" / "stab001_metrics.csv"
SELECTED = ROOT / "v3" / "reports" / "stab001_selected_features.csv"

METHOD_ID = "STAB-001"
BLOCK_COUNT = 4
MIN_BLOCK_ROWS = 100
MIN_SIGN_BLOCKS = 3
MIN_MEDIAN_ABS_SPEARMAN = 0.05
MIN_SELECTED_FEATURES = 3
MIN_SELECTED_FAMILIES = 2
EPS = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_family(name: str) -> str:
    if name == "fear_greed" or name.startswith("fg_"):
        return "fear_greed"
    if name.startswith("treasury_"):
        return "treasury"
    return "spx_interaction"


def _association(values: pd.Series, target: pd.Series) -> tuple[float, int]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & target.notna()
    count = int(valid.sum())
    if count < MIN_BLOCK_ROWS:
        return float("nan"), count
    x = numeric.loc[valid].to_numpy(float)
    y = target.loc[valid].astype(int).to_numpy()
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan"), count
    rho = float(spearmanr(x, y).statistic)
    if not np.isfinite(rho):
        return float("nan"), count
    return rho, count


def _sign(value: float) -> int:
    if not np.isfinite(value) or abs(value) <= EPS:
        return 0
    return 1 if value > 0 else -1


def split_training_blocks(train: pd.DataFrame) -> list[pd.DataFrame]:
    ordered = train.sort_values("decision_date")
    if len(ordered) < BLOCK_COUNT * MIN_BLOCK_ROWS:
        raise ValueError(
            f"STAB-001 requires at least {BLOCK_COUNT * MIN_BLOCK_ROWS} training rows; got {len(ordered)}"
        )
    indices = np.array_split(np.arange(len(ordered)), BLOCK_COUNT)
    return [ordered.iloc[index].copy() for index in indices]


def select_stable_features(
    train: pd.DataFrame,
    features: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks = split_training_blocks(train)
    target_name = "favorable_entry_20d"
    selected: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for feature in features:
        correlations: list[float] = []
        observed: list[int] = []
        for block in blocks:
            rho, count = _association(block[feature], block[target_name])
            correlations.append(rho)
            observed.append(count)

        signs = [_sign(value) for value in correlations]
        positive = int(sum(sign == 1 for sign in signs))
        negative = int(sum(sign == -1 for sign in signs))
        majority_sign = 1 if positive > negative else (-1 if negative > positive else 0)
        majority_count = max(positive, negative)
        finite_abs = [abs(value) for value in correlations if np.isfinite(value)]
        median_abs = float(np.median(finite_abs)) if finite_abs else float("nan")
        consistency = float(majority_count / BLOCK_COUNT)
        recent_agrees = bool(signs[-1] == majority_sign and majority_sign != 0)
        all_blocks_supported = bool(all(count >= MIN_BLOCK_ROWS for count in observed))
        eligible = bool(
            all_blocks_supported
            and len(finite_abs) == BLOCK_COUNT
            and majority_count >= MIN_SIGN_BLOCKS
            and recent_agrees
            and np.isfinite(median_abs)
            and median_abs >= MIN_MEDIAN_ABS_SPEARMAN
        )
        weight = float(median_abs * consistency) if eligible else 0.0

        row: dict[str, Any] = {
            "feature": feature,
            "family": feature_family(feature),
            "selected": eligible,
            "majority_sign": majority_sign,
            "sign_consistency": consistency,
            "median_abs_spearman": median_abs,
            "weight": weight,
            "recent_block_agrees": recent_agrees,
            "all_blocks_supported": all_blocks_supported,
        }
        for index, (rho, count) in enumerate(zip(correlations, observed), start=1):
            row[f"block_{index}_spearman"] = rho
            row[f"block_{index}_observed_rows"] = count
        diagnostics.append(row)
        if eligible:
            selected.append(row.copy())

    selected.sort(key=lambda item: (-float(item["weight"]), str(item["feature"])))
    diagnostics.sort(key=lambda item: str(item["feature"]))
    return selected, diagnostics


def empirical_percentile(train_values: pd.Series, values: pd.Series) -> np.ndarray:
    reference = pd.to_numeric(train_values, errors="coerce").dropna().to_numpy(float)
    if len(reference) == 0:
        return np.full(len(values), 0.5, dtype=float)
    reference.sort()
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    result = np.full(len(numeric), 0.5, dtype=float)
    valid = np.isfinite(numeric)
    result[valid] = np.searchsorted(reference, numeric[valid], side="right") / len(reference)
    return np.clip(result, 0.0, 1.0)


def stability_raw_score(
    train: pd.DataFrame,
    frame: pd.DataFrame,
    selected: list[dict[str, Any]],
) -> np.ndarray:
    if not selected:
        return np.zeros(len(frame), dtype=float)
    total_weight = float(sum(abs(float(item["weight"])) for item in selected))
    if total_weight <= 0:
        return np.zeros(len(frame), dtype=float)
    score = np.zeros(len(frame), dtype=float)
    for item in selected:
        feature = str(item["feature"])
        direction = int(item["majority_sign"])
        weight = float(item["weight"])
        percentile = empirical_percentile(train[feature], frame[feature])
        score += weight * direction * (percentile - 0.5)
    return score / total_weight


def score_to_probability(train_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    reference = np.asarray(train_scores, dtype=float)
    reference = reference[np.isfinite(reference)]
    if len(reference) == 0:
        raise ValueError("STAB-001 training score distribution is empty")
    reference.sort()
    values = np.asarray(scores, dtype=float)
    probability = np.searchsorted(reference, values, side="right") / len(reference)
    return np.clip(probability.astype(float), 0.0, 1.0)


def _expected_exp006_hashes() -> dict[str, str]:
    metrics = pd.read_csv(EXP006_METRICS)
    required = {"fold", "sample_sha256"}
    if not required.issubset(metrics.columns):
        raise ValueError("STAB-001 cannot recover EXP-006 sample hashes")
    hashes: dict[str, str] = {}
    for fold, group in metrics.groupby("fold", sort=False):
        values = group["sample_sha256"].dropna().astype(str).unique().tolist()
        if len(values) != 1:
            raise ValueError(f"EXP-006 sample hash is ambiguous for {fold}")
        hashes[str(fold)] = values[0]
    return hashes


def summarize(metrics: pd.DataFrame) -> dict[str, Any]:
    support_folds = int(metrics["selected_feature_count"].ge(MIN_SELECTED_FEATURES).sum())
    family_support_folds = int(metrics["selected_family_count"].ge(MIN_SELECTED_FAMILIES).sum())
    mean_auc = float(metrics["roc_auc"].mean())
    positive_auc = int(metrics["roc_auc"].gt(0.5).sum())
    minimum_auc = float(metrics["roc_auc"].min())
    mean_relative_brier = float(metrics["relative_brier_improvement"].mean())
    positive_brier = int(metrics["relative_brier_improvement"].gt(0.0).sum())
    hashes_match = bool(metrics["sample_hash_matches_exp006"].all())
    viable = bool(
        support_folds >= 2
        and family_support_folds >= 2
        and mean_auc > 0.52
        and positive_auc >= 2
        and minimum_auc >= 0.45
        and mean_relative_brier > 0.0
        and positive_brier >= 2
        and hashes_match
    )
    return {
        "selection_support_folds": support_folds,
        "family_support_folds": family_support_folds,
        "mean_roc_auc": mean_auc,
        "positive_auc_folds": positive_auc,
        "minimum_fold_roc_auc": minimum_auc,
        "mean_relative_brier_improvement": mean_relative_brier,
        "positive_relative_brier_folds": positive_brier,
        "sample_hashes_match_exp006": hashes_match,
        "viability_gate_pass": viable,
    }


def run() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    folds = config.get("validation", {}).get("folds", [])
    if len(folds) != 3:
        raise ValueError("STAB-001 requires the frozen three chronological folds")

    frame = pd.read_parquet(DATASET, engine="pyarrow").copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
    frame = frame.sort_values("decision_date").reset_index(drop=True)
    frame = add_opportunity_targets(frame)
    feature_version, features = load_feature_registry(REGISTRY)
    validate_feature_columns(frame, features)
    if feature_version != manifest.get("feature_set_version"):
        raise ValueError("STAB-001 feature registry version does not match manifest")
    if len(features) != int(manifest.get("feature_count", -1)):
        raise ValueError("STAB-001 feature count does not match manifest")

    expected_hashes = _expected_exp006_hashes()
    metric_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for fold in folds:
        fold_name = str(fold["name"])
        train = frame.loc[eligible_training_mask(frame, fold["train_end"])].copy()
        test = frame.loc[realized_test_mask(frame, fold)].copy()
        if test.empty:
            raise ValueError(f"STAB-001 fold {fold_name} has no mature test rows")
        if train["favorable_entry_20d"].astype(int).nunique() != 2:
            raise ValueError(f"STAB-001 fold {fold_name} training sample lacks both classes")
        y_test = test["favorable_entry_20d"].astype(int).to_numpy()
        if len(np.unique(y_test)) != 2:
            raise ValueError(f"STAB-001 fold {fold_name} test sample lacks both classes")

        selected, diagnostics = select_stable_features(train, features)
        for row in diagnostics:
            row["fold"] = fold_name
            selected_rows.append(row)

        base_rate = float(train["favorable_entry_20d"].astype(int).mean())
        if selected:
            train_scores = stability_raw_score(train, train, selected)
            test_scores = stability_raw_score(train, test, selected)
            probability = score_to_probability(train_scores, test_scores)
        else:
            probability = np.full(len(test), base_rate, dtype=float)

        baseline_probability = np.full(len(test), base_rate, dtype=float)
        brier = float(np.mean((probability - y_test) ** 2))
        baseline_brier = float(np.mean((baseline_probability - y_test) ** 2))
        relative_brier = (
            float((baseline_brier - brier) / baseline_brier)
            if baseline_brier > 0
            else float("nan")
        )
        auc = float(roc_auc_score(y_test, probability)) if selected else 0.5
        date_hash = sample_hash(test["decision_date"])
        expected_hash = expected_hashes.get(fold_name)
        if expected_hash is None:
            raise ValueError(f"Missing EXP-006 expected sample hash for {fold_name}")
        family_count = len({str(item["family"]) for item in selected})

        metric_rows.append(
            {
                "method_id": METHOD_ID,
                "fold": fold_name,
                "training_rows": int(len(train)),
                "test_rows": int(len(test)),
                "sample_sha256": date_hash,
                "exp006_sample_sha256": expected_hash,
                "sample_hash_matches_exp006": bool(date_hash == expected_hash),
                "training_prevalence": base_rate,
                "test_prevalence": float(np.mean(y_test)),
                "selected_feature_count": int(len(selected)),
                "selected_family_count": int(family_count),
                "selected_features": "|".join(str(item["feature"]) for item in selected),
                "brier_score": brier,
                "baseline_brier_score": baseline_brier,
                "relative_brier_improvement": relative_brier,
                "roc_auc": auc,
            }
        )

    metrics = pd.DataFrame(metric_rows)
    selected_frame = pd.DataFrame(selected_rows)
    viability = summarize(metrics)
    decision = (
        "STABILITY_SELECTION_WORTH_EXP_010"
        if viability["viability_gate_pass"]
        else "DO_NOT_ADVANCE_STABILITY_SELECTOR_UNDER_STAB_001"
    )
    report = {
        "method_id": METHOD_ID,
        "as_of": AS_OF.strftime("%Y-%m-%d"),
        "status": "STAB_001_EVALUATION_COMPLETE",
        "feature_version": feature_version,
        "feature_count": len(features),
        "dataset_sha256": _sha256(DATASET),
        "target_source_experiment": "EXP-006",
        "target": "favorable_entry_20d",
        "selection_rule": {
            "chronological_blocks": BLOCK_COUNT,
            "minimum_observed_rows_per_block": MIN_BLOCK_ROWS,
            "minimum_majority_sign_blocks": MIN_SIGN_BLOCKS,
            "recent_block_must_agree": True,
            "minimum_median_absolute_spearman": MIN_MEDIAN_ABS_SPEARMAN,
        },
        "viability": viability,
        "method_viability_pass": bool(viability["viability_gate_pass"]),
        "decision": decision,
        "development_evidence_only": True,
        "research_exposed_periods": ["2024", "2025", "2026_ytd"],
        "evid001_outcomes_opened": False,
        "champion_selected": False,
        "v3_019_eligible": False,
        "current_sizing_multiplier": 1.0,
        "note": "STAB-001 may justify EXP-010 methodology only. Its outer folds are already research-exposed after DIAG-001 and cannot serve as final champion promotion evidence.",
    }

    EVALUATION.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS, index=False, lineterminator="\n")
    selected_frame.to_csv(SELECTED, index=False, lineterminator="\n")
    EVALUATION.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nSTAB-001 fold metrics:")
    print(metrics.to_csv(index=False, lineterminator="\n"))
    return report


if __name__ == "__main__":
    run()
