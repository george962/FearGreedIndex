#!/usr/bin/env python3
"""Run STAB-003 long-memory/short-memory consensus with abstention."""

from __future__ import annotations

import hashlib
import json
import math
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
from v3.evaluation.stab001_past_only import (
    empirical_percentile,
    feature_family,
    select_stable_features,
)
from v3.models.common import load_feature_registry, validate_feature_columns

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config.json"
DATASET = ROOT / "v3" / "data" / "model_dataset_treasury.parquet"
REGISTRY = ROOT / "v3" / "reports" / "feature_registry_treasury.json"
MANIFEST = ROOT / "v3" / "methodology" / "STAB-003" / "manifest.json"
EXP006_METRICS = ROOT / "v3" / "reports" / "exp006_metrics.csv"
EVID001_CHECKPOINTS = ROOT / "v3" / "evidence" / "forward_checkpoints.json"
EVALUATION = ROOT / "v3" / "reports" / "stab003_evaluation.json"
METRICS = ROOT / "v3" / "reports" / "stab003_metrics.csv"
FEATURES_OUT = ROOT / "v3" / "reports" / "stab003_consensus_features.csv"

METHOD_ID = "STAB-003"
TARGET = "favorable_entry_20d"
SHORT_MAX_ROWS = 504
SHORT_MIN_ROWS = 360
SHORT_BLOCKS = 3
MIN_BLOCK_ROWS = 100
MIN_ABS_SPEARMAN = 0.05
MIN_FEATURES = 3
MIN_FAMILIES = 2
LOW_Q = 0.20
HIGH_Q = 0.80
EPS = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sign(value: float) -> int:
    if not np.isfinite(value) or abs(value) <= EPS:
        return 0
    return 1 if value > 0 else -1


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
    return (rho if np.isfinite(rho) else float("nan")), count


def select_short_memory_features(
    train: pd.DataFrame,
    features: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = train.sort_values("decision_date").tail(SHORT_MAX_ROWS).copy()
    if len(ordered) < SHORT_MIN_ROWS:
        return [], []
    blocks = [
        ordered.iloc[index].copy()
        for index in np.array_split(np.arange(len(ordered)), SHORT_BLOCKS)
    ]

    selected: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for feature in features:
        correlations: list[float] = []
        counts: list[int] = []
        for block in blocks:
            rho, count = _association(block[feature], block[TARGET])
            correlations.append(rho)
            counts.append(count)
        signs = [_sign(value) for value in correlations]
        same_sign = bool(signs[0] != 0 and signs.count(signs[0]) == SHORT_BLOCKS)
        finite_abs = [abs(value) for value in correlations if np.isfinite(value)]
        median_abs = (
            float(np.median(finite_abs))
            if len(finite_abs) == SHORT_BLOCKS
            else float("nan")
        )
        supported = bool(all(count >= MIN_BLOCK_ROWS for count in counts))
        eligible = bool(
            supported
            and same_sign
            and np.isfinite(median_abs)
            and median_abs >= MIN_ABS_SPEARMAN
        )
        row: dict[str, Any] = {
            "feature": feature,
            "family": feature_family(feature),
            "selected": eligible,
            "direction": signs[0] if same_sign else 0,
            "median_abs_spearman": median_abs,
            "weight": median_abs if eligible else 0.0,
            "training_rows_considered": int(len(ordered)),
        }
        for index, (rho, count) in enumerate(zip(correlations, counts), start=1):
            row[f"block_{index}_spearman"] = rho
            row[f"block_{index}_observed_rows"] = count
        diagnostics.append(row)
        if eligible:
            selected.append(row.copy())

    selected.sort(key=lambda item: (-float(item["weight"]), str(item["feature"])))
    diagnostics.sort(key=lambda item: str(item["feature"]))
    return selected, diagnostics


def build_consensus(
    long_selected: list[dict[str, Any]],
    short_selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    long_map = {str(item["feature"]): item for item in long_selected}
    short_map = {str(item["feature"]): item for item in short_selected}
    consensus: list[dict[str, Any]] = []
    for feature in sorted(set(long_map) & set(short_map)):
        long_item = long_map[feature]
        short_item = short_map[feature]
        long_direction = int(long_item["majority_sign"])
        short_direction = int(short_item["direction"])
        if long_direction == 0 or long_direction != short_direction:
            continue
        long_weight = float(long_item["weight"])
        short_weight = float(short_item["weight"])
        if long_weight <= 0 or short_weight <= 0:
            continue
        consensus.append(
            {
                "feature": feature,
                "family": feature_family(feature),
                "direction": long_direction,
                "long_weight": long_weight,
                "short_weight": short_weight,
                "consensus_weight": float(math.sqrt(long_weight * short_weight)),
            }
        )
    consensus.sort(
        key=lambda item: (-float(item["consensus_weight"]), str(item["feature"]))
    )
    return consensus


def consensus_score(
    train: pd.DataFrame,
    frame: pd.DataFrame,
    consensus: list[dict[str, Any]],
) -> np.ndarray:
    if not consensus:
        return np.zeros(len(frame), dtype=float)
    total_weight = float(
        sum(abs(float(item["consensus_weight"])) for item in consensus)
    )
    if total_weight <= 0:
        return np.zeros(len(frame), dtype=float)
    score = np.zeros(len(frame), dtype=float)
    for item in consensus:
        feature = str(item["feature"])
        percentile = empirical_percentile(train[feature], frame[feature])
        score += (
            float(item["consensus_weight"])
            * int(item["direction"])
            * (percentile - 0.5)
        )
    return score / total_weight


def training_abstention_thresholds(scores: np.ndarray) -> tuple[float, float]:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("STAB-003 legal training score distribution is empty")
    low = float(np.quantile(values, LOW_Q))
    high = float(np.quantile(values, HIGH_Q))
    if not low < high:
        raise ValueError("STAB-003 training abstention thresholds collapsed")
    return low, high


def call_states(scores: np.ndarray, low: float, high: float) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    states = np.full(len(values), "ABSTAIN", dtype=object)
    states[values <= low] = "STRONG_UNFAVORABLE"
    states[values >= high] = "STRONG_FAVORABLE"
    return states


def _expected_exp006_hashes() -> dict[str, str]:
    metrics = pd.read_csv(EXP006_METRICS)
    hashes: dict[str, str] = {}
    for fold, group in metrics.groupby("fold", sort=False):
        values = group["sample_sha256"].dropna().astype(str).unique().tolist()
        if len(values) != 1:
            raise ValueError(f"EXP-006 sample hash is ambiguous for {fold}")
        hashes[str(fold)] = values[0]
    return hashes


def evid001_outcomes_are_sealed() -> bool:
    payload = json.loads(EVID001_CHECKPOINTS.read_text(encoding="utf-8"))
    return bool(len(payload.get("checkpoints", [])) == 0)


def summarize(metrics: pd.DataFrame) -> dict[str, Any]:
    support = metrics["support_pass"].astype(bool)
    supported = metrics.loc[support].copy()
    support_folds = int(support.sum())
    mean_auc = float(metrics["roc_auc"].mean())
    auc_above = int(metrics["roc_auc"].gt(0.52).sum())
    minimum_auc = float(metrics["roc_auc"].min())
    favorable_lift_folds = (
        int(supported["favorable_lift"].gt(0.05).sum())
        if not supported.empty
        else 0
    )
    unfavorable_separation_folds = (
        int(supported["unfavorable_separation"].gt(0.05).sum())
        if not supported.empty
        else 0
    )
    coverage_pass = bool(
        not supported.empty
        and supported["non_abstain_coverage"]
        .between(0.20, 0.55, inclusive="both")
        .all()
    )
    hashes_match = bool(metrics["sample_hash_matches_exp006"].all())
    evid_sealed = evid001_outcomes_are_sealed()
    viable = bool(
        support_folds >= 2
        and mean_auc > 0.55
        and auc_above >= 2
        and minimum_auc >= 0.45
        and favorable_lift_folds >= 2
        and unfavorable_separation_folds >= 2
        and coverage_pass
        and hashes_match
        and evid_sealed
    )
    return {
        "support_folds": support_folds,
        "mean_roc_auc": mean_auc,
        "roc_auc_above_0_52_folds": auc_above,
        "minimum_fold_roc_auc": minimum_auc,
        "favorable_lift_above_0_05_folds": favorable_lift_folds,
        "unfavorable_separation_above_0_05_folds": unfavorable_separation_folds,
        "supported_fold_coverage_gate_pass": coverage_pass,
        "sample_hashes_match_exp006": hashes_match,
        "evid001_outcomes_sealed": evid_sealed,
        "viability_gate_pass": viable,
    }


def run() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"pre_registered", "complete_reject"}:
        raise ValueError("STAB-003 manifest status is invalid for evaluation/reproduction")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    folds = config.get("validation", {}).get("folds", [])
    if len(folds) != 3:
        raise ValueError("STAB-003 requires the frozen three chronological folds")
    if not evid001_outcomes_are_sealed():
        raise ValueError("STAB-003 refuses to run after EVID-001 outcomes are opened")

    frame = pd.read_parquet(DATASET, engine="pyarrow").copy()
    frame["decision_date"] = pd.to_datetime(
        frame["decision_date"], errors="raise"
    ).dt.normalize()
    frame = frame.sort_values("decision_date").reset_index(drop=True)
    frame = add_opportunity_targets(frame)
    feature_version, features = load_feature_registry(REGISTRY)
    validate_feature_columns(frame, features)
    if feature_version != manifest.get("feature_set_version"):
        raise ValueError("STAB-003 feature registry version does not match manifest")
    if len(features) != int(manifest.get("feature_count", -1)):
        raise ValueError("STAB-003 feature count does not match manifest")

    expected_hashes = _expected_exp006_hashes()
    metric_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for fold in folds:
        fold_name = str(fold["name"])
        train = frame.loc[eligible_training_mask(frame, fold["train_end"])].copy()
        test = frame.loc[realized_test_mask(frame, fold)].copy()
        if test.empty:
            raise ValueError(f"STAB-003 fold {fold_name} has no mature test rows")
        y_test = test[TARGET].astype(int).to_numpy()
        if train[TARGET].astype(int).nunique() != 2 or len(np.unique(y_test)) != 2:
            raise ValueError(f"STAB-003 fold {fold_name} lacks both target classes")

        long_selected, _ = select_stable_features(train, features)
        short_selected, _ = select_short_memory_features(train, features)
        consensus = build_consensus(long_selected, short_selected)
        families = {str(item["family"]) for item in consensus}
        support_pass = bool(
            len(consensus) >= MIN_FEATURES and len(families) >= MIN_FAMILIES
        )

        for item in consensus:
            row = dict(item)
            row["fold"] = fold_name
            feature_rows.append(row)

        base_rate = float(train[TARGET].astype(int).mean())
        if support_pass:
            train_scores = consensus_score(train, train, consensus)
            test_scores = consensus_score(train, test, consensus)
            low, high = training_abstention_thresholds(train_scores)
            states = call_states(test_scores, low, high)
            auc = float(roc_auc_score(y_test, test_scores))
            favorable_mask = states == "STRONG_FAVORABLE"
            unfavorable_mask = states == "STRONG_UNFAVORABLE"
            favorable_prev = (
                float(np.mean(y_test[favorable_mask]))
                if favorable_mask.any()
                else float("nan")
            )
            unfavorable_prev = (
                float(np.mean(y_test[unfavorable_mask]))
                if unfavorable_mask.any()
                else float("nan")
            )
            coverage = float(np.mean(states != "ABSTAIN"))
            favorable_lift = (
                favorable_prev - base_rate
                if np.isfinite(favorable_prev)
                else float("nan")
            )
            unfavorable_separation = (
                base_rate - unfavorable_prev
                if np.isfinite(unfavorable_prev)
                else float("nan")
            )
            favorable_count = int(favorable_mask.sum())
            unfavorable_count = int(unfavorable_mask.sum())
        else:
            low = float("nan")
            high = float("nan")
            auc = 0.5
            favorable_prev = float("nan")
            unfavorable_prev = float("nan")
            favorable_lift = float("nan")
            unfavorable_separation = float("nan")
            coverage = 0.0
            favorable_count = 0
            unfavorable_count = 0

        date_hash = sample_hash(test["decision_date"])
        expected_hash = expected_hashes.get(fold_name)
        if expected_hash is None:
            raise ValueError(f"Missing EXP-006 sample hash for {fold_name}")

        metric_rows.append(
            {
                "method_id": METHOD_ID,
                "fold": fold_name,
                "training_rows": int(len(train)),
                "short_memory_rows": int(min(len(train), SHORT_MAX_ROWS)),
                "test_rows": int(len(test)),
                "sample_sha256": date_hash,
                "exp006_sample_sha256": expected_hash,
                "sample_hash_matches_exp006": bool(date_hash == expected_hash),
                "training_prevalence": base_rate,
                "test_prevalence": float(np.mean(y_test)),
                "long_selected_feature_count": int(len(long_selected)),
                "short_selected_feature_count": int(len(short_selected)),
                "consensus_feature_count": int(len(consensus)),
                "consensus_family_count": int(len(families)),
                "support_pass": support_pass,
                "training_score_q20": low,
                "training_score_q80": high,
                "roc_auc": auc,
                "strong_favorable_count": favorable_count,
                "strong_unfavorable_count": unfavorable_count,
                "strong_favorable_prevalence": favorable_prev,
                "strong_unfavorable_prevalence": unfavorable_prev,
                "favorable_lift": favorable_lift,
                "unfavorable_separation": unfavorable_separation,
                "non_abstain_coverage": coverage,
                "consensus_features": "|".join(
                    str(item["feature"]) for item in consensus
                ),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    features_frame = pd.DataFrame(feature_rows)
    viability = summarize(metrics)
    decision = (
        "CONSENSUS_ABSTENTION_WORTH_EXP_010"
        if viability["viability_gate_pass"]
        else "DO_NOT_ADVANCE_CONSENSUS_ABSTENTION_UNDER_STAB_003"
    )
    report = {
        "method_id": METHOD_ID,
        "as_of": AS_OF.strftime("%Y-%m-%d"),
        "status": "STAB_003_EVALUATION_COMPLETE",
        "feature_version": feature_version,
        "feature_count": len(features),
        "dataset_sha256": _sha256(DATASET),
        "target": TARGET,
        "target_source_experiment": "EXP-006",
        "viability": viability,
        "method_viability_pass": bool(viability["viability_gate_pass"]),
        "decision": decision,
        "development_evidence_only": True,
        "research_exposed_periods": ["2024", "2025", "2026_ytd"],
        "evid001_outcomes_opened": False,
        "champion_selected": False,
        "v3_019_eligible": False,
        "current_sizing_multiplier": 1.0,
        "note": "STAB-003 tests causal relationship consensus and abstention only. It cannot promote a model or consume untouched forward outcomes.",
    }

    METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS, index=False, lineterminator="\n")
    features_frame.to_csv(FEATURES_OUT, index=False, lineterminator="\n")
    EVALUATION.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
