#!/usr/bin/env python3
"""Run STAB-004 causal rolling score normalization with redundancy control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from v3.evaluation.exp006_opportunity import (
    AS_OF,
    add_opportunity_targets,
    eligible_training_mask,
    realized_test_mask,
    sample_hash,
)
from v3.evaluation.stab001_past_only import select_stable_features
from v3.evaluation.stab003_consensus_abstention import (
    build_consensus,
    consensus_score,
    evid001_outcomes_are_sealed,
    select_short_memory_features,
)
from v3.models.common import load_feature_registry, validate_feature_columns

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config.json"
DATASET = ROOT / "v3" / "data" / "model_dataset_treasury.parquet"
REGISTRY = ROOT / "v3" / "reports" / "feature_registry_treasury.json"
MANIFEST = ROOT / "v3" / "methodology" / "STAB-004" / "manifest.json"
EXP006_METRICS = ROOT / "v3" / "reports" / "exp006_metrics.csv"
EVALUATION = ROOT / "v3" / "reports" / "stab004_evaluation.json"
METRICS = ROOT / "v3" / "reports" / "stab004_metrics.csv"
CLUSTERS = ROOT / "v3" / "reports" / "stab004_redundancy_clusters.csv"
SCORES = ROOT / "v3" / "reports" / "stab004_scores.csv"

METHOD_ID = "STAB-004"
TARGET = "favorable_entry_20d"
REDUNDANCY_THRESHOLD = 0.90
MIN_REPRESENTATIVES = 3
REFERENCE_WINDOW = 252
MIN_REFERENCE = 126
LOW_PERCENTILE = 0.20
HIGH_PERCENTILE = 0.80


def _expected_exp006_hashes() -> dict[str, str]:
    metrics = pd.read_csv(EXP006_METRICS)
    hashes: dict[str, str] = {}
    for fold, group in metrics.groupby("fold", sort=False):
        values = group["sample_sha256"].dropna().astype(str).unique().tolist()
        if len(values) != 1:
            raise ValueError(f"EXP-006 sample hash is ambiguous for {fold}")
        hashes[str(fold)] = values[0]
    return hashes


def _pairwise_abs_spearman(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if not features:
        return pd.DataFrame(index=[], columns=[], dtype=float)
    numeric = train.loc[:, features].apply(pd.to_numeric, errors="coerce")
    return numeric.corr(method="spearman", min_periods=20).abs()


def redundancy_clusters(
    train: pd.DataFrame,
    consensus: list[dict[str, Any]],
    threshold: float = REDUNDANCY_THRESHOLD,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collapse highly dependent consensus features using training-only Spearman graph components."""
    items = {str(item["feature"]): dict(item) for item in consensus}
    features = sorted(items)
    if not features:
        return [], []

    corr = _pairwise_abs_spearman(train, features)
    adjacency: dict[str, set[str]] = {feature: set() for feature in features}
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and float(value) >= threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)

    components: list[list[str]] = []
    seen: set[str] = set()
    for seed in features:
        if seed in seen:
            continue
        stack = [seed]
        component: list[str] = []
        seen.add(seed)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))

    representatives: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    for cluster_number, members in enumerate(components, start=1):
        representative_name = sorted(
            members,
            key=lambda feature: (
                -float(items[feature]["consensus_weight"]),
                feature,
            ),
        )[0]
        representative = dict(items[representative_name])
        representative["cluster_id"] = cluster_number
        representative["cluster_size"] = len(members)
        representatives.append(representative)

        for member in members:
            cluster_rows.append(
                {
                    "cluster_id": cluster_number,
                    "member_feature": member,
                    "representative_feature": representative_name,
                    "is_representative": member == representative_name,
                    "member_consensus_weight": float(items[member]["consensus_weight"]),
                    "representative_consensus_weight": float(
                        items[representative_name]["consensus_weight"]
                    ),
                }
            )

    representatives.sort(
        key=lambda item: (-float(item["consensus_weight"]), str(item["feature"]))
    )
    cluster_rows.sort(key=lambda row: (int(row["cluster_id"]), str(row["member_feature"])))
    return representatives, cluster_rows


def rolling_score_percentiles(
    dates: pd.Series,
    scores: np.ndarray,
    target_dates: set[pd.Timestamp],
    reference_window: int = REFERENCE_WINDOW,
    minimum_reference: int = MIN_REFERENCE,
) -> pd.DataFrame:
    """Rank each requested score only against strictly prior causal score history."""
    date_values = pd.to_datetime(dates, errors="raise").dt.normalize().reset_index(drop=True)
    score_values = np.asarray(scores, dtype=float)
    if len(date_values) != len(score_values):
        raise ValueError("STAB-004 date/score length mismatch")

    rows: list[dict[str, Any]] = []
    for index, current_date in enumerate(date_values):
        if current_date not in target_dates:
            continue
        start = max(0, index - reference_window)
        prior = score_values[start:index]
        prior = prior[np.isfinite(prior)]
        current = float(score_values[index])
        if len(prior) < minimum_reference or not np.isfinite(current):
            percentile = float("nan")
            state = "ABSTAIN"
        else:
            percentile = float(np.searchsorted(np.sort(prior), current, side="right") / len(prior))
            if percentile <= LOW_PERCENTILE:
                state = "STRONG_UNFAVORABLE"
            elif percentile >= HIGH_PERCENTILE:
                state = "STRONG_FAVORABLE"
            else:
                state = "ABSTAIN"
        rows.append(
            {
                "decision_date": current_date,
                "raw_score": current,
                "rolling_percentile": percentile,
                "reference_count": int(len(prior)),
                "call_state": state,
            }
        )
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> dict[str, Any]:
    support = metrics["support_pass"].astype(bool)
    supported = metrics.loc[support].copy()
    support_folds = int(support.sum())
    mean_auc = float(metrics["roc_auc"].mean())
    auc_above = int(metrics["roc_auc"].gt(0.52).sum())
    minimum_auc = float(metrics["roc_auc"].min())
    favorable_folds = int(supported["favorable_enrichment"].gt(0.05).sum()) if not supported.empty else 0
    unfavorable_folds = int(supported["unfavorable_depletion"].gt(0.05).sum()) if not supported.empty else 0
    call_count_pass = bool(
        not supported.empty
        and supported["strong_favorable_count"].ge(15).all()
        and supported["strong_unfavorable_count"].ge(15).all()
    )
    coverage_pass = bool(
        not supported.empty
        and supported["non_abstain_coverage"].between(0.25, 0.55, inclusive="both").all()
    )
    hashes_match = bool(metrics["sample_hash_matches_exp006"].all())
    evid_sealed = evid001_outcomes_are_sealed()
    viable = bool(
        support_folds >= 2
        and mean_auc > 0.55
        and auc_above >= 2
        and minimum_auc >= 0.45
        and favorable_folds >= 2
        and unfavorable_folds >= 2
        and call_count_pass
        and coverage_pass
        and hashes_match
        and evid_sealed
    )
    return {
        "support_folds": support_folds,
        "mean_roc_auc": mean_auc,
        "roc_auc_above_0_52_folds": auc_above,
        "minimum_fold_roc_auc": minimum_auc,
        "favorable_enrichment_above_0_05_folds": favorable_folds,
        "unfavorable_depletion_above_0_05_folds": unfavorable_folds,
        "supported_fold_call_count_gate_pass": call_count_pass,
        "supported_fold_coverage_gate_pass": coverage_pass,
        "sample_hashes_match_exp006": hashes_match,
        "evid001_outcomes_sealed": evid_sealed,
        "viability_gate_pass": viable,
    }


def run() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"pre_registered", "complete_pass", "complete_reject"}:
        raise ValueError("STAB-004 manifest status is invalid for evaluation/reproduction")
    if manifest.get("pre_registered_issue") != 72:
        raise ValueError("STAB-004 pre-registration issue drift")
    if not evid001_outcomes_are_sealed():
        raise ValueError("STAB-004 refuses to run after EVID-001 outcomes are opened")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    folds = config.get("validation", {}).get("folds", [])
    if len(folds) != 3:
        raise ValueError("STAB-004 requires the frozen three chronological folds")

    frame = pd.read_parquet(DATASET, engine="pyarrow").copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
    frame = frame.sort_values("decision_date").reset_index(drop=True)
    frame = frame.loc[frame["decision_date"] <= pd.Timestamp(AS_OF)].copy()
    frame = add_opportunity_targets(frame)

    feature_version, features = load_feature_registry(REGISTRY)
    validate_feature_columns(frame, features)
    if feature_version != manifest.get("feature_set_version"):
        raise ValueError("STAB-004 feature registry version does not match manifest")
    if len(features) != int(manifest.get("feature_count", -1)):
        raise ValueError("STAB-004 feature count does not match manifest")

    expected_hashes = _expected_exp006_hashes()
    metric_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []

    for fold in folds:
        fold_name = str(fold["name"])
        train = frame.loc[eligible_training_mask(frame, fold["train_end"])].copy()
        test = frame.loc[realized_test_mask(frame, fold)].copy()
        if test.empty:
            raise ValueError(f"STAB-004 fold {fold_name} has no mature test rows")
        y_test = test[TARGET].astype(int).to_numpy()
        if train[TARGET].astype(int).nunique() != 2 or len(np.unique(y_test)) != 2:
            raise ValueError(f"STAB-004 fold {fold_name} lacks both target classes")

        long_selected, _ = select_stable_features(train, features)
        short_selected, _ = select_short_memory_features(train, features)
        consensus = build_consensus(long_selected, short_selected)
        representatives, fold_clusters = redundancy_clusters(train, consensus)
        support_pass = len(representatives) >= MIN_REPRESENTATIVES

        for row in fold_clusters:
            enriched = dict(row)
            enriched["fold"] = fold_name
            cluster_rows.append(enriched)

        full_test_prevalence = float(np.mean(y_test))
        if support_pass:
            scoring_frame = frame.loc[
                frame["decision_date"] <= test["decision_date"].max()
            ].copy()
            raw_scores = consensus_score(train, scoring_frame, representatives)
            target_dates = set(test["decision_date"].tolist())
            rolling = rolling_score_percentiles(
                scoring_frame["decision_date"], raw_scores, target_dates
            )
            rolling = test[["decision_date", TARGET]].merge(
                rolling, on="decision_date", how="left", validate="one_to_one"
            )
            if rolling["call_state"].isna().any():
                raise ValueError(f"STAB-004 fold {fold_name} missing rolling call state")

            test_scores = rolling["raw_score"].to_numpy(float)
            auc = float(roc_auc_score(y_test, test_scores))
            states = rolling["call_state"].astype(str).to_numpy()
            favorable_mask = states == "STRONG_FAVORABLE"
            unfavorable_mask = states == "STRONG_UNFAVORABLE"
            favorable_count = int(favorable_mask.sum())
            unfavorable_count = int(unfavorable_mask.sum())
            favorable_prev = (
                float(np.mean(y_test[favorable_mask])) if favorable_count else float("nan")
            )
            unfavorable_prev = (
                float(np.mean(y_test[unfavorable_mask])) if unfavorable_count else float("nan")
            )
            favorable_enrichment = (
                favorable_prev - full_test_prevalence
                if np.isfinite(favorable_prev)
                else float("nan")
            )
            unfavorable_depletion = (
                full_test_prevalence - unfavorable_prev
                if np.isfinite(unfavorable_prev)
                else float("nan")
            )
            coverage = float(np.mean(states != "ABSTAIN"))

            for row in rolling.to_dict(orient="records"):
                row["fold"] = fold_name
                score_rows.append(row)
        else:
            auc = 0.5
            favorable_count = 0
            unfavorable_count = 0
            favorable_prev = float("nan")
            unfavorable_prev = float("nan")
            favorable_enrichment = float("nan")
            unfavorable_depletion = float("nan")
            coverage = 0.0

        date_hash = sample_hash(test["decision_date"])
        expected_hash = expected_hashes.get(fold_name)
        if expected_hash is None:
            raise ValueError(f"Missing EXP-006 sample hash for {fold_name}")

        metric_rows.append(
            {
                "method_id": METHOD_ID,
                "fold": fold_name,
                "training_rows": int(len(train)),
                "test_rows": int(len(test)),
                "consensus_feature_count": int(len(consensus)),
                "representative_cluster_count": int(len(representatives)),
                "support_pass": bool(support_pass),
                "roc_auc": auc,
                "full_test_prevalence": full_test_prevalence,
                "strong_favorable_count": favorable_count,
                "strong_favorable_prevalence": favorable_prev,
                "strong_unfavorable_count": unfavorable_count,
                "strong_unfavorable_prevalence": unfavorable_prev,
                "favorable_enrichment": favorable_enrichment,
                "unfavorable_depletion": unfavorable_depletion,
                "non_abstain_coverage": coverage,
                "sample_sha256": date_hash,
                "expected_exp006_sample_sha256": expected_hash,
                "sample_hash_matches_exp006": date_hash == expected_hash,
            }
        )

    metrics = pd.DataFrame(metric_rows)
    clusters = pd.DataFrame(cluster_rows)
    scores = pd.DataFrame(score_rows)
    viability = summarize(metrics)
    passed = bool(viability["viability_gate_pass"])
    decision = (
        "ADVANCE_TO_PRE_REGISTER_EXP_010_FROM_STAB_004"
        if passed
        else "DO_NOT_ADVANCE_CAUSAL_ROLLING_NORMALIZATION_UNDER_STAB_004"
    )

    METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS, index=False)
    clusters.to_csv(CLUSTERS, index=False)
    scores.to_csv(SCORES, index=False)

    evaluation = {
        "method_id": METHOD_ID,
        "as_of": str(AS_OF),
        "status": "complete_pass" if passed else "complete_reject",
        "feature_version": feature_version,
        "feature_count": len(features),
        "target": TARGET,
        "target_source_experiment": "EXP-006",
        "relationship_source_method": "STAB-003",
        "method_viability_pass": passed,
        "decision": decision,
        "development_evidence_only": True,
        "research_exposed_periods": ["2024", "2025", "2026_ytd"],
        "evid001_outcomes_opened": False,
        "champion_selected": False,
        "v3_019_eligible": False,
        "current_sizing_multiplier": 1.0,
        "viability": viability,
    }
    EVALUATION.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evaluation, indent=2))
    return evaluation


if __name__ == "__main__":
    run()
