#!/usr/bin/env python3
"""Run ADAPT-001: daily causal long/short relationship consensus with abstention."""

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
from v3.evaluation.stab001_past_only import (
    feature_family,
    score_to_probability,
    select_stable_features,
    stability_raw_score,
)
from v3.models.common import load_feature_registry, validate_feature_columns

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config.json"
DATASET = ROOT / "v3" / "data" / "model_dataset_treasury.parquet"
REGISTRY = ROOT / "v3" / "reports" / "feature_registry_treasury.json"
MANIFEST = ROOT / "v3" / "methodology" / "ADAPT-001" / "manifest.json"
STAB001_EVAL = ROOT / "v3" / "reports" / "stab001_evaluation.json"
STAB001_METRICS = ROOT / "v3" / "reports" / "stab001_metrics.csv"
EVALUATION = ROOT / "v3" / "reports" / "adapt001_evaluation.json"
METRICS = ROOT / "v3" / "reports" / "adapt001_metrics.csv"
DAILY = ROOT / "v3" / "reports" / "adapt001_daily.csv"

METHOD_ID = "ADAPT-001"
SHORT_MEMORY_ROWS = 126
MIN_SHORT_OBSERVED = 100
MIN_SHORT_ABS_SPEARMAN = 0.03
MIN_CONSENSUS_FEATURES = 3
MIN_CONSENSUS_FAMILIES = 2
MIN_CONSENSUS_WEIGHT_SHARE = 0.60
MIN_LEGAL_HISTORY = 400
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


def short_memory_association(short: pd.DataFrame, feature: str) -> tuple[float, int, bool]:
    x = pd.to_numeric(short[feature], errors="coerce")
    y = short["favorable_entry_20d"]
    valid = x.notna() & y.notna()
    observed = int(valid.sum())
    if observed < MIN_SHORT_OBSERVED:
        return float("nan"), observed, False
    xv = x.loc[valid].to_numpy(float)
    yv = y.loc[valid].astype(int).to_numpy()
    if len(np.unique(xv)) < 2 or len(np.unique(yv)) < 2:
        return float("nan"), observed, False
    rho = float(spearmanr(xv, yv).statistic)
    usable = bool(np.isfinite(rho) and abs(rho) >= MIN_SHORT_ABS_SPEARMAN and _sign(rho) != 0)
    return rho, observed, usable


def consensus_features(
    legal_history: pd.DataFrame,
    long_selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float, list[dict[str, Any]]]:
    if len(legal_history) < SHORT_MEMORY_ROWS:
        return [], 0.0, []
    short = legal_history.sort_values("decision_date").iloc[-SHORT_MEMORY_ROWS:].copy()
    total_weight = float(sum(abs(float(item["weight"])) for item in long_selected))
    active: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    active_weight = 0.0

    for item in long_selected:
        feature = str(item["feature"])
        long_sign = int(item["majority_sign"])
        rho, observed, usable = short_memory_association(short, feature)
        short_sign = _sign(rho)
        agrees = bool(usable and short_sign == long_sign)
        if agrees:
            cloned = dict(item)
            cloned["short_spearman"] = rho
            active.append(cloned)
            active_weight += abs(float(item["weight"]))
        diagnostics.append(
            {
                "feature": feature,
                "long_sign": long_sign,
                "long_weight": float(item["weight"]),
                "short_spearman": rho,
                "short_observed_rows": observed,
                "short_usable": usable,
                "short_sign": short_sign,
                "consensus": agrees,
            }
        )

    share = float(active_weight / total_weight) if total_weight > 0 else 0.0
    return active, share, diagnostics


def active_gate(active: list[dict[str, Any]], weight_share: float) -> tuple[bool, int]:
    families = len({feature_family(str(item["feature"])) for item in active})
    passed = bool(
        len(active) >= MIN_CONSENSUS_FEATURES
        and families >= MIN_CONSENSUS_FAMILIES
        and weight_share >= MIN_CONSENSUS_WEIGHT_SHARE
    )
    return passed, families


def rank_current_row(
    legal_history: pd.DataFrame,
    current_row: pd.DataFrame,
    active: list[dict[str, Any]],
) -> float:
    history_scores = stability_raw_score(legal_history, legal_history, active)
    current_score = stability_raw_score(legal_history, current_row, active)
    rank = score_to_probability(history_scores, current_score)
    return float(rank[0])


def _expected_hashes() -> dict[str, str]:
    metrics = pd.read_csv(STAB001_METRICS)
    result: dict[str, str] = {}
    for fold, group in metrics.groupby("fold", sort=False):
        values = group["sample_sha256"].astype(str).unique().tolist()
        if len(values) != 1:
            raise ValueError(f"STAB-001 sample hash ambiguity for {fold}")
        result[str(fold)] = values[0]
    return result


def evaluate_fold(rows: pd.DataFrame, fold_name: str, expected_hash: str) -> dict[str, Any]:
    ordered = rows.sort_values("decision_date").copy()
    total_rows = int(len(ordered))
    active_rows = ordered.loc[ordered["active"]].copy()
    active_count = int(len(active_rows))
    coverage = float(active_count / total_rows) if total_rows else 0.0
    sample = sample_hash(ordered["decision_date"])
    sample_match = bool(sample == expected_hash)
    active_classes = int(active_rows["favorable_entry_20d"].astype(int).nunique()) if active_count else 0

    if active_count >= 2 and active_classes == 2:
        active_auc = float(
            roc_auc_score(
                active_rows["favorable_entry_20d"].astype(int).to_numpy(),
                active_rows["opportunity_rank"].to_numpy(float),
            )
        )
    else:
        active_auc = float("nan")

    all_classes = int(ordered["favorable_entry_20d"].astype(int).nunique())
    all_auc = (
        float(roc_auc_score(ordered["favorable_entry_20d"].astype(int), ordered["opportunity_rank"]))
        if all_classes == 2
        else float("nan")
    )

    if active_count:
        active_prevalence = float(active_rows["favorable_entry_20d"].astype(int).mean())
        q75 = float(active_rows["opportunity_rank"].quantile(0.75))
        q25 = float(active_rows["opportunity_rank"].quantile(0.25))
        top = active_rows.loc[active_rows["opportunity_rank"].ge(q75)]
        bottom = active_rows.loc[active_rows["opportunity_rank"].le(q25)]
        top_prevalence = float(top["favorable_entry_20d"].astype(int).mean()) if len(top) else float("nan")
        bottom_prevalence = float(bottom["favorable_entry_20d"].astype(int).mean()) if len(bottom) else float("nan")
        top_lift = float(top_prevalence - active_prevalence) if np.isfinite(top_prevalence) else float("nan")
        median_share = float(active_rows["consensus_weight_share"].median())
        median_features = float(active_rows["consensus_feature_count"].median())
        median_families = float(active_rows["consensus_family_count"].median())
    else:
        active_prevalence = top_prevalence = bottom_prevalence = top_lift = float("nan")
        median_share = median_features = median_families = float("nan")

    return {
        "method_id": METHOD_ID,
        "fold": fold_name,
        "test_rows": total_rows,
        "sample_sha256": sample,
        "expected_sample_sha256": expected_hash,
        "sample_hash_matches_stab001": sample_match,
        "active_rows": active_count,
        "active_coverage": coverage,
        "active_target_classes": active_classes,
        "active_target_prevalence": active_prevalence,
        "active_roc_auc": active_auc,
        "all_row_roc_auc_with_abstention": all_auc,
        "active_top_quartile_prevalence": top_prevalence,
        "active_bottom_quartile_prevalence": bottom_prevalence,
        "active_top_quartile_lift": top_lift,
        "median_active_consensus_weight_share": median_share,
        "median_active_feature_count": median_features,
        "median_active_family_count": median_families,
    }


def summarize(metrics: pd.DataFrame, stab001_mean_auc: float) -> dict[str, Any]:
    coverage_mean = float(metrics["active_coverage"].mean())
    coverage_min = float(metrics["active_coverage"].min())
    active_support = bool(
        metrics["active_rows"].ge(50).all()
        and metrics["active_target_classes"].eq(2).all()
    )
    mean_auc = float(metrics["active_roc_auc"].mean())
    positive_auc = int(metrics["active_roc_auc"].gt(0.5).sum())
    minimum_auc = float(metrics["active_roc_auc"].min())
    auc_gain = float(mean_auc - stab001_mean_auc)
    positive_lift = int(metrics["active_top_quartile_lift"].gt(0.0).sum())
    mean_lift = float(metrics["active_top_quartile_lift"].mean())
    hashes_match = bool(metrics["sample_hash_matches_stab001"].all())
    viable = bool(
        hashes_match
        and coverage_mean >= 0.30
        and coverage_min >= 0.20
        and active_support
        and mean_auc > 0.57
        and positive_auc >= 2
        and minimum_auc >= 0.48
        and auc_gain > 0.01
        and positive_lift >= 2
        and mean_lift >= 0.05
    )
    return {
        "sample_hashes_match_stab001": hashes_match,
        "mean_active_coverage": coverage_mean,
        "minimum_fold_active_coverage": coverage_min,
        "active_support_pass": active_support,
        "mean_active_roc_auc": mean_auc,
        "positive_active_auc_folds": positive_auc,
        "minimum_fold_active_roc_auc": minimum_auc,
        "frozen_stab001_mean_roc_auc": stab001_mean_auc,
        "mean_active_auc_gain_vs_stab001": auc_gain,
        "positive_top_quartile_lift_folds": positive_lift,
        "mean_active_top_quartile_lift": mean_lift,
        "viability_gate_pass": viable,
    }


def run() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    folds = config.get("validation", {}).get("folds", [])
    if len(folds) != 3:
        raise ValueError("ADAPT-001 requires the frozen three chronological folds")

    frame = pd.read_parquet(DATASET, engine="pyarrow").copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
    frame = frame.sort_values("decision_date").reset_index(drop=True)
    frame = add_opportunity_targets(frame)
    feature_version, features = load_feature_registry(REGISTRY)
    validate_feature_columns(frame, features)
    if feature_version != manifest.get("feature_set_version") or len(features) != int(manifest.get("feature_count", -1)):
        raise ValueError("ADAPT-001 frozen feature contract mismatch")

    expected_hashes = _expected_hashes()
    daily_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for fold in folds:
        fold_name = str(fold["name"])
        test = frame.loc[realized_test_mask(frame, fold)].copy().sort_values("decision_date")
        if test.empty:
            raise ValueError(f"ADAPT-001 fold {fold_name} has no mature test rows")

        for current in test.itertuples(index=False):
            decision_date = pd.Timestamp(current.decision_date).normalize()
            causal_cutoff = decision_date - pd.Timedelta(days=1)
            legal = frame.loc[eligible_training_mask(frame, causal_cutoff)].copy()
            legal = legal.sort_values("decision_date").reset_index(drop=True)
            if len(legal) < MIN_LEGAL_HISTORY:
                long_selected: list[dict[str, Any]] = []
                consensus: list[dict[str, Any]] = []
                share = 0.0
                is_active = False
                family_count = 0
                rank = 0.5
            else:
                long_selected, _ = select_stable_features(legal, features)
                consensus, share, _ = consensus_features(legal, long_selected)
                is_active, family_count = active_gate(consensus, share)
                if is_active:
                    current_frame = pd.DataFrame([current._asdict()])
                    rank = rank_current_row(legal, current_frame, consensus)
                else:
                    rank = 0.5

            daily_rows.append(
                {
                    "method_id": METHOD_ID,
                    "fold": fold_name,
                    "decision_date": decision_date.strftime("%Y-%m-%d"),
                    "causal_history_cutoff": causal_cutoff.strftime("%Y-%m-%d"),
                    "legal_history_rows": int(len(legal)),
                    "long_selected_feature_count": int(len(long_selected)),
                    "consensus_feature_count": int(len(consensus)),
                    "consensus_family_count": int(family_count),
                    "consensus_weight_share": float(share),
                    "active": bool(is_active),
                    "opportunity_rank": float(rank),
                    "favorable_entry_20d": int(current.favorable_entry_20d),
                }
            )

        fold_daily = pd.DataFrame([row for row in daily_rows if row["fold"] == fold_name])
        fold_daily["decision_date"] = pd.to_datetime(fold_daily["decision_date"])
        fold_rows.append(evaluate_fold(fold_daily, fold_name, expected_hashes[fold_name]))

    daily = pd.DataFrame(daily_rows)
    metrics = pd.DataFrame(fold_rows)
    stab001 = json.loads(STAB001_EVAL.read_text(encoding="utf-8"))
    stab001_mean_auc = float(stab001["viability"]["mean_roc_auc"])
    viability = summarize(metrics, stab001_mean_auc)
    decision = (
        "LONG_SHORT_CONSENSUS_WORTH_SELECTIVE_EXP_010"
        if viability["viability_gate_pass"]
        else "DO_NOT_ADVANCE_LONG_SHORT_CONSENSUS_UNDER_ADAPT_001"
    )
    report = {
        "method_id": METHOD_ID,
        "as_of": AS_OF.strftime("%Y-%m-%d"),
        "status": "ADAPT_001_EVALUATION_COMPLETE",
        "feature_version": feature_version,
        "feature_count": len(features),
        "dataset_sha256": _sha256(DATASET),
        "target_source_experiment": "EXP-006",
        "target": "favorable_entry_20d",
        "long_memory_method": "STAB-001",
        "adaptation_contract": {
            "short_memory_rows": SHORT_MEMORY_ROWS,
            "minimum_short_observed_rows": MIN_SHORT_OBSERVED,
            "minimum_short_absolute_spearman": MIN_SHORT_ABS_SPEARMAN,
            "minimum_consensus_features": MIN_CONSENSUS_FEATURES,
            "minimum_consensus_families": MIN_CONSENSUS_FAMILIES,
            "minimum_consensus_weight_share": MIN_CONSENSUS_WEIGHT_SHARE,
            "minimum_legal_history_rows": MIN_LEGAL_HISTORY,
            "abstention_rank": 0.5,
            "probability_claimed": False,
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
        "note": "ADAPT-001 tests causal selective ranking only. It may abstain when long/short relationships disagree and does not claim calibrated probabilities or promotion evidence.",
    }

    EVALUATION.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DAILY, index=False, lineterminator="\n")
    metrics.to_csv(METRICS, index=False, lineterminator="\n")
    EVALUATION.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nADAPT-001 fold metrics:")
    print(metrics.to_csv(index=False, lineterminator="\n"))
    return report


if __name__ == "__main__":
    run()
