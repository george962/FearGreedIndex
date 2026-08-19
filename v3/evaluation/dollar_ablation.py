#!/usr/bin/env python3
"""V3-015B controlled ablation for the broad-dollar feature family."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.testing import assert_frame_equal

from v3.evaluation.tournament import summarize_experiments
from v3.evaluation.vix_ablation import ABLATION_AS_OF, FULL_INTERFACE_EXPERIMENTS, REQUIRED_TARGET_TYPES, compare_metric_frames, freeze_model_dataset_asof, summarize_lane_improvements
from v3.evaluation.walk_forward import run_common_evaluation
from v3.features.build_dollar_features import DEFAULT_BASE_FEATURES, DEFAULT_OUTPUT as DOLLAR_FEATURES_PATH, DEFAULT_OUTPUT_REGISTRY as DOLLAR_REGISTRY, DEFAULT_REPORT as DOLLAR_FEATURE_REPORT, DEFAULT_SOURCE as DOLLAR_SNAPSHOT, run_build as build_dollar_features
from v3.models.common import DEFAULT_FEATURE_REGISTRY, DEFAULT_MODEL_DATASET

ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = ROOT / "v3" / "data" / "dollar_source.json"
ABLATION_VERSION = "v3-dollar-ablation-001"
FROZEN_BASE_DATASET = ROOT / "v3" / "data" / "model_dataset_dollar_baseline_asof_2026_08_18.parquet"
FROZEN_DOLLAR_DATASET = ROOT / "v3" / "data" / "model_dataset_dollar_asof_2026_08_18.parquet"
BASE_PREDICTIONS = ROOT / "v3" / "reports" / "dollar_ablation_baseline_predictions.parquet"
BASE_METRICS = ROOT / "v3" / "reports" / "dollar_ablation_baseline_metrics.csv"
BASE_SUMMARY = ROOT / "v3" / "reports" / "dollar_ablation_baseline_summary.json"
DOLLAR_PREDICTIONS = ROOT / "v3" / "reports" / "dollar_ablation_candidate_predictions.parquet"
DOLLAR_METRICS = ROOT / "v3" / "reports" / "dollar_ablation_candidate_metrics.csv"
DOLLAR_SUMMARY = ROOT / "v3" / "reports" / "dollar_ablation_candidate_summary.json"
COMPARISON_OUTPUT = ROOT / "v3" / "reports" / "dollar_ablation_comparison.csv"
LANE_OUTPUT = ROOT / "v3" / "reports" / "dollar_ablation_lane_summary.csv"
TOURNAMENT_OUTPUT = ROOT / "v3" / "reports" / "dollar_ablation_tournament.csv"
REPORT_OUTPUT = ROOT / "v3" / "reports" / "dollar_ablation.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_snapshot() -> dict[str, Any]:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if _sha256(DOLLAR_SNAPSHOT) != manifest.get("snapshot_sha256"):
        raise ValueError("Dollar snapshot hash does not match its manifest")
    return manifest


def prepare_frozen_datasets() -> dict[str, Any]:
    manifest = verify_snapshot()
    build_dollar_features(base_features_path=DEFAULT_BASE_FEATURES, source_path=DOLLAR_SNAPSHOT, output_path=DOLLAR_FEATURES_PATH, output_registry_path=DOLLAR_REGISTRY, report_path=DOLLAR_FEATURE_REPORT)
    base_features = pd.read_parquet(DEFAULT_BASE_FEATURES, engine="pyarrow")
    candidate_features = pd.read_parquet(DOLLAR_FEATURES_PATH, engine="pyarrow")
    base_model = pd.read_parquet(DEFAULT_MODEL_DATASET, engine="pyarrow")
    assert_frame_equal(candidate_features[base_features.columns].reset_index(drop=True), base_features.reset_index(drop=True), check_dtype=True)
    label_columns = [c for c in base_model.columns if c not in base_features.columns]
    candidate_model = candidate_features.merge(base_model[["decision_date", *label_columns]], on="decision_date", how="left", validate="one_to_one")
    assert_frame_equal(candidate_model[["decision_date", *label_columns]].reset_index(drop=True), base_model[["decision_date", *label_columns]].reset_index(drop=True), check_dtype=True)
    frozen_base = freeze_model_dataset_asof(base_model, ABLATION_AS_OF)
    frozen_candidate = freeze_model_dataset_asof(candidate_model, ABLATION_AS_OF)
    assert_frame_equal(frozen_candidate[["decision_date", *label_columns]].reset_index(drop=True), frozen_base[["decision_date", *label_columns]].reset_index(drop=True), check_dtype=True)
    FROZEN_BASE_DATASET.parent.mkdir(parents=True, exist_ok=True)
    frozen_base.to_parquet(FROZEN_BASE_DATASET, index=False, engine="pyarrow")
    frozen_candidate.to_parquet(FROZEN_DOLLAR_DATASET, index=False, engine="pyarrow")
    return {"manifest": manifest, "rows": len(frozen_base), "baseline_sha": _sha256(FROZEN_BASE_DATASET), "candidate_sha": _sha256(FROZEN_DOLLAR_DATASET)}


def dollar_family_decision(lanes: pd.DataFrame) -> dict[str, Any]:
    full = lanes.loc[lanes["experiment_id"].isin(FULL_INTERFACE_EXPERIMENTS)].copy()
    missing = sorted(set(FULL_INTERFACE_EXPERIMENTS).difference(full["experiment_id"]))
    if missing:
        raise ValueError(f"Missing full-interface ablation models: {missing}")
    lane_results: dict[str, bool] = {}
    details: dict[str, Any] = {}
    for target_type in REQUIRED_TARGET_TYPES:
        lane = full.loc[full["target_type"] == target_type]
        by_model = {str(row["experiment_id"]): bool(row["robust_improvement"]) for _, row in lane.iterrows()}
        if set(by_model) != set(FULL_INTERFACE_EXPERIMENTS):
            raise ValueError(f"Target lane {target_type} does not contain both full-interface models")
        lane_results[target_type] = all(by_model.values())
        details[target_type] = by_model
    count = int(sum(lane_results.values()))
    retain = count >= 2
    return {
        "retain_dollar": retain,
        "decision": "KEEP_DOLLAR_FOR_LATER_RESEARCH" if retain else "DO_NOT_RETAIN_DOLLAR_FEATURE_FAMILY",
        "robust_lane_count": count,
        "required_robust_lanes": 2,
        "family_lane_robust_in_both_full_models": lane_results,
        "per_model_lane_robustness": details,
        "criterion": "A lane is robust only when its aggregate primary metric improves and at least 2 of 3 chronological folds improve. Dollar is retained only when at least 2 of 3 lanes are robust in both EXP-003 and EXP-004.",
    }


def build_tournament(base: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    base = base.copy(); candidate = candidate.copy()
    base["experiment_id"] = "BASE-" + base["experiment_id"].astype(str)
    base["model_name"] = base["model_name"].astype(str) + " [baseline]"
    candidate["experiment_id"] = "USD-" + candidate["experiment_id"].astype(str)
    candidate["model_name"] = candidate["model_name"].astype(str) + " [+Dollar]"
    return summarize_experiments(pd.concat([base, candidate], ignore_index=True, sort=False))


def run_dollar_ablation(report_output: Path = REPORT_OUTPUT) -> dict[str, Any]:
    prepared = prepare_frozen_datasets()
    base_report = run_common_evaluation(dataset_path=FROZEN_BASE_DATASET, registry_path=DEFAULT_FEATURE_REGISTRY, predictions_output=BASE_PREDICTIONS, metrics_output=BASE_METRICS, summary_output=BASE_SUMMARY)
    candidate_report = run_common_evaluation(dataset_path=FROZEN_DOLLAR_DATASET, registry_path=DOLLAR_REGISTRY, predictions_output=DOLLAR_PREDICTIONS, metrics_output=DOLLAR_METRICS, summary_output=DOLLAR_SUMMARY)
    base_metrics = pd.read_csv(BASE_METRICS); candidate_metrics = pd.read_csv(DOLLAR_METRICS)
    comparison = compare_metric_frames(base_metrics, candidate_metrics)
    lanes = summarize_lane_improvements(comparison)
    decision = dollar_family_decision(lanes)
    tournament = build_tournament(base_metrics, candidate_metrics)
    comparison.to_csv(COMPARISON_OUTPUT, index=False); lanes.to_csv(LANE_OUTPUT, index=False); tournament.to_csv(TOURNAMENT_OUTPUT, index=False)
    full = tournament.loc[tournament["full_candidate"].astype(bool)].sort_values(["overall_full_candidate_rank", "experiment_id"], na_position="last")
    best = None if full.empty else str(full.iloc[0]["experiment_id"])
    promotion = tournament.loc[tournament["promotion_ready"].astype(bool), "experiment_id"].astype(str).tolist()
    report = {
        "status": "DOLLAR_ABLATION_COMPLETE", "ablation_version": ABLATION_VERSION,
        "ablation_as_of": ABLATION_AS_OF.date().isoformat(),
        "baseline_feature_version": base_report.get("feature_set_version"), "candidate_feature_version": candidate_report.get("feature_set_version"),
        "rows": int(prepared["rows"]), "baseline_frozen_dataset_sha256": prepared["baseline_sha"], "candidate_frozen_dataset_sha256": prepared["candidate_sha"],
        "snapshot_sha256": prepared["manifest"].get("snapshot_sha256"), "source_end": prepared["manifest"].get("end"),
        "comparison_cells": int(len(comparison)), "sample_hashes_match": True,
        "feature_family_decision": decision, "best_ranked_full_candidate_in_ablation_tournament": best,
        "promotion_ready_experiments_from_absolute_gates": promotion,
        "champion_selected": False, "champion_selection_status": "DEFERRED_TO_V3_018_V3_019",
        "trading_policy_status": "UNCHANGED_AND_DEFERRED_TO_V3_016",
        "next": "V3-015 credit-spread source gate / V3-016 decision policy after research prerequisites",
    }
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT); args = parser.parse_args()
    report = run_dollar_ablation(args.report_output)
    return 0 if report["status"] == "DOLLAR_ABLATION_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
