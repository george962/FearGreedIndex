#!/usr/bin/env python3
"""V3-015C controlled ablation for the combined retained feature families."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.testing import assert_frame_equal

from v3.ci.run_dollar_feature_stage import ensure_snapshot as ensure_dollar_snapshot
from v3.ci.run_treasury_feature_stage import ensure_snapshot as ensure_treasury_snapshot
from v3.evaluation.tournament import summarize_experiments
from v3.evaluation.vix_ablation import ABLATION_AS_OF, FULL_INTERFACE_EXPERIMENTS, REQUIRED_TARGET_TYPES, compare_metric_frames, freeze_model_dataset_asof, summarize_lane_improvements
from v3.evaluation.walk_forward import run_common_evaluation
from v3.features.build_retained_features import DEFAULT_BASE_FEATURES, DEFAULT_OUTPUT as COMBINED_FEATURES_PATH, DEFAULT_OUTPUT_REGISTRY as COMBINED_REGISTRY, run_build as build_combined
from v3.models.common import DEFAULT_FEATURE_REGISTRY, DEFAULT_MODEL_DATASET

ROOT = Path(__file__).resolve().parents[2]
ABLATION_VERSION = "v3-retained-combined-ablation-001"
FROZEN_BASE = ROOT / "v3" / "data" / "model_dataset_retained_combined_baseline_asof_2026_08_18.parquet"
FROZEN_COMBINED = ROOT / "v3" / "data" / "model_dataset_retained_combined_asof_2026_08_18.parquet"
BASE_PREDICTIONS = ROOT / "v3" / "reports" / "retained_combined_baseline_predictions.parquet"
BASE_METRICS = ROOT / "v3" / "reports" / "retained_combined_baseline_metrics.csv"
BASE_SUMMARY = ROOT / "v3" / "reports" / "retained_combined_baseline_summary.json"
COMBINED_PREDICTIONS = ROOT / "v3" / "reports" / "retained_combined_candidate_predictions.parquet"
COMBINED_METRICS = ROOT / "v3" / "reports" / "retained_combined_candidate_metrics.csv"
COMBINED_SUMMARY = ROOT / "v3" / "reports" / "retained_combined_candidate_summary.json"
COMPARISON_OUTPUT = ROOT / "v3" / "reports" / "retained_combined_ablation_comparison.csv"
LANE_OUTPUT = ROOT / "v3" / "reports" / "retained_combined_ablation_lane_summary.csv"
TOURNAMENT_OUTPUT = ROOT / "v3" / "reports" / "retained_combined_ablation_tournament.csv"
CONTEXT_OUTPUT = ROOT / "v3" / "reports" / "retained_combined_family_context.csv"
REPORT_OUTPUT = ROOT / "v3" / "reports" / "retained_combined_ablation.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family_decision(lanes: pd.DataFrame) -> dict[str, Any]:
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
        "retain_combined": retain,
        "decision": "KEEP_COMBINED_RETAINED_FEATURE_SET" if retain else "DO_NOT_RETAIN_COMBINED_FEATURE_SET",
        "robust_lane_count": count,
        "required_robust_lanes": 2,
        "family_lane_robust_in_both_full_models": lane_results,
        "per_model_lane_robustness": details,
        "criterion": "Combined set is retained only when at least 2 of 3 lanes improve in aggregate and in >=2/3 folds in both EXP-003 and EXP-004.",
    }


def prepare() -> dict[str, Any]:
    ensure_treasury_snapshot()
    ensure_dollar_snapshot()
    feature_report = build_combined()
    base_features = pd.read_parquet(DEFAULT_BASE_FEATURES, engine="pyarrow")
    combined_features = pd.read_parquet(COMBINED_FEATURES_PATH, engine="pyarrow")
    base_model = pd.read_parquet(DEFAULT_MODEL_DATASET, engine="pyarrow")
    assert_frame_equal(combined_features[base_features.columns].reset_index(drop=True), base_features.reset_index(drop=True), check_dtype=True)
    label_columns = [c for c in base_model.columns if c not in base_features.columns]
    candidate = combined_features.merge(base_model[["decision_date", *label_columns]], on="decision_date", how="left", validate="one_to_one")
    assert_frame_equal(candidate[["decision_date", *label_columns]].reset_index(drop=True), base_model[["decision_date", *label_columns]].reset_index(drop=True), check_dtype=True)
    frozen_base = freeze_model_dataset_asof(base_model, ABLATION_AS_OF)
    frozen_candidate = freeze_model_dataset_asof(candidate, ABLATION_AS_OF)
    assert_frame_equal(frozen_candidate[["decision_date", *label_columns]].reset_index(drop=True), frozen_base[["decision_date", *label_columns]].reset_index(drop=True), check_dtype=True)
    FROZEN_BASE.parent.mkdir(parents=True, exist_ok=True)
    frozen_base.to_parquet(FROZEN_BASE, index=False, engine="pyarrow")
    frozen_candidate.to_parquet(FROZEN_COMBINED, index=False, engine="pyarrow")
    return {"feature_report": feature_report, "rows": len(frozen_base), "baseline_sha": _sha(FROZEN_BASE), "candidate_sha": _sha(FROZEN_COMBINED)}


def build_tournament(base: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    base = base.copy(); candidate = candidate.copy()
    base["experiment_id"] = "BASE-" + base["experiment_id"].astype(str)
    base["model_name"] = base["model_name"].astype(str) + " [baseline]"
    candidate["experiment_id"] = "COMBO-" + candidate["experiment_id"].astype(str)
    candidate["model_name"] = candidate["model_name"].astype(str) + " [+retained combined]"
    return summarize_experiments(pd.concat([base, candidate], ignore_index=True, sort=False))


def build_family_context(combined_tournament: pd.DataFrame) -> pd.DataFrame:
    frames = [combined_tournament.assign(context_source="combined_ablation")]
    for path, prefix, label in (
        (ROOT / "v3" / "reports" / "relative_strength_ablation_tournament.csv", "RS-EXP-", "relative_strength"),
        (ROOT / "v3" / "reports" / "treasury_ablation_tournament.csv", "UST-EXP-", "treasury"),
        (ROOT / "v3" / "reports" / "dollar_ablation_tournament.csv", "USD-EXP-", "dollar"),
    ):
        if path.exists():
            frame = pd.read_csv(path)
            frame = frame.loc[frame["experiment_id"].astype(str).str.startswith(prefix)].copy()
            frame["context_source"] = label
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def run_ablation(report_output: Path = REPORT_OUTPUT) -> dict[str, Any]:
    prepared = prepare()
    base_report = run_common_evaluation(dataset_path=FROZEN_BASE, registry_path=DEFAULT_FEATURE_REGISTRY, predictions_output=BASE_PREDICTIONS, metrics_output=BASE_METRICS, summary_output=BASE_SUMMARY)
    candidate_report = run_common_evaluation(dataset_path=FROZEN_COMBINED, registry_path=COMBINED_REGISTRY, predictions_output=COMBINED_PREDICTIONS, metrics_output=COMBINED_METRICS, summary_output=COMBINED_SUMMARY)
    base_metrics = pd.read_csv(BASE_METRICS); candidate_metrics = pd.read_csv(COMBINED_METRICS)
    comparison = compare_metric_frames(base_metrics, candidate_metrics)
    lanes = summarize_lane_improvements(comparison)
    decision = family_decision(lanes)
    tournament = build_tournament(base_metrics, candidate_metrics)
    context = build_family_context(tournament)
    comparison.to_csv(COMPARISON_OUTPUT, index=False); lanes.to_csv(LANE_OUTPUT, index=False); tournament.to_csv(TOURNAMENT_OUTPUT, index=False); context.to_csv(CONTEXT_OUTPUT, index=False)
    full = tournament.loc[tournament["full_candidate"].astype(bool)].sort_values(["overall_full_candidate_rank", "experiment_id"], na_position="last")
    best = None if full.empty else str(full.iloc[0]["experiment_id"])
    promotion = tournament.loc[tournament["promotion_ready"].astype(bool), "experiment_id"].astype(str).tolist()
    report = {
        "status": "RETAINED_COMBINED_ABLATION_COMPLETE",
        "ablation_version": ABLATION_VERSION,
        "ablation_as_of": ABLATION_AS_OF.date().isoformat(),
        "baseline_feature_version": base_report.get("feature_set_version"),
        "candidate_feature_version": candidate_report.get("feature_set_version"),
        "component_feature_versions": ["v3-features-003-relative-strength", "v3-features-004-treasury", "v3-features-005-dollar"],
        "rows": int(prepared["rows"]), "comparison_cells": int(len(comparison)), "sample_hashes_match": True,
        "baseline_frozen_dataset_sha256": prepared["baseline_sha"], "candidate_frozen_dataset_sha256": prepared["candidate_sha"],
        "feature_family_decision": decision,
        "best_ranked_full_candidate_in_ablation_tournament": best,
        "promotion_ready_experiments_from_absolute_gates": promotion,
        "champion_selected": False,
        "champion_selection_status": "DEFERRED_TO_V3_018_V3_019",
        "trading_policy_status": "UNCHANGED_AND_DEFERRED_TO_V3_016",
        "next": "V3-016 decision-policy research framework",
    }
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT); args = parser.parse_args()
    report = run_ablation(args.report_output)
    return 0 if report["status"] == "RETAINED_COMBINED_ABLATION_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
