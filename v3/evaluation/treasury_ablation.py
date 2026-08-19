#!/usr/bin/env python3
"""V3-015A controlled ablation for the Treasury-rate feature family."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.testing import assert_frame_equal

from v3.evaluation.tournament import summarize_experiments
from v3.evaluation.vix_ablation import (
    ABLATION_AS_OF,
    FULL_INTERFACE_EXPERIMENTS,
    REQUIRED_TARGET_TYPES,
    compare_metric_frames,
    freeze_model_dataset_asof,
    summarize_lane_improvements,
)
from v3.evaluation.walk_forward import run_common_evaluation
from v3.features.build_treasury_features import (
    DEFAULT_BASE_FEATURES,
    DEFAULT_OUTPUT as TREASURY_FEATURES_PATH,
    DEFAULT_OUTPUT_REGISTRY as TREASURY_REGISTRY,
    DEFAULT_REPORT as TREASURY_FEATURE_REPORT,
    DEFAULT_SOURCE as TREASURY_SNAPSHOT,
    run_build as build_treasury_features,
)
from v3.models.common import DEFAULT_FEATURE_REGISTRY, DEFAULT_MODEL_DATASET

ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = ROOT / "v3" / "data" / "treasury_source.json"
ABLATION_VERSION = "v3-treasury-ablation-001"

FROZEN_BASE_DATASET = (
    ROOT / "v3" / "data" / "model_dataset_treasury_baseline_asof_2026_08_18.parquet"
)
FROZEN_TREASURY_DATASET = (
    ROOT / "v3" / "data" / "model_dataset_treasury_asof_2026_08_18.parquet"
)

BASE_PREDICTIONS = ROOT / "v3" / "reports" / "treasury_ablation_baseline_predictions.parquet"
BASE_METRICS = ROOT / "v3" / "reports" / "treasury_ablation_baseline_metrics.csv"
BASE_SUMMARY = ROOT / "v3" / "reports" / "treasury_ablation_baseline_summary.json"
TREASURY_PREDICTIONS = ROOT / "v3" / "reports" / "treasury_ablation_candidate_predictions.parquet"
TREASURY_METRICS = ROOT / "v3" / "reports" / "treasury_ablation_candidate_metrics.csv"
TREASURY_SUMMARY = ROOT / "v3" / "reports" / "treasury_ablation_candidate_summary.json"
COMPARISON_OUTPUT = ROOT / "v3" / "reports" / "treasury_ablation_comparison.csv"
LANE_OUTPUT = ROOT / "v3" / "reports" / "treasury_ablation_lane_summary.csv"
TOURNAMENT_OUTPUT = ROOT / "v3" / "reports" / "treasury_ablation_tournament.csv"
REPORT_OUTPUT = ROOT / "v3" / "reports" / "treasury_ablation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_snapshot() -> dict[str, Any]:
    if not TREASURY_SNAPSHOT.exists() or not SOURCE_MANIFEST.exists():
        raise ValueError(
            "V3-015A requires the frozen Treasury snapshot and source manifest"
        )
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    expected = str(manifest.get("snapshot_sha256", ""))
    actual = _sha256(TREASURY_SNAPSHOT)
    if not expected or expected != actual:
        raise ValueError(
            "Treasury snapshot hash does not match its manifest; do not silently refetch"
        )
    return manifest


def prepare_frozen_datasets() -> dict[str, Any]:
    manifest = verify_snapshot()
    feature_report = build_treasury_features(
        base_features_path=DEFAULT_BASE_FEATURES,
        source_path=TREASURY_SNAPSHOT,
        output_path=TREASURY_FEATURES_PATH,
        output_registry_path=TREASURY_REGISTRY,
        report_path=TREASURY_FEATURE_REPORT,
    )

    base_features = pd.read_parquet(DEFAULT_BASE_FEATURES, engine="pyarrow")
    candidate_features = pd.read_parquet(TREASURY_FEATURES_PATH, engine="pyarrow")
    base_model = pd.read_parquet(DEFAULT_MODEL_DATASET, engine="pyarrow")

    assert_frame_equal(
        candidate_features[base_features.columns].reset_index(drop=True),
        base_features.reset_index(drop=True),
        check_dtype=True,
    )

    label_columns = [
        column for column in base_model.columns if column not in base_features.columns
    ]
    if not label_columns:
        raise ValueError("Baseline model dataset contains no label columns")

    candidate_model = candidate_features.merge(
        base_model[["decision_date", *label_columns]],
        on="decision_date",
        how="left",
        validate="one_to_one",
    )
    assert_frame_equal(
        candidate_model[["decision_date", *label_columns]].reset_index(drop=True),
        base_model[["decision_date", *label_columns]].reset_index(drop=True),
        check_dtype=True,
    )

    frozen_base = freeze_model_dataset_asof(base_model, ABLATION_AS_OF)
    frozen_candidate = freeze_model_dataset_asof(candidate_model, ABLATION_AS_OF)
    assert_frame_equal(
        frozen_candidate[["decision_date", *label_columns]].reset_index(drop=True),
        frozen_base[["decision_date", *label_columns]].reset_index(drop=True),
        check_dtype=True,
    )

    FROZEN_BASE_DATASET.parent.mkdir(parents=True, exist_ok=True)
    frozen_base.to_parquet(FROZEN_BASE_DATASET, index=False, engine="pyarrow")
    frozen_candidate.to_parquet(FROZEN_TREASURY_DATASET, index=False, engine="pyarrow")

    return {
        "manifest": manifest,
        "feature_report": feature_report,
        "rows": int(len(frozen_base)),
        "as_of": ABLATION_AS_OF.date().isoformat(),
        "baseline_dataset_sha256": _sha256(FROZEN_BASE_DATASET),
        "candidate_dataset_sha256": _sha256(FROZEN_TREASURY_DATASET),
    }


def treasury_family_decision(lanes: pd.DataFrame) -> dict[str, Any]:
    """Apply the pre-registered 2-of-3-lanes rule in both full models."""
    full = lanes.loc[lanes["experiment_id"].isin(FULL_INTERFACE_EXPERIMENTS)].copy()
    missing_models = sorted(
        set(FULL_INTERFACE_EXPERIMENTS).difference(full["experiment_id"])
    )
    if missing_models:
        raise ValueError(f"Missing full-interface ablation models: {missing_models}")

    lane_results: dict[str, bool] = {}
    details: dict[str, Any] = {}
    for target_type in REQUIRED_TARGET_TYPES:
        lane = full.loc[full["target_type"] == target_type]
        by_model = {
            str(row["experiment_id"]): bool(row["robust_improvement"])
            for _, row in lane.iterrows()
        }
        if set(by_model) != set(FULL_INTERFACE_EXPERIMENTS):
            raise ValueError(
                f"Target lane {target_type} does not contain both full-interface models"
            )
        lane_results[target_type] = all(by_model.values())
        details[target_type] = by_model

    robust_lane_count = int(sum(lane_results.values()))
    retain = robust_lane_count >= 2
    return {
        "retain_treasury": retain,
        "decision": (
            "KEEP_TREASURY_FOR_LATER_RESEARCH"
            if retain
            else "DO_NOT_RETAIN_TREASURY_FEATURE_FAMILY"
        ),
        "robust_lane_count": robust_lane_count,
        "required_robust_lanes": 2,
        "family_lane_robust_in_both_full_models": lane_results,
        "per_model_lane_robustness": details,
        "criterion": (
            "A lane is robust only when its aggregate primary metric improves and "
            "at least 2 of 3 chronological folds improve. Treasury is retained only "
            "when at least 2 of 3 lanes are robust in both EXP-003 and EXP-004."
        ),
    }


def build_tournament(
    baseline_metrics: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
) -> pd.DataFrame:
    baseline = baseline_metrics.copy()
    baseline["experiment_id"] = "BASE-" + baseline["experiment_id"].astype(str)
    baseline["model_name"] = baseline["model_name"].astype(str) + " [baseline]"
    candidate = candidate_metrics.copy()
    candidate["experiment_id"] = "UST-" + candidate["experiment_id"].astype(str)
    candidate["model_name"] = candidate["model_name"].astype(str) + " [+Treasury]"
    return summarize_experiments(
        pd.concat([baseline, candidate], ignore_index=True, sort=False)
    )


def run_treasury_ablation(report_output: Path = REPORT_OUTPUT) -> dict[str, Any]:
    prepared = prepare_frozen_datasets()

    baseline_report = run_common_evaluation(
        dataset_path=FROZEN_BASE_DATASET,
        registry_path=DEFAULT_FEATURE_REGISTRY,
        predictions_output=BASE_PREDICTIONS,
        metrics_output=BASE_METRICS,
        summary_output=BASE_SUMMARY,
    )
    candidate_report = run_common_evaluation(
        dataset_path=FROZEN_TREASURY_DATASET,
        registry_path=TREASURY_REGISTRY,
        predictions_output=TREASURY_PREDICTIONS,
        metrics_output=TREASURY_METRICS,
        summary_output=TREASURY_SUMMARY,
    )

    baseline_metrics = pd.read_csv(BASE_METRICS)
    candidate_metrics = pd.read_csv(TREASURY_METRICS)
    comparison = compare_metric_frames(baseline_metrics, candidate_metrics)
    lanes = summarize_lane_improvements(comparison)
    decision = treasury_family_decision(lanes)
    tournament = build_tournament(baseline_metrics, candidate_metrics)

    COMPARISON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_OUTPUT, index=False)
    lanes.to_csv(LANE_OUTPUT, index=False)
    tournament.to_csv(TOURNAMENT_OUTPUT, index=False)

    full = tournament.loc[tournament["full_candidate"].astype(bool)].sort_values(
        ["overall_full_candidate_rank", "experiment_id"], na_position="last"
    )
    best_ranked = None if full.empty else str(full.iloc[0]["experiment_id"])
    promotion_ready = tournament.loc[
        tournament["promotion_ready"].astype(bool), "experiment_id"
    ].astype(str).tolist()

    report: dict[str, Any] = {
        "status": "TREASURY_ABLATION_COMPLETE",
        "ablation_version": ABLATION_VERSION,
        "ablation_as_of": prepared["as_of"],
        "baseline_feature_version": baseline_report.get("feature_set_version"),
        "candidate_feature_version": candidate_report.get("feature_set_version"),
        "rows": prepared["rows"],
        "baseline_frozen_dataset_sha256": prepared["baseline_dataset_sha256"],
        "candidate_frozen_dataset_sha256": prepared["candidate_dataset_sha256"],
        "snapshot_sha256": prepared["manifest"].get("snapshot_sha256"),
        "source_end": prepared["manifest"].get("end"),
        "comparison_cells": int(len(comparison)),
        "sample_hashes_match": True,
        "feature_family_decision": decision,
        "best_ranked_full_candidate_in_ablation_tournament": best_ranked,
        "promotion_ready_experiments_from_absolute_gates": promotion_ready,
        "champion_selected": False,
        "champion_selection_status": "DEFERRED_TO_V3_018_V3_019",
        "trading_policy_status": "UNCHANGED_AND_DEFERRED_TO_V3_016",
        "next": "V3-015B broad-dollar feature family",
        "note": (
            "V3-015A changes only the Treasury feature family. The experiment is "
            "frozen as-of 2026-08-18 and does not promote a model or tune a trading policy."
        ),
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    args = parse_args()
    report = run_treasury_ablation(args.report_output)
    return 0 if report["status"] == "TREASURY_ABLATION_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
