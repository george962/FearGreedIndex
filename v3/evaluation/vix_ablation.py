#!/usr/bin/env python3
"""V3-012 controlled ablation: frozen baseline features versus baseline + VIX.

The ablation changes exactly one research dimension: the feature registry/dataset.
Model classes, hyperparameters, folds, labels, maturity gates, seeds, and common
evaluation metrics are inherited unchanged from V3-005 through V3-010.

The experiment is frozen as-of 2026-08-18. Outcomes that were not legally
knowable by that date are censored before either lane is evaluated, so later
live-data growth cannot retroactively change the V3-012 result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from v3.evaluation.tournament import summarize_experiments
from v3.evaluation.walk_forward import run_common_evaluation
from v3.features.build_vix_features import (
    DEFAULT_BASE_FEATURES,
    DEFAULT_OUTPUT as VIX_FEATURES,
    DEFAULT_REGISTRY as VIX_REGISTRY,
    DEFAULT_REPORT as VIX_MISSINGNESS_REPORT,
    DEFAULT_VIX as VIX_SNAPSHOT,
    run_build as build_vix_features,
)
from v3.models.common import DEFAULT_FEATURE_REGISTRY, DEFAULT_MODEL_DATASET, HORIZONS

ROOT = Path(__file__).resolve().parents[2]
VIX_SOURCE_MANIFEST = ROOT / "v3" / "data" / "vix_source.json"
ABLATION_AS_OF = pd.Timestamp("2026-08-18")
ABLATION_VERSION = "v3-vix-ablation-001"

FROZEN_BASE_MODEL_DATASET = (
    ROOT / "v3" / "data" / "model_dataset_ablation_asof_2026_08_18.parquet"
)
FROZEN_VIX_MODEL_DATASET = (
    ROOT / "v3" / "data" / "model_dataset_vix_ablation_asof_2026_08_18.parquet"
)

BASE_PREDICTIONS = ROOT / "v3" / "reports" / "vix_ablation_baseline_predictions.parquet"
BASE_METRICS = ROOT / "v3" / "reports" / "vix_ablation_baseline_metrics.csv"
BASE_SUMMARY = ROOT / "v3" / "reports" / "vix_ablation_baseline_summary.json"
VIX_PREDICTIONS = ROOT / "v3" / "reports" / "vix_ablation_vix_predictions.parquet"
VIX_METRICS = ROOT / "v3" / "reports" / "vix_ablation_vix_metrics.csv"
VIX_SUMMARY = ROOT / "v3" / "reports" / "vix_ablation_vix_summary.json"
COMPARISON_OUTPUT = ROOT / "v3" / "reports" / "vix_ablation_comparison.csv"
LANE_OUTPUT = ROOT / "v3" / "reports" / "vix_ablation_lane_summary.csv"
TOURNAMENT_OUTPUT = ROOT / "v3" / "reports" / "vix_ablation_tournament.csv"
REPORT_OUTPUT = ROOT / "v3" / "reports" / "vix_ablation.json"

FULL_INTERFACE_EXPERIMENTS = ("EXP-003", "EXP-004")
REQUIRED_TARGET_TYPES = (
    "classification",
    "return_regression",
    "drawdown_regression",
)
KEY_COLUMNS = (
    "experiment_id",
    "model_name",
    "fold",
    "target_type",
    "target",
    "horizon",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_vix_snapshot() -> dict[str, Any]:
    if not VIX_SNAPSHOT.exists() or not VIX_SOURCE_MANIFEST.exists():
        raise ValueError(
            "V3-012 requires the checked-in V3-011 VIX snapshot and source manifest"
        )
    manifest = json.loads(VIX_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    expected = str(manifest.get("snapshot_sha256", ""))
    actual = _sha256(VIX_SNAPSHOT)
    if not expected or expected != actual:
        raise ValueError(
            "Checked-in VIX snapshot hash does not match vix_source.json; "
            "do not run an ablation against silently changed data"
        )
    return manifest


def freeze_model_dataset_asof(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp | str = ABLATION_AS_OF,
) -> pd.DataFrame:
    """Freeze decision rows and censor labels unavailable at the experiment cutoff."""
    cutoff_ts = pd.Timestamp(cutoff).normalize()
    frozen = frame.copy()
    frozen["decision_date"] = pd.to_datetime(
        frozen["decision_date"], errors="raise"
    ).dt.normalize()
    frozen = frozen.loc[frozen["decision_date"].le(cutoff_ts)].copy()
    frozen = frozen.sort_values("decision_date").reset_index(drop=True)

    for horizon in HORIZONS:
        known_column = f"_forward_{horizon}d_known_date"
        return_column = f"forward_return_{horizon}d"
        positive_column = f"forward_positive_{horizon}d"
        drawdown_column = f"max_drawdown_{horizon}d"
        required = {
            known_column,
            return_column,
            positive_column,
            drawdown_column,
        }
        missing = sorted(required.difference(frozen.columns))
        if missing:
            raise ValueError(
                f"Model dataset missing horizon {horizon} freeze columns: {missing}"
            )

        known = pd.to_datetime(frozen[known_column], errors="coerce").dt.normalize()
        unavailable = known.isna() | known.gt(cutoff_ts)
        frozen[known_column] = known
        frozen.loc[unavailable, known_column] = pd.NaT
        frozen.loc[unavailable, return_column] = np.nan
        frozen.loc[unavailable, drawdown_column] = np.nan
        frozen.loc[unavailable, positive_column] = pd.NA

        if horizon == 20 and "further_5pct_decline_20d" in frozen.columns:
            frozen.loc[unavailable, "further_5pct_decline_20d"] = pd.NA

    if frozen["decision_date"].duplicated().any():
        raise ValueError("Frozen ablation dataset has duplicate decision dates")
    if not frozen["decision_date"].is_monotonic_increasing:
        raise ValueError("Frozen ablation dataset is not chronologically sorted")
    return frozen


def prepare_frozen_model_datasets() -> dict[str, Any]:
    """Build +VIX features, preserve labels exactly, then freeze both lanes as-of."""
    source_manifest = verify_vix_snapshot()
    feature_report = build_vix_features(
        base_features_path=DEFAULT_BASE_FEATURES,
        vix_path=VIX_SNAPSHOT,
        output_path=VIX_FEATURES,
        registry_path=VIX_REGISTRY,
        report_path=VIX_MISSINGNESS_REPORT,
    )

    base_features = pd.read_parquet(DEFAULT_BASE_FEATURES, engine="pyarrow")
    vix_features = pd.read_parquet(VIX_FEATURES, engine="pyarrow")
    base_model = pd.read_parquet(DEFAULT_MODEL_DATASET, engine="pyarrow")

    # V3-011/V3-012 are feature-family experiments only. Existing feature values
    # and decision rows must remain byte-equivalent in DataFrame terms.
    assert_frame_equal(
        vix_features[base_features.columns].reset_index(drop=True),
        base_features.reset_index(drop=True),
        check_dtype=True,
    )

    label_columns = [
        column for column in base_model.columns if column not in base_features.columns
    ]
    if not label_columns:
        raise ValueError("Baseline model dataset contains no label columns")

    vix_model = vix_features.merge(
        base_model[["decision_date", *label_columns]],
        on="decision_date",
        how="left",
        validate="one_to_one",
    )
    assert_frame_equal(
        vix_model[["decision_date", *label_columns]].reset_index(drop=True),
        base_model[["decision_date", *label_columns]].reset_index(drop=True),
        check_dtype=True,
    )

    frozen_base = freeze_model_dataset_asof(base_model, ABLATION_AS_OF)
    frozen_vix = freeze_model_dataset_asof(vix_model, ABLATION_AS_OF)

    # Re-check the label contract after censoring unknown-as-of outcomes.
    frozen_label_columns = ["decision_date", *label_columns]
    assert_frame_equal(
        frozen_vix[frozen_label_columns].reset_index(drop=True),
        frozen_base[frozen_label_columns].reset_index(drop=True),
        check_dtype=True,
    )

    FROZEN_BASE_MODEL_DATASET.parent.mkdir(parents=True, exist_ok=True)
    frozen_base.to_parquet(FROZEN_BASE_MODEL_DATASET, index=False, engine="pyarrow")
    frozen_vix.to_parquet(FROZEN_VIX_MODEL_DATASET, index=False, engine="pyarrow")

    return {
        "source_manifest": source_manifest,
        "feature_report": feature_report,
        "label_columns": label_columns,
        "rows": int(len(frozen_base)),
        "as_of": ABLATION_AS_OF.date().isoformat(),
        "baseline_dataset_sha256": _sha256(FROZEN_BASE_MODEL_DATASET),
        "vix_dataset_sha256": _sha256(FROZEN_VIX_MODEL_DATASET),
    }


def compare_metric_frames(
    baseline: pd.DataFrame,
    vix: pd.DataFrame,
) -> pd.DataFrame:
    """Compare identical fold/target cells and fail on any sample mismatch."""
    required = set(KEY_COLUMNS) | {"sample_sha256"}
    for name, frame in (("baseline", baseline), ("vix", vix)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} metrics missing columns: {missing}")

    merged = baseline.merge(
        vix,
        on=list(KEY_COLUMNS),
        how="outer",
        validate="one_to_one",
        suffixes=("_baseline", "_vix"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        bad = merged.loc[
            ~merged["_merge"].eq("both"), list(KEY_COLUMNS) + ["_merge"]
        ]
        raise ValueError(
            "Baseline and VIX evaluation cells differ: "
            + repr(bad.head(10).to_dict(orient="records"))
        )
    same_samples = merged["sample_sha256_baseline"].eq(
        merged["sample_sha256_vix"]
    )
    if not same_samples.all():
        bad = merged.loc[~same_samples, list(KEY_COLUMNS)]
        raise ValueError(
            "Baseline and VIX metrics use different realized-date samples: "
            + repr(bad.head(10).to_dict(orient="records"))
        )

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        target_type = str(row["target_type"])
        common = {column: row[column] for column in KEY_COLUMNS}
        common["sample_sha256"] = row["sample_sha256_baseline"]

        if target_type == "classification":
            primary_metric = "brier_score"
            primary_baseline = float(row["brier_score_baseline"])
            primary_vix = float(row["brier_score_vix"])
            primary_improvement = primary_baseline - primary_vix
            secondary_metric = "relative_brier_improvement"
            secondary_baseline = float(row["relative_brier_improvement_baseline"])
            secondary_vix = float(row["relative_brier_improvement_vix"])
            secondary_improvement = secondary_vix - secondary_baseline
        elif target_type in {"return_regression", "drawdown_regression"}:
            primary_metric = "spearman_rank_correlation"
            primary_baseline = float(row["spearman_rank_correlation_baseline"])
            primary_vix = float(row["spearman_rank_correlation_vix"])
            primary_improvement = primary_vix - primary_baseline
            secondary_metric = "rmse"
            secondary_baseline = float(row["rmse_baseline"])
            secondary_vix = float(row["rmse_vix"])
            secondary_improvement = secondary_baseline - secondary_vix
        else:
            raise ValueError(f"Unsupported target_type in ablation: {target_type}")

        rows.append(
            {
                **common,
                "primary_metric": primary_metric,
                "primary_baseline": primary_baseline,
                "primary_vix": primary_vix,
                "primary_improvement": primary_improvement,
                "secondary_metric": secondary_metric,
                "secondary_baseline": secondary_baseline,
                "secondary_vix": secondary_vix,
                "secondary_improvement": secondary_improvement,
                "vix_primary_improved": bool(primary_improvement > 0.0),
            }
        )

    return pd.DataFrame(rows).sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def summarize_lane_improvements(comparison: pd.DataFrame) -> pd.DataFrame:
    """Apply the pre-registered robust-lane rule before feature-family selection."""
    fold_rows: list[dict[str, Any]] = []
    for (experiment_id, model_name, target_type, fold), group in comparison.groupby(
        ["experiment_id", "model_name", "target_type", "fold"], sort=True
    ):
        fold_rows.append(
            {
                "experiment_id": experiment_id,
                "model_name": model_name,
                "target_type": target_type,
                "fold": fold,
                "fold_primary_improvement": float(
                    group["primary_improvement"].mean()
                ),
            }
        )
    fold_summary = pd.DataFrame(fold_rows)

    rows: list[dict[str, Any]] = []
    for (experiment_id, model_name, target_type), group in comparison.groupby(
        ["experiment_id", "model_name", "target_type"], sort=True
    ):
        folds = fold_summary.loc[
            (fold_summary["experiment_id"] == experiment_id)
            & (fold_summary["model_name"] == model_name)
            & (fold_summary["target_type"] == target_type)
        ]
        aggregate = float(group["primary_improvement"].mean())
        improved_folds = int((folds["fold_primary_improvement"] > 0.0).sum())
        total_folds = int(len(folds))
        rows.append(
            {
                "experiment_id": experiment_id,
                "model_name": model_name,
                "target_type": target_type,
                "primary_metric": str(group["primary_metric"].iloc[0]),
                "aggregate_primary_improvement": aggregate,
                "improved_folds": improved_folds,
                "total_folds": total_folds,
                "robust_improvement": bool(
                    aggregate > 0.0 and total_folds >= 3 and improved_folds >= 2
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["experiment_id", "target_type"]
    ).reset_index(drop=True)


def vix_family_decision(lanes: pd.DataFrame) -> dict[str, Any]:
    """Retain VIX only when >=2 lanes improve robustly in both full models."""
    full = lanes.loc[lanes["experiment_id"].isin(FULL_INTERFACE_EXPERIMENTS)].copy()
    missing_models = sorted(
        set(FULL_INTERFACE_EXPERIMENTS).difference(full["experiment_id"])
    )
    if missing_models:
        raise ValueError(f"Missing full-interface ablation models: {missing_models}")

    lane_results: dict[str, bool] = {}
    details: dict[str, Any] = {}
    for target_type in REQUIRED_TARGET_TYPES:
        lane = full.loc[full["target_type"] == target_type].copy()
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
        "retain_vix": retain,
        "decision": (
            "KEEP_VIX_FOR_LATER_RESEARCH"
            if retain
            else "DO_NOT_RETAIN_VIX_FEATURE_FAMILY"
        ),
        "robust_lane_count": robust_lane_count,
        "required_robust_lanes": 2,
        "family_lane_robust_in_both_full_models": lane_results,
        "per_model_lane_robustness": details,
        "criterion": (
            "A lane is robust only when its aggregate primary metric improves and "
            "at least 2 of 3 chronological folds improve. VIX is retained only when "
            "at least 2 of 3 lanes are robust in both EXP-003 and EXP-004."
        ),
    }


def build_ablation_tournament(
    baseline_metrics: pd.DataFrame,
    vix_metrics: pd.DataFrame,
) -> pd.DataFrame:
    baseline = baseline_metrics.copy()
    baseline["experiment_id"] = "BASE-" + baseline["experiment_id"].astype(str)
    baseline["model_name"] = baseline["model_name"].astype(str) + " [baseline]"
    vix = vix_metrics.copy()
    vix["experiment_id"] = "VIX-" + vix["experiment_id"].astype(str)
    vix["model_name"] = vix["model_name"].astype(str) + " [+VIX]"
    combined = pd.concat([baseline, vix], ignore_index=True, sort=False)
    return summarize_experiments(combined)


def run_vix_ablation(report_output: Path = REPORT_OUTPUT) -> dict[str, Any]:
    prepared = prepare_frozen_model_datasets()

    baseline_report = run_common_evaluation(
        dataset_path=FROZEN_BASE_MODEL_DATASET,
        registry_path=DEFAULT_FEATURE_REGISTRY,
        predictions_output=BASE_PREDICTIONS,
        metrics_output=BASE_METRICS,
        summary_output=BASE_SUMMARY,
    )
    vix_report = run_common_evaluation(
        dataset_path=FROZEN_VIX_MODEL_DATASET,
        registry_path=VIX_REGISTRY,
        predictions_output=VIX_PREDICTIONS,
        metrics_output=VIX_METRICS,
        summary_output=VIX_SUMMARY,
    )

    baseline_metrics = pd.read_csv(BASE_METRICS)
    vix_metrics = pd.read_csv(VIX_METRICS)
    comparison = compare_metric_frames(baseline_metrics, vix_metrics)
    lanes = summarize_lane_improvements(comparison)
    decision = vix_family_decision(lanes)
    tournament = build_ablation_tournament(baseline_metrics, vix_metrics)

    COMPARISON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_OUTPUT, index=False)
    lanes.to_csv(LANE_OUTPUT, index=False)
    tournament.to_csv(TOURNAMENT_OUTPUT, index=False)

    full_tournament = tournament.loc[tournament["full_candidate"].astype(bool)].copy()
    full_tournament = full_tournament.sort_values(
        ["overall_full_candidate_rank", "experiment_id"], na_position="last"
    )
    best_ranked = (
        None if full_tournament.empty else str(full_tournament.iloc[0]["experiment_id"])
    )
    promotion_ready = tournament.loc[
        tournament["promotion_ready"].astype(bool), "experiment_id"
    ].astype(str).tolist()

    report: dict[str, Any] = {
        "status": "VIX_ABLATION_COMPLETE",
        "ablation_version": ABLATION_VERSION,
        "ablation_as_of": prepared["as_of"],
        "baseline_feature_version": baseline_report.get("feature_set_version"),
        "vix_feature_version": vix_report.get("feature_set_version"),
        "rows": prepared["rows"],
        "baseline_frozen_dataset_sha256": prepared["baseline_dataset_sha256"],
        "vix_frozen_dataset_sha256": prepared["vix_dataset_sha256"],
        "vix_source_end": prepared["source_manifest"].get("end"),
        "vix_snapshot_sha256": prepared["source_manifest"].get("snapshot_sha256"),
        "comparison_cells": int(len(comparison)),
        "sample_hashes_match": True,
        "feature_family_decision": decision,
        "best_ranked_full_candidate_in_ablation_tournament": best_ranked,
        "promotion_ready_experiments_from_absolute_gates": promotion_ready,
        "champion_selected": False,
        "champion_selection_status": "DEFERRED_TO_V3_018_V3_019",
        "trading_policy_status": "UNCHANGED_AND_DEFERRED_TO_V3_016",
        "next": "V3-013 QQQ/SPY relative-strength features",
        "note": (
            "Relative VIX improvement and V3-010 absolute promotion gates are both "
            "reported, but V3-012 does not promote a model or tune a decision policy. "
            "The experiment is frozen as-of 2026-08-18; later-maturing outcomes are "
            "censored from both lanes."
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
    report = run_vix_ablation(report_output=args.report_output)
    return 0 if report["status"] == "VIX_ABLATION_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
