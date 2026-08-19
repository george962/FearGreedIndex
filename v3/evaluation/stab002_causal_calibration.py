#!/usr/bin/env python3
"""Run STAB-002: nested causal calibration of the frozen STAB-001 ranking score."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from v3.evaluation.exp006_opportunity import (
    AS_OF,
    add_opportunity_targets,
    eligible_training_mask,
    expected_calibration_error,
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
MANIFEST = ROOT / "v3" / "methodology" / "STAB-002" / "manifest.json"
STAB001_METRICS = ROOT / "v3" / "reports" / "stab001_metrics.csv"
EVALUATION = ROOT / "v3" / "reports" / "stab002_evaluation.json"
METRICS = ROOT / "v3" / "reports" / "stab002_metrics.csv"
INNER = ROOT / "v3" / "reports" / "stab002_inner_calibration.csv"

METHOD_ID = "STAB-002"
CALIBRATION_ROWS = 240
CALIBRATION_BLOCKS = 3
CALIBRATION_BLOCK_ROWS = 80
MIN_INNER_FORMATION_ROWS = 400
MIN_INNER_SELECTED_FEATURES = 3
MIN_INNER_SELECTED_FAMILIES = 2
PLATT_PARAMS = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibration_blocks(outer_train: pd.DataFrame) -> list[pd.DataFrame]:
    ordered = outer_train.sort_values("decision_date").reset_index(drop=True)
    if len(ordered) < MIN_INNER_FORMATION_ROWS + CALIBRATION_ROWS:
        raise ValueError(
            "STAB-002 outer training history is too short for 400 formation + 240 calibration rows"
        )
    tail = ordered.iloc[-CALIBRATION_ROWS:].copy()
    blocks = [
        tail.iloc[index * CALIBRATION_BLOCK_ROWS : (index + 1) * CALIBRATION_BLOCK_ROWS].copy()
        for index in range(CALIBRATION_BLOCKS)
    ]
    if any(len(block) != CALIBRATION_BLOCK_ROWS for block in blocks):
        raise ValueError("STAB-002 calibration blocks must be exactly 3 x 80 rows")
    return blocks


def _block_classes(block: pd.DataFrame) -> int:
    return int(block["favorable_entry_20d"].astype(int).nunique())


def build_nested_oof_calibration(
    frame: pd.DataFrame,
    outer_train: pd.DataFrame,
    features: list[str],
    *,
    outer_fold: str,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    score_rows: list[pd.DataFrame] = []
    support_rows: list[dict[str, Any]] = []
    support_ok = True

    for index, block in enumerate(calibration_blocks(outer_train), start=1):
        block = block.sort_values("decision_date").reset_index(drop=True)
        block_start = pd.Timestamp(block["decision_date"].min()).normalize()
        causal_cutoff = block_start - pd.Timedelta(days=1)
        formation = frame.loc[eligible_training_mask(frame, causal_cutoff)].copy()
        formation = formation.sort_values("decision_date").reset_index(drop=True)

        selected, _ = select_stable_features(formation, features)
        selected_families = {feature_family(str(item["feature"])) for item in selected}
        formation_rows = int(len(formation))
        target_classes = _block_classes(block)
        row_ok = bool(
            formation_rows >= MIN_INNER_FORMATION_ROWS
            and len(selected) >= MIN_INNER_SELECTED_FEATURES
            and len(selected_families) >= MIN_INNER_SELECTED_FAMILIES
            and target_classes == 2
        )
        support_ok = support_ok and row_ok

        if selected:
            raw_score = stability_raw_score(formation, block, selected)
        else:
            raw_score = np.zeros(len(block), dtype=float)

        score_rows.append(
            pd.DataFrame(
                {
                    "decision_date": block["decision_date"].to_numpy(),
                    "outer_fold": outer_fold,
                    "inner_block": index,
                    "raw_stability_score": raw_score,
                    "favorable_entry_20d": block["favorable_entry_20d"].astype(int).to_numpy(),
                }
            )
        )
        support_rows.append(
            {
                "outer_fold": outer_fold,
                "inner_block": index,
                "block_start": block_start.strftime("%Y-%m-%d"),
                "causal_formation_cutoff": causal_cutoff.strftime("%Y-%m-%d"),
                "formation_rows": formation_rows,
                "calibration_rows": int(len(block)),
                "calibration_target_classes": target_classes,
                "selected_feature_count": int(len(selected)),
                "selected_family_count": int(len(selected_families)),
                "selected_features": "|".join(str(item["feature"]) for item in selected),
                "support_pass": row_ok,
            }
        )

    oof = pd.concat(score_rows, ignore_index=True)
    support = pd.DataFrame(support_rows)
    if len(oof) != CALIBRATION_ROWS:
        raise ValueError("STAB-002 must generate exactly 240 inner OOF calibration rows")
    if oof["decision_date"].duplicated().any():
        raise ValueError("STAB-002 inner calibration dates overlap")
    return oof, support, support_ok


def fit_platt(oof: pd.DataFrame) -> tuple[LogisticRegression, float, float, bool]:
    if len(oof) != CALIBRATION_ROWS:
        raise ValueError("STAB-002 Platt fit requires exactly 240 OOF rows")
    y = oof["favorable_entry_20d"].astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("STAB-002 pooled calibration history must contain both classes")
    x = oof[["raw_stability_score"]].to_numpy(float)
    model = LogisticRegression(**PLATT_PARAMS)
    model.fit(x, y)
    slope = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    return model, slope, intercept, bool(np.isfinite(slope) and slope > 0.0)


def _probability(model: LogisticRegression, raw_score: np.ndarray) -> np.ndarray:
    classes = model.classes_.tolist()
    if 1 not in classes:
        raise ValueError("STAB-002 Platt calibrator lacks positive class")
    index = classes.index(1)
    return np.clip(
        np.asarray(model.predict_proba(np.asarray(raw_score, dtype=float).reshape(-1, 1))[:, index]),
        0.0,
        1.0,
    )


def _expected_sample_hashes() -> dict[str, str]:
    metrics = pd.read_csv(STAB001_METRICS)
    hashes: dict[str, str] = {}
    for fold, group in metrics.groupby("fold", sort=False):
        values = group["sample_sha256"].dropna().astype(str).unique().tolist()
        if len(values) != 1:
            raise ValueError(f"STAB-001 sample hash is ambiguous for {fold}")
        hashes[str(fold)] = values[0]
    return hashes


def summarize(metrics: pd.DataFrame, inner: pd.DataFrame) -> dict[str, Any]:
    support_pass = bool(
        len(inner) == CALIBRATION_BLOCKS * 3
        and inner["calibration_rows"].eq(CALIBRATION_BLOCK_ROWS).all()
        and inner["formation_rows"].ge(MIN_INNER_FORMATION_ROWS).all()
        and inner["calibration_target_classes"].eq(2).all()
        and inner["selected_feature_count"].ge(MIN_INNER_SELECTED_FEATURES).all()
        and inner["selected_family_count"].ge(MIN_INNER_SELECTED_FAMILIES).all()
        and inner["support_pass"].all()
        and metrics["calibration_rows"].eq(CALIBRATION_ROWS).all()
        and metrics["platt_positive_slope"].all()
        and metrics["sample_hash_matches_stab001"].all()
    )
    mean_auc = float(metrics["roc_auc"].mean())
    positive_auc = int(metrics["roc_auc"].gt(0.5).sum())
    minimum_auc = float(metrics["roc_auc"].min())
    mean_relative = float(metrics["relative_brier_improvement"].mean())
    positive_relative = int(metrics["relative_brier_improvement"].gt(0.0).sum())
    minimum_relative = float(metrics["relative_brier_improvement"].min())
    better_raw_folds = int(metrics["brier_improvement_vs_raw_stab001"].gt(0.0).sum())
    aggregate_better_raw = bool(metrics["brier_score"].mean() < metrics["raw_stab001_brier_score"].mean())
    mean_ece = float(metrics["expected_calibration_error"].mean())
    raw_mean_ece = float(metrics["raw_stab001_expected_calibration_error"].mean())
    ece_improved = bool(mean_ece < raw_mean_ece)

    viable = bool(
        support_pass
        and mean_auc > 0.52
        and positive_auc == 3
        and minimum_auc >= 0.50
        and mean_relative > 0.0
        and positive_relative >= 2
        and minimum_relative >= -0.05
        and better_raw_folds >= 2
        and aggregate_better_raw
        and ece_improved
        and mean_ece <= 0.15
    )
    return {
        "support_requirements_pass": support_pass,
        "mean_roc_auc": mean_auc,
        "positive_auc_folds": positive_auc,
        "minimum_fold_roc_auc": minimum_auc,
        "mean_relative_brier_improvement": mean_relative,
        "positive_relative_brier_folds": positive_relative,
        "minimum_fold_relative_brier_improvement": minimum_relative,
        "brier_better_than_raw_stab001_folds": better_raw_folds,
        "aggregate_brier_better_than_raw_stab001": aggregate_better_raw,
        "mean_expected_calibration_error": mean_ece,
        "raw_stab001_mean_expected_calibration_error": raw_mean_ece,
        "mean_ece_improved_vs_raw_stab001": ece_improved,
        "sample_hashes_match_stab001": bool(metrics["sample_hash_matches_stab001"].all()),
        "positive_platt_slope_folds": int(metrics["platt_positive_slope"].sum()),
        "viability_gate_pass": viable,
    }


def run() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    folds = config.get("validation", {}).get("folds", [])
    if len(folds) != 3:
        raise ValueError("STAB-002 requires the frozen three chronological folds")

    frame = pd.read_parquet(DATASET, engine="pyarrow").copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
    frame = frame.sort_values("decision_date").reset_index(drop=True)
    frame = add_opportunity_targets(frame)
    feature_version, features = load_feature_registry(REGISTRY)
    validate_feature_columns(frame, features)
    if feature_version != manifest.get("feature_set_version"):
        raise ValueError("STAB-002 feature registry version does not match manifest")
    if len(features) != int(manifest.get("feature_count", -1)):
        raise ValueError("STAB-002 feature count does not match manifest")

    expected_hashes = _expected_sample_hashes()
    metric_rows: list[dict[str, Any]] = []
    inner_frames: list[pd.DataFrame] = []

    for fold in folds:
        fold_name = str(fold["name"])
        outer_train = frame.loc[eligible_training_mask(frame, fold["train_end"])].copy()
        test = frame.loc[realized_test_mask(frame, fold)].copy()
        if test.empty:
            raise ValueError(f"STAB-002 fold {fold_name} has no mature test rows")

        oof, inner_support, inner_support_ok = build_nested_oof_calibration(
            frame, outer_train, features, outer_fold=fold_name
        )
        inner_frames.append(inner_support)
        calibrator, slope, intercept, positive_slope = fit_platt(oof)

        selected, _ = select_stable_features(outer_train, features)
        outer_train_raw = stability_raw_score(outer_train, outer_train, selected)
        test_raw = stability_raw_score(outer_train, test, selected)
        raw_stab_probability = score_to_probability(outer_train_raw, test_raw)

        base_rate = float(outer_train["favorable_entry_20d"].astype(int).mean())
        if positive_slope and inner_support_ok:
            probability = _probability(calibrator, test_raw)
        else:
            probability = np.full(len(test), base_rate, dtype=float)

        y_test = test["favorable_entry_20d"].astype(int).to_numpy()
        baseline_probability = np.full(len(test), base_rate, dtype=float)
        brier = float(np.mean((probability - y_test) ** 2))
        baseline_brier = float(np.mean((baseline_probability - y_test) ** 2))
        relative_brier = (
            float((baseline_brier - brier) / baseline_brier)
            if baseline_brier > 0
            else float("nan")
        )
        raw_brier = float(np.mean((raw_stab_probability - y_test) ** 2))
        auc = float(roc_auc_score(y_test, probability)) if positive_slope and inner_support_ok else 0.5
        ece = expected_calibration_error(y_test, probability)
        raw_ece = expected_calibration_error(y_test, raw_stab_probability)
        date_hash = sample_hash(test["decision_date"])
        expected_hash = expected_hashes.get(fold_name)
        if expected_hash is None:
            raise ValueError(f"Missing STAB-001 expected sample hash for {fold_name}")

        metric_rows.append(
            {
                "method_id": METHOD_ID,
                "fold": fold_name,
                "training_rows": int(len(outer_train)),
                "calibration_rows": int(len(oof)),
                "test_rows": int(len(test)),
                "sample_sha256": date_hash,
                "stab001_sample_sha256": expected_hash,
                "sample_hash_matches_stab001": bool(date_hash == expected_hash),
                "training_prevalence": base_rate,
                "test_prevalence": float(np.mean(y_test)),
                "outer_selected_feature_count": int(len(selected)),
                "outer_selected_family_count": int(len({feature_family(str(item["feature"])) for item in selected})),
                "inner_support_pass": bool(inner_support_ok),
                "platt_slope": slope,
                "platt_intercept": intercept,
                "platt_positive_slope": positive_slope,
                "brier_score": brier,
                "baseline_brier_score": baseline_brier,
                "relative_brier_improvement": relative_brier,
                "raw_stab001_brier_score": raw_brier,
                "brier_improvement_vs_raw_stab001": float(raw_brier - brier),
                "expected_calibration_error": ece,
                "raw_stab001_expected_calibration_error": raw_ece,
                "roc_auc": auc,
            }
        )

    metrics = pd.DataFrame(metric_rows)
    inner = pd.concat(inner_frames, ignore_index=True)
    viability = summarize(metrics, inner)
    decision = (
        "CAUSAL_CALIBRATION_WORTH_EXP_010"
        if viability["viability_gate_pass"]
        else "DO_NOT_ADVANCE_CAUSAL_CALIBRATION_UNDER_STAB_002"
    )
    report = {
        "method_id": METHOD_ID,
        "as_of": AS_OF.strftime("%Y-%m-%d"),
        "status": "STAB_002_EVALUATION_COMPLETE",
        "feature_version": feature_version,
        "feature_count": len(features),
        "dataset_sha256": _sha256(DATASET),
        "target_source_experiment": "EXP-006",
        "target": "favorable_entry_20d",
        "ranking_source_method": "STAB-001",
        "calibration_protocol": {
            "calibration_history_rows": CALIBRATION_ROWS,
            "calibration_blocks": CALIBRATION_BLOCKS,
            "rows_per_block": CALIBRATION_BLOCK_ROWS,
            "minimum_inner_formation_rows": MIN_INNER_FORMATION_ROWS,
            "platt_model": PLATT_PARAMS,
            "positive_slope_required": True,
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
        "note": "STAB-002 changes only causal calibration of the frozen STAB-001 ranking methodology. It cannot promote a champion or change sizing, and exposed outer folds are development evidence only.",
    }

    EVALUATION.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS, index=False, lineterminator="\n")
    inner.to_csv(INNER, index=False, lineterminator="\n")
    EVALUATION.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nSTAB-002 fold metrics:")
    print(metrics.to_csv(index=False, lineterminator="\n"))
    return report


if __name__ == "__main__":
    run()
