#!/usr/bin/env python3
"""Verify DIAG-001 immutable evidence and semantic reproducibility."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "v3" / "diagnostics" / "diag001_manifest.json"
REGISTRY = ROOT / "v3" / "reports" / "feature_registry_treasury.json"
TARGET_DRIFT = ROOT / "v3" / "reports" / "diag001_target_drift.csv"
FEATURE_DRIFT = ROOT / "v3" / "reports" / "diag001_feature_drift.csv"
ASSOCIATION = ROOT / "v3" / "reports" / "diag001_feature_target_association.csv"
STABILITY = ROOT / "v3" / "reports" / "diag001_feature_stability.csv"
SUMMARY = ROOT / "v3" / "reports" / "diag001_summary.json"
ATOL = 1e-8
FOLDS = ("2024", "2025", "2026_ytd")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def head_bytes(path: Path) -> bytes:
    relative = path.relative_to(ROOT).as_posix()
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def compare_value(current: Any, frozen: Any, label: str) -> None:
    if isinstance(frozen, bool) or isinstance(frozen, int):
        require(current == frozen, f"{label} drift: {current!r} != {frozen!r}")
        return
    if isinstance(frozen, float):
        require(
            np.isclose(float(current), float(frozen), rtol=0.0, atol=ATOL, equal_nan=True),
            f"{label} numeric drift: {current!r} != {frozen!r}",
        )
        return
    require(current == frozen, f"{label} drift: {current!r} != {frozen!r}")


def compare_summary(current: dict[str, Any], frozen: dict[str, Any]) -> None:
    # Whole generated-dataset hashes may change when rows strictly after the
    # frozen 2026-08-18 research cutoff are appended. The frozen fold samples,
    # complete 53x3 diagnostic tables, counts, and conclusions remain strict.
    exact_keys = (
        "diagnostic_id",
        "status",
        "as_of",
        "feature_version",
        "feature_count",
        "target",
        "folds",
        "sample_hashes_match_exp006",
        "feature_drift_rows",
        "association_rows",
        "stability_rows",
        "features_with_any_training_to_test_sign_reversal",
        "features_with_any_adjacent_test_fold_sign_transition",
        "minimum_association_rows",
        "model_fitted",
        "champion_selected",
        "v3_019_eligible",
        "current_sizing_multiplier",
    )
    for key in exact_keys:
        compare_value(current.get(key), frozen.get(key), f"DIAG-001 summary.{key}")

    current_prevalence = current.get("target_prevalence", {})
    frozen_prevalence = frozen.get("target_prevalence", {})
    require(set(current_prevalence) == set(frozen_prevalence) == set(FOLDS), "DIAG-001 target prevalence fold drift")
    for fold in FOLDS:
        require(set(current_prevalence[fold]) == set(frozen_prevalence[fold]), f"DIAG-001 {fold} target prevalence keys drift")
        for key, frozen_value in frozen_prevalence[fold].items():
            compare_value(current_prevalence[fold][key], frozen_value, f"DIAG-001 target_prevalence.{fold}.{key}")


def compare_frame(
    current: pd.DataFrame,
    frozen: pd.DataFrame,
    *,
    keys: list[str],
    label: str,
) -> None:
    require(list(current.columns) == list(frozen.columns), f"{label} column order/set drift")
    current = current.sort_values(keys, kind="stable").reset_index(drop=True)
    frozen = frozen.sort_values(keys, kind="stable").reset_index(drop=True)
    require(len(current) == len(frozen), f"{label} row-count drift")

    for column in frozen.columns:
        frozen_series = frozen[column]
        current_series = current[column]
        if pd.api.types.is_bool_dtype(frozen_series.dtype):
            left = current_series.astype("boolean").fillna(False).tolist()
            right = frozen_series.astype("boolean").fillna(False).tolist()
            require(left == right, f"{label} boolean drift in {column}")
        elif pd.api.types.is_numeric_dtype(frozen_series.dtype):
            left = pd.to_numeric(current_series, errors="coerce").to_numpy(float)
            right = pd.to_numeric(frozen_series, errors="coerce").to_numpy(float)
            require(
                np.allclose(left, right, rtol=0.0, atol=ATOL, equal_nan=True),
                f"{label} numeric drift in {column}",
            )
        else:
            left = current_series.fillna("<NA>").astype(str).tolist()
            right = frozen_series.fillna("<NA>").astype(str).tolist()
            require(left == right, f"{label} discrete drift in {column}")


def load_head_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(head_bytes(path)))


def main() -> int:
    evidence_paths = (TARGET_DRIFT, FEATURE_DRIFT, ASSOCIATION, STABILITY, SUMMARY)
    for path in (MANIFEST, REGISTRY, *evidence_paths):
        require(path.exists(), f"Missing DIAG-001 artifact: {path}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(manifest.get("diagnostic_id") == "DIAG-001", "DIAG-001 manifest id drift")
    require(manifest.get("status") == "complete", "DIAG-001 manifest status drift")
    require(manifest.get("pre_registered_issue") == 55, "DIAG-001 issue lineage drift")
    require(manifest.get("research_cutoff") == "2026-08-18", "DIAG-001 cutoff drift")
    require(manifest.get("feature_set_version") == "v3-features-004-treasury", "DIAG-001 feature version drift")
    require(manifest.get("feature_count") == 53, "DIAG-001 feature count drift")
    require(manifest.get("minimum_association_rows") == 20, "DIAG-001 association row contract drift")
    require(manifest.get("diagnostic_only") is True, "DIAG-001 must remain diagnostic-only")
    require(manifest.get("model_fitted") is False, "DIAG-001 must not fit a model")
    require(manifest.get("feature_selection_permitted") is False, "DIAG-001 must not select features")
    require(manifest.get("champion_selected") is False, "DIAG-001 must not select a champion")
    require(manifest.get("v3_019_eligible") is False, "DIAG-001 must not unlock V3-019")
    require(float(manifest.get("current_sizing_multiplier")) == 1.0, "DIAG-001 must not change sizing")

    governance = manifest.get("governance", {})
    require(governance.get("outer_folds_exposed_to_research") is True, "DIAG-001 exposed-fold governance drift")
    require(governance.get("outer_fold_periods_are_final_promotion_evidence") is False, "DIAG-001 must not reuse exposed folds as final promotion evidence")
    require(governance.get("future_feature_or_model_decisions_require_new_pre_registration") is True, "DIAG-001 future pre-registration contract drift")
    require(governance.get("fresh_promotion_evidence_required") is True, "DIAG-001 fresh-evidence requirement drift")

    evidence = manifest.get("evidence", {})
    require(len(evidence) == 5, "DIAG-001 manifest must hash all five evidence files")
    for path in evidence_paths:
        relative = path.relative_to(ROOT).as_posix()
        frozen_payload = head_bytes(path)
        require(
            evidence.get(relative) == sha256_bytes(frozen_payload),
            f"DIAG-001 committed evidence hash mismatch: {relative}",
        )

    current_summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    frozen_summary = json.loads(head_bytes(SUMMARY).decode("utf-8"))
    compare_summary(current_summary, frozen_summary)

    current_target = pd.read_csv(TARGET_DRIFT)
    current_drift = pd.read_csv(FEATURE_DRIFT)
    current_assoc = pd.read_csv(ASSOCIATION)
    current_stability = pd.read_csv(STABILITY)
    frozen_target = load_head_csv(TARGET_DRIFT)
    frozen_drift = load_head_csv(FEATURE_DRIFT)
    frozen_assoc = load_head_csv(ASSOCIATION)
    frozen_stability = load_head_csv(STABILITY)

    compare_frame(current_target, frozen_target, keys=["fold"], label="DIAG-001 target drift")
    compare_frame(current_drift, frozen_drift, keys=["feature", "fold"], label="DIAG-001 feature drift")
    compare_frame(current_assoc, frozen_assoc, keys=["feature", "fold"], label="DIAG-001 association")
    compare_frame(current_stability, frozen_stability, keys=["feature"], label="DIAG-001 stability")

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    registered = [str(item["name"]) for item in registry.get("features", [])]
    require(len(registered) == 53 and len(set(registered)) == 53, "Treasury registry feature count drift")
    registered_set = set(registered)

    require(len(current_target) == 3, "DIAG-001 target drift must have 3 folds")
    require(set(current_target["fold"].astype(str)) == set(FOLDS), "DIAG-001 target fold set drift")
    require(current_target["sample_hash_matches_exp006"].astype(bool).all(), "DIAG-001 target samples no longer match EXP-006")

    for frame, label in ((current_drift, "feature drift"), (current_assoc, "association")):
        require(len(frame) == 159, f"DIAG-001 {label} must preserve 53x3 rows")
        require(set(frame["feature"].astype(str)) == registered_set, f"DIAG-001 {label} feature set drift")
        counts = frame.groupby("feature")["fold"].nunique()
        require(counts.eq(3).all(), f"DIAG-001 {label} does not preserve all folds per feature")

    require(len(current_stability) == 53, "DIAG-001 stability must preserve 53 rows")
    require(set(current_stability["feature"].astype(str)) == registered_set, "DIAG-001 stability feature set drift")
    require(
        int(current_stability["training_to_test_sign_reversals"].gt(0).sum()) == 42,
        "DIAG-001 training-to-test reversal count drift",
    )
    require(
        int(current_stability["adjacent_test_fold_sign_transitions"].gt(0).sum()) == 34,
        "DIAG-001 adjacent test-fold sign-transition count drift",
    )

    require(current_summary.get("model_fitted") is False, "DIAG-001 summary model state drift")
    require(current_summary.get("champion_selected") is False, "DIAG-001 summary champion state drift")
    require(current_summary.get("v3_019_eligible") is False, "DIAG-001 summary V3-019 state drift")
    require(float(current_summary.get("current_sizing_multiplier")) == 1.0, "DIAG-001 summary sizing drift")

    print("DIAG-001 immutable bytes + semantic reproduction: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
