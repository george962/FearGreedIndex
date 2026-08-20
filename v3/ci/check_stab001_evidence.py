#!/usr/bin/env python3
"""Verify STAB-001 immutable evidence and semantic reproduction."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "v3" / "methodology" / "STAB-001" / "manifest.json"
EVALUATION = ROOT / "v3" / "reports" / "stab001_evaluation.json"
METRICS = ROOT / "v3" / "reports" / "stab001_metrics.csv"
SELECTED = ROOT / "v3" / "reports" / "stab001_selected_features.csv"
CHECKPOINTS = ROOT / "v3" / "evidence" / "forward_checkpoints.json"
ATOL = 1e-8


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def head_bytes(path: Path) -> bytes:
    relative = path.relative_to(ROOT).as_posix()
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=ROOT, check=True, capture_output=True
    )
    return completed.stdout


def compare_frame(current: pd.DataFrame, frozen: pd.DataFrame, *, name: str) -> None:
    require(list(current.columns) == list(frozen.columns), f"{name} column drift")
    require(len(current) == len(frozen), f"{name} row-count drift")
    for column in frozen.columns:
        if pd.api.types.is_numeric_dtype(frozen[column]):
            left = pd.to_numeric(current[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(frozen[column], errors="coerce").to_numpy(float)
            require(
                np.allclose(left, right, rtol=0.0, atol=ATOL, equal_nan=True),
                f"{name} numeric drift in {column}",
            )
        else:
            left = current[column].fillna("<NA>").astype(str).tolist()
            right = frozen[column].fillna("<NA>").astype(str).tolist()
            require(left == right, f"{name} discrete drift in {column}")


def compare_evaluation(current: dict, frozen: dict) -> None:
    # dataset_sha256 intentionally is not compared here. The generated research
    # dataset may gain rows strictly after the frozen 2026-08-18 cutoff. Those
    # future rows must not invalidate a historical experiment when its mature
    # test sample hashes, metrics, selected-feature diagnostics, and decisions
    # reproduce exactly.
    exact_keys = (
        "method_id",
        "as_of",
        "status",
        "feature_version",
        "feature_count",
        "target_source_experiment",
        "target",
        "selection_rule",
        "method_viability_pass",
        "decision",
        "development_evidence_only",
        "research_exposed_periods",
        "evid001_outcomes_opened",
        "champion_selected",
        "v3_019_eligible",
        "current_sizing_multiplier",
    )
    for key in exact_keys:
        require(current.get(key) == frozen.get(key), f"STAB-001 evaluation drift in {key}")

    current_viability = current.get("viability", {})
    frozen_viability = frozen.get("viability", {})
    require(set(current_viability) == set(frozen_viability), "STAB-001 viability keys drift")
    for key, frozen_value in frozen_viability.items():
        current_value = current_viability[key]
        if isinstance(frozen_value, (bool, int)):
            require(current_value == frozen_value, f"STAB-001 viability drift in {key}")
        else:
            require(
                np.isclose(float(current_value), float(frozen_value), rtol=0.0, atol=ATOL),
                f"STAB-001 viability numeric drift in {key}",
            )


def main() -> int:
    for path in (MANIFEST, EVALUATION, METRICS, SELECTED, CHECKPOINTS):
        require(path.exists(), f"Missing STAB-001 artifact: {path}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(manifest.get("method_id") == "STAB-001", "STAB-001 id drift")
    require(manifest.get("status") == "complete_reject", "STAB-001 status drift")
    require(manifest.get("feature_set_version") == "v3-features-004-treasury", "STAB-001 feature version drift")
    require(manifest.get("feature_count") == 53, "STAB-001 feature count drift")
    require(manifest.get("tuning_after_result_allowed_under_same_version") is False, "STAB-001 post-result tuning must remain disabled")

    current_evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    current_metrics = pd.read_csv(METRICS)
    current_selected = pd.read_csv(SELECTED)

    frozen_evaluation_bytes = head_bytes(EVALUATION)
    frozen_metrics_bytes = head_bytes(METRICS)
    frozen_selected_bytes = head_bytes(SELECTED)
    frozen_evaluation = json.loads(frozen_evaluation_bytes.decode("utf-8"))
    frozen_metrics = pd.read_csv(io.BytesIO(frozen_metrics_bytes))
    frozen_selected = pd.read_csv(io.BytesIO(frozen_selected_bytes))

    evidence = manifest.get("evidence", {})
    require(
        evidence.get("v3/reports/stab001_evaluation.json") == sha256_bytes(frozen_evaluation_bytes),
        "STAB-001 committed evaluation hash mismatch",
    )
    require(
        evidence.get("v3/reports/stab001_metrics.csv") == sha256_bytes(frozen_metrics_bytes),
        "STAB-001 committed metrics hash mismatch",
    )
    require(
        evidence.get("v3/reports/stab001_selected_features.csv") == sha256_bytes(frozen_selected_bytes),
        "STAB-001 committed selected-feature hash mismatch",
    )

    compare_evaluation(current_evaluation, frozen_evaluation)
    compare_frame(current_metrics, frozen_metrics, name="STAB-001 metrics")
    compare_frame(current_selected, frozen_selected, name="STAB-001 selected diagnostics")

    require(
        current_evaluation.get("decision") == "DO_NOT_ADVANCE_STABILITY_SELECTOR_UNDER_STAB_001",
        "STAB-001 decision drift",
    )
    require(current_evaluation.get("method_viability_pass") is False, "STAB-001 unexpectedly passed")
    require(current_evaluation.get("evid001_outcomes_opened") is False, "STAB-001 opened EVID-001")
    require(current_evaluation.get("v3_019_eligible") is False, "STAB-001 must not unlock V3-019")
    require(float(current_evaluation.get("current_sizing_multiplier")) == 1.0, "STAB-001 sizing changed")

    viability = current_evaluation.get("viability", {})
    require(int(viability.get("positive_auc_folds", -1)) == 3, "STAB-001 ranking conclusion changed")
    require(int(viability.get("positive_relative_brier_folds", -1)) == 0, "STAB-001 Brier conclusion changed")
    require(bool(viability.get("sample_hashes_match_exp006")) is True, "STAB-001 sample hashes changed")

    checkpoints = json.loads(CHECKPOINTS.read_text(encoding="utf-8"))
    require(checkpoints.get("checkpoints") == [], "EVID-001 checkpoint outcomes were opened")

    print("STAB-001 immutable evidence + semantic reproduction: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
