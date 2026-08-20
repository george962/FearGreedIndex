#!/usr/bin/env python3
"""Verify STAB-002 immutable evidence and semantic reproduction."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "v3" / "methodology" / "STAB-002" / "manifest.json"
EVALUATION = ROOT / "v3" / "reports" / "stab002_evaluation.json"
METRICS = ROOT / "v3" / "reports" / "stab002_metrics.csv"
INNER = ROOT / "v3" / "reports" / "stab002_inner_calibration.csv"
CHECKPOINTS = ROOT / "v3" / "evidence" / "forward_checkpoints.json"
ATOL = 1e-8


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def head_bytes(path: Path) -> bytes:
    relative = path.relative_to(ROOT).as_posix()
    return subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=ROOT, check=True, capture_output=True
    ).stdout


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def compare_frame(current: pd.DataFrame, frozen: pd.DataFrame, name: str) -> None:
    require(list(current.columns) == list(frozen.columns), f"{name} columns drift")
    require(len(current) == len(frozen), f"{name} row count drift")
    for column in frozen.columns:
        if pd.api.types.is_numeric_dtype(frozen[column]):
            left = pd.to_numeric(current[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(frozen[column], errors="coerce").to_numpy(float)
            require(np.allclose(left, right, rtol=0.0, atol=ATOL, equal_nan=True), f"{name} numeric drift: {column}")
        else:
            left = current[column].fillna("<NA>").astype(str).tolist()
            right = frozen[column].fillna("<NA>").astype(str).tolist()
            require(left == right, f"{name} discrete drift: {column}")


def compare_evaluation(current: dict, frozen: dict) -> None:
    # Whole generated-dataset hashes are intentionally excluded from semantic
    # reproduction. Appending rows strictly after the frozen research cutoff is
    # allowed; the historical sample hashes, calibration evidence, metrics,
    # decisions, and governance state must still reproduce exactly.
    exact = (
        "method_id", "as_of", "status", "feature_version", "feature_count",
        "target_source_experiment", "target", "ranking_source_method",
        "calibration_protocol", "method_viability_pass", "decision",
        "development_evidence_only", "research_exposed_periods", "evid001_outcomes_opened",
        "champion_selected", "v3_019_eligible", "current_sizing_multiplier",
    )
    for key in exact:
        require(current.get(key) == frozen.get(key), f"STAB-002 evaluation drift: {key}")
    cv = current.get("viability", {})
    fv = frozen.get("viability", {})
    require(set(cv) == set(fv), "STAB-002 viability keys drift")
    for key, value in fv.items():
        if isinstance(value, (bool, int)):
            require(cv[key] == value, f"STAB-002 viability drift: {key}")
        else:
            require(np.isclose(float(cv[key]), float(value), rtol=0.0, atol=ATOL), f"STAB-002 viability numeric drift: {key}")


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(manifest.get("method_id") == "STAB-002", "STAB-002 id drift")
    require(manifest.get("status") == "complete_reject", "STAB-002 status drift")
    require(manifest.get("ranking_source_method") == "STAB-001", "STAB-002 ranking lineage drift")
    require(manifest.get("tuning_after_result_allowed_under_same_version") is False, "STAB-002 tuning must remain disabled")

    frozen_eval_bytes = head_bytes(EVALUATION)
    frozen_metrics_bytes = head_bytes(METRICS)
    frozen_inner_bytes = head_bytes(INNER)
    evidence = manifest.get("evidence", {})
    require(evidence.get("v3/reports/stab002_evaluation.json") == sha(frozen_eval_bytes), "STAB-002 evaluation hash mismatch")
    require(evidence.get("v3/reports/stab002_metrics.csv") == sha(frozen_metrics_bytes), "STAB-002 metrics hash mismatch")
    require(evidence.get("v3/reports/stab002_inner_calibration.csv") == sha(frozen_inner_bytes), "STAB-002 inner evidence hash mismatch")

    current_eval = json.loads(EVALUATION.read_text(encoding="utf-8"))
    frozen_eval = json.loads(frozen_eval_bytes.decode("utf-8"))
    compare_evaluation(current_eval, frozen_eval)
    compare_frame(pd.read_csv(METRICS), pd.read_csv(io.BytesIO(frozen_metrics_bytes)), "STAB-002 metrics")
    compare_frame(pd.read_csv(INNER), pd.read_csv(io.BytesIO(frozen_inner_bytes)), "STAB-002 inner calibration")

    require(current_eval.get("decision") == "DO_NOT_ADVANCE_CAUSAL_CALIBRATION_UNDER_STAB_002", "STAB-002 decision drift")
    require(current_eval.get("method_viability_pass") is False, "STAB-002 unexpectedly passed")
    viability = current_eval.get("viability", {})
    require(int(viability.get("positive_platt_slope_folds", -1)) == 1, "STAB-002 slope conclusion changed")
    require(int(viability.get("positive_relative_brier_folds", -1)) == 0, "STAB-002 relative-Brier conclusion changed")
    require(bool(viability.get("aggregate_brier_better_than_raw_stab001")) is True, "STAB-002 raw-Brier comparison changed")
    require(bool(viability.get("mean_ece_improved_vs_raw_stab001")) is True, "STAB-002 ECE comparison changed")
    require(bool(viability.get("sample_hashes_match_stab001")) is True, "STAB-002 sample lineage changed")
    require(current_eval.get("evid001_outcomes_opened") is False, "STAB-002 opened EVID-001")
    require(current_eval.get("v3_019_eligible") is False, "STAB-002 unlocked V3-019")
    require(float(current_eval.get("current_sizing_multiplier")) == 1.0, "STAB-002 sizing changed")

    checkpoints = json.loads(CHECKPOINTS.read_text(encoding="utf-8"))
    require(checkpoints.get("checkpoints") == [], "EVID-001 checkpoint outcomes were opened")
    print("STAB-002 immutable evidence + semantic reproduction: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
