#!/usr/bin/env python3
"""Verify frozen EXP-009 evidence and semantically reproduce the experiment."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "v3" / "experiments" / "EXP-009" / "manifest.json"
EVALUATION = ROOT / "v3" / "reports" / "exp009_evaluation.json"
METRICS = ROOT / "v3" / "reports" / "exp009_metrics.csv"
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


def compare_frame(current: pd.DataFrame, frozen: pd.DataFrame) -> None:
    require(set(current.columns) == set(frozen.columns), "EXP-009 metrics column drift")
    current = current[frozen.columns].sort_values("fold").reset_index(drop=True)
    frozen = frozen.sort_values("fold").reset_index(drop=True)
    require(len(current) == len(frozen) == 3, "EXP-009 metrics row-count drift")
    for column in frozen.columns:
        if pd.api.types.is_numeric_dtype(frozen[column]):
            left = pd.to_numeric(current[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(frozen[column], errors="coerce").to_numpy(float)
            require(
                np.allclose(left, right, rtol=0.0, atol=ATOL, equal_nan=True),
                f"EXP-009 numeric drift in {column}",
            )
        else:
            left = current[column].fillna("<NA>").astype(str).tolist()
            right = frozen[column].fillna("<NA>").astype(str).tolist()
            require(left == right, f"EXP-009 discrete drift in {column}")


def compare_evaluation(current: dict, frozen: dict) -> None:
    # Future post-cutoff rows may change only the generated whole-dataset hash.
    # Frozen samples, metrics, adaptation contract, and decision remain strict.
    exact_keys = (
        "experiment_id", "as_of", "status", "feature_version", "feature_count",
        "target_source_experiment", "model_source_experiment",
        "model_name", "adaptation", "experiment_viability_pass", "decision",
        "champion_selected", "v3_019_eligible", "current_sizing_multiplier",
        "champion_gate_version",
    )
    for key in exact_keys:
        require(current.get(key) == frozen.get(key), f"EXP-009 evaluation drift in {key}")

    current_viability = current.get("viability", {})
    frozen_viability = frozen.get("viability", {})
    require(set(current_viability) == set(frozen_viability), "EXP-009 viability keys drift")
    for key, frozen_value in frozen_viability.items():
        current_value = current_viability[key]
        if isinstance(frozen_value, bool) or isinstance(frozen_value, int):
            require(current_value == frozen_value, f"EXP-009 viability drift in {key}")
        else:
            require(
                np.isclose(float(current_value), float(frozen_value), rtol=0.0, atol=ATOL),
                f"EXP-009 viability numeric drift in {key}",
            )


def main() -> int:
    for path in (MANIFEST, EVALUATION, METRICS):
        require(path.exists(), f"Missing EXP-009 artifact: {path}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    current_evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    current_metrics = pd.read_csv(METRICS)

    frozen_evaluation_bytes = head_bytes(EVALUATION)
    frozen_metrics_bytes = head_bytes(METRICS)
    frozen_evaluation = json.loads(frozen_evaluation_bytes.decode("utf-8"))
    frozen_metrics = pd.read_csv(io.BytesIO(frozen_metrics_bytes))

    require(manifest.get("experiment_id") == "EXP-009", "EXP-009 id drift")
    require(manifest.get("status") == "complete_reject", "EXP-009 status drift")
    require(manifest.get("feature_set_version") == "v3-features-004-treasury", "EXP-009 feature version drift")
    require(manifest.get("feature_count") == 53, "EXP-009 feature count drift")
    require(manifest.get("adaptation", {}).get("window_eligible_rows") == 504, "EXP-009 window drift")
    require(manifest.get("adaptation", {}).get("alternative_windows_allowed_under_same_experiment") is False, "EXP-009 window search must remain disabled")
    require(manifest.get("tuning_after_result_allowed_under_same_version") is False, "EXP-009 post-result tuning must remain disabled")

    evidence = manifest.get("evidence", {})
    require(evidence.get("v3/reports/exp009_evaluation.json") == sha256_bytes(frozen_evaluation_bytes), "EXP-009 committed evaluation hash mismatch")
    require(evidence.get("v3/reports/exp009_metrics.csv") == sha256_bytes(frozen_metrics_bytes), "EXP-009 committed metrics hash mismatch")

    compare_evaluation(current_evaluation, frozen_evaluation)
    compare_frame(current_metrics, frozen_metrics)

    require(current_evaluation.get("decision") == "DO_NOT_ADVANCE_RECENT_WINDOW_UNDER_EXP_009", "EXP-009 decision drift")
    require(current_evaluation.get("experiment_viability_pass") is False, "EXP-009 unexpectedly passed")
    require(current_evaluation.get("v3_019_eligible") is False, "EXP-009 must not unlock V3-019")
    require(float(current_evaluation.get("current_sizing_multiplier")) == 1.0, "EXP-009 sizing changed")

    viability = current_evaluation.get("viability", {})
    require(viability.get("exact_504_training_rows_each_fold") is True, "EXP-009 504-row contract changed")
    require(viability.get("sample_hashes_match_exp006") is True, "EXP-009 sample hashes changed")
    require(int(viability.get("positive_relative_brier_folds", -1)) == 0, "EXP-009 Brier conclusion changed")
    require(int(viability.get("brier_better_than_full_history_folds", -1)) == 0, "EXP-009 full-history Brier conclusion changed")
    require(int(viability.get("auc_better_than_full_history_folds", -1)) == 2, "EXP-009 full-history AUC conclusion changed")

    print("EXP-009 immutable bytes + semantic reproduction: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
