#!/usr/bin/env python3
"""Verify frozen EXP-006 evidence and semantically reproduce the experiment.

The committed evidence bytes remain immutable and SHA-256 protected. A fresh
rerun may differ by tiny floating-point serialization noise, so rerun outputs
are compared against the committed evidence with a tight numeric tolerance
while all identifiers, samples, decisions, gates, and discrete contracts remain
exact.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "v3" / "experiments" / "EXP-006" / "manifest.json"
EVALUATION = ROOT / "v3" / "reports" / "exp006_evaluation.json"
METRICS = ROOT / "v3" / "reports" / "exp006_metrics.csv"
STATES = ROOT / "v3" / "reports" / "exp006_state_distribution.csv"
ATOL = 1e-8


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


def head_json(path: Path) -> dict:
    return json.loads(head_bytes(path).decode("utf-8"))


def head_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(head_bytes(path)))


def compare_frame(current: pd.DataFrame, frozen: pd.DataFrame, keys: list[str], label: str) -> None:
    require(set(current.columns) == set(frozen.columns), f"{label} column set drift")
    current = current[frozen.columns].sort_values(keys).reset_index(drop=True)
    frozen = frozen.sort_values(keys).reset_index(drop=True)
    require(len(current) == len(frozen), f"{label} row-count drift")

    for column in frozen.columns:
        if pd.api.types.is_numeric_dtype(frozen[column]):
            left = pd.to_numeric(current[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(frozen[column], errors="coerce").to_numpy(float)
            require(
                np.allclose(left, right, rtol=0.0, atol=ATOL, equal_nan=True),
                f"{label} numeric drift in {column}",
            )
        else:
            left = current[column].fillna("<NA>").astype(str).tolist()
            right = frozen[column].fillna("<NA>").astype(str).tolist()
            require(left == right, f"{label} discrete drift in {column}")


def compare_evaluation(current: dict, frozen: dict) -> None:
    exact_keys = (
        "experiment_id",
        "as_of",
        "status",
        "feature_version",
        "feature_count",
        "dataset_sha256",
        "sample_hashes_match",
        "experiment_viability_pass",
        "decision",
        "champion_selected",
        "v3_019_eligible",
        "current_sizing_multiplier",
        "champion_gate_version",
        "target",
    )
    for key in exact_keys:
        require(current.get(key) == frozen.get(key), f"EXP-006 evaluation drift in {key}")

    require(set(current.get("models", {})) == set(frozen.get("models", {})), "EXP-006 model summary set drift")
    for model_name, frozen_summary in frozen["models"].items():
        current_summary = current["models"][model_name]
        require(set(current_summary) == set(frozen_summary), f"EXP-006 {model_name} summary keys drift")
        for key, frozen_value in frozen_summary.items():
            current_value = current_summary[key]
            if isinstance(frozen_value, bool) or isinstance(frozen_value, int):
                require(current_value == frozen_value, f"EXP-006 {model_name}.{key} drift")
            elif isinstance(frozen_value, float):
                require(
                    np.isclose(float(current_value), frozen_value, rtol=0.0, atol=ATOL),
                    f"EXP-006 {model_name}.{key} numeric drift",
                )
            else:
                require(current_value == frozen_value, f"EXP-006 {model_name}.{key} drift")


def main() -> int:
    for path in (MANIFEST, EVALUATION, METRICS, STATES):
        require(path.exists(), f"Missing EXP-006 artifact: {path}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    current_evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    current_metrics = pd.read_csv(METRICS)
    current_states = pd.read_csv(STATES)

    frozen_evaluation_bytes = head_bytes(EVALUATION)
    frozen_metrics_bytes = head_bytes(METRICS)
    frozen_states_bytes = head_bytes(STATES)
    frozen_evaluation = json.loads(frozen_evaluation_bytes.decode("utf-8"))
    frozen_metrics = pd.read_csv(io.BytesIO(frozen_metrics_bytes))
    frozen_states = pd.read_csv(io.BytesIO(frozen_states_bytes))

    require(manifest.get("experiment_id") == "EXP-006", "EXP-006 id drift")
    require(manifest.get("status") == "completed_negative_experiment", "EXP-006 status drift")
    require(manifest.get("feature_set_version") == "v3-features-004-treasury", "EXP-006 feature version drift")
    require(manifest.get("feature_count") == 53, "EXP-006 feature count drift")
    require(manifest.get("tuning_after_result_allowed_under_same_version") is False, "EXP-006 post-result tuning must remain disabled")

    target = manifest.get("target", {})
    require(target.get("return_minimum") == 0.02, "EXP-006 target return threshold drift")
    require(target.get("drawdown_strictly_greater_than") == -0.05, "EXP-006 drawdown threshold drift")
    require(target.get("ordinal_bad_return_maximum") == -0.02, "EXP-006 BAD return threshold drift")
    require(target.get("ordinal_excellent_return_minimum") == 0.05, "EXP-006 EXCELLENT threshold drift")

    gate = manifest.get("viability_gate", {})
    require(gate.get("minimum_positive_relative_brier_folds") == 2, "EXP-006 Brier-fold gate drift")
    require(gate.get("minimum_mean_roc_auc_exclusive") == 0.52, "EXP-006 mean-AUC gate drift")
    require(gate.get("minimum_positive_auc_folds") == 2, "EXP-006 AUC-fold gate drift")
    require(gate.get("require_identical_test_sample_hashes") is True, "EXP-006 sample-hash gate drift")

    evidence = manifest.get("evidence", {})
    require(evidence.get("evaluation_sha256") == sha256_bytes(frozen_evaluation_bytes), "EXP-006 committed evaluation hash mismatch")
    require(evidence.get("metrics_sha256") == sha256_bytes(frozen_metrics_bytes), "EXP-006 committed metrics hash mismatch")
    require(evidence.get("state_distribution_sha256") == sha256_bytes(frozen_states_bytes), "EXP-006 committed state-distribution hash mismatch")

    compare_evaluation(current_evaluation, frozen_evaluation)
    compare_frame(current_metrics, frozen_metrics, ["model_name", "fold"], "EXP-006 metrics")
    compare_frame(current_states, frozen_states, ["fold", "state"], "EXP-006 state distribution")

    require(current_evaluation.get("experiment_viability_pass") is False, "EXP-006 unexpectedly passed")
    require(current_evaluation.get("decision") == "DO_NOT_ADVANCE_OPPORTUNITY_TARGET_UNDER_EXP_006", "EXP-006 decision drift")
    require(current_evaluation.get("sample_hashes_match") is True, "EXP-006 sample hashes do not match")
    require(current_evaluation.get("champion_selected") is False, "EXP-006 must not select a champion")
    require(current_evaluation.get("v3_019_eligible") is False, "EXP-006 must not unlock V3-019")
    require(float(current_evaluation.get("current_sizing_multiplier")) == 1.0, "EXP-006 sizing changed")

    require(set(current_metrics["model_name"]) == {"opportunity_logistic_l2_v1", "opportunity_random_forest_v1"}, "EXP-006 model set drift")
    require(set(current_metrics["fold"].astype(str)) == {"2024", "2025", "2026_ytd"}, "EXP-006 fold set drift")
    require(current_metrics.groupby("fold")["sample_sha256"].nunique().eq(1).all(), "EXP-006 model sample hashes differ")
    require((current_metrics["relative_brier_improvement"] <= 0.0).all(), "EXP-006 frozen Brier conclusion changed")

    rf = current_metrics.loc[current_metrics["model_name"].eq("opportunity_random_forest_v1")].set_index("fold")
    require(float(rf.loc["2024", "roc_auc"]) > 0.5, "EXP-006 2024 RF AUC relationship changed")
    require(float(rf.loc["2025", "roc_auc"]) < 0.5, "EXP-006 2025 RF reversal changed")
    require(float(rf.loc["2026_ytd", "roc_auc"]) < 0.5, "EXP-006 2026 RF reversal changed")

    require(set(current_states["state"]) == {"BAD", "NORMAL", "GOOD", "EXCELLENT"}, "EXP-006 state set drift")
    print("EXP-006 immutable bytes + semantic reproduction: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
