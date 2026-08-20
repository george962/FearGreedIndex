#!/usr/bin/env python3
"""Verify frozen EXP-008 evidence and semantic reproduction."""

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
MANIFEST = ROOT / "v3" / "experiments" / "EXP-008" / "manifest.json"
EVALUATION = ROOT / "v3" / "reports" / "exp008_evaluation.json"
METRICS = ROOT / "v3" / "reports" / "exp008_metrics.csv"
DIAGNOSTICS = ROOT / "v3" / "reports" / "exp008_sentiment_diagnostics.csv"
ATOL = 1e-10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def head_bytes(path: Path) -> bytes:
    relative = path.relative_to(ROOT).as_posix()
    return subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=ROOT, check=True, capture_output=True
    ).stdout


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def compare_value(current: Any, frozen: Any, label: str) -> None:
    if isinstance(frozen, bool) or isinstance(frozen, int):
        require(current == frozen, f"{label} drift")
    elif isinstance(frozen, float):
        require(
            np.isclose(float(current), float(frozen), rtol=0.0, atol=ATOL, equal_nan=True),
            f"{label} numeric drift",
        )
    else:
        require(current == frozen, f"{label} drift")


def compare_evaluation(current: dict[str, Any], frozen: dict[str, Any]) -> None:
    require(set(current) == set(frozen), "EXP-008 evaluation key-set drift")
    for key, frozen_value in frozen.items():
        if key == "dataset_sha256":
            continue
        current_value = current[key]
        if isinstance(frozen_value, dict):
            require(set(current_value) == set(frozen_value), f"EXP-008 {key} key-set drift")
            for inner_key, inner_value in frozen_value.items():
                compare_value(current_value[inner_key], inner_value, f"EXP-008 {key}.{inner_key}")
        elif isinstance(frozen_value, list):
            require(current_value == frozen_value, f"EXP-008 {key} drift")
        else:
            compare_value(current_value, frozen_value, f"EXP-008 {key}")


def compare_frame(current: pd.DataFrame, frozen: pd.DataFrame, keys: list[str], label: str) -> None:
    require(list(current.columns) == list(frozen.columns), f"{label} column drift")
    current = current.sort_values(keys, kind="stable").reset_index(drop=True)
    frozen = frozen.sort_values(keys, kind="stable").reset_index(drop=True)
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


def main() -> int:
    for path in (MANIFEST, EVALUATION, METRICS, DIAGNOSTICS):
        require(path.exists(), f"Missing EXP-008 artifact: {path}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    metrics = pd.read_csv(METRICS)
    diagnostics = pd.read_csv(DIAGNOSTICS)

    frozen_eval_bytes = head_bytes(EVALUATION)
    frozen_metrics_bytes = head_bytes(METRICS)
    frozen_diag_bytes = head_bytes(DIAGNOSTICS)
    frozen_evaluation = json.loads(frozen_eval_bytes.decode("utf-8"))
    frozen_metrics = pd.read_csv(io.BytesIO(frozen_metrics_bytes))
    frozen_diagnostics = pd.read_csv(io.BytesIO(frozen_diag_bytes))

    require(manifest.get("experiment_id") == "EXP-008", "EXP-008 manifest experiment ID drift")
    require(manifest.get("status") == "complete_reject", "EXP-008 manifest must remain complete_reject")
    require(manifest.get("sentiment_state", {}).get("extreme_fear_max") == 25.0, "EXP-008 extreme-fear threshold drift")
    require(manifest.get("sentiment_state", {}).get("extreme_greed_min") == 75.0, "EXP-008 extreme-greed threshold drift")
    require(manifest.get("tuning_after_result_allowed_under_same_version") is False, "EXP-008 post-result tuning must remain disabled")

    expected = manifest.get("evidence", {})
    require(expected, "EXP-008 manifest has no frozen evidence hashes")
    require(expected.get("v3/reports/exp008_evaluation.json") == sha256_bytes(frozen_eval_bytes), "EXP-008 committed evaluation hash mismatch")
    require(expected.get("v3/reports/exp008_metrics.csv") == sha256_bytes(frozen_metrics_bytes), "EXP-008 committed metrics hash mismatch")
    require(expected.get("v3/reports/exp008_sentiment_diagnostics.csv") == sha256_bytes(frozen_diag_bytes), "EXP-008 committed diagnostics hash mismatch")

    compare_evaluation(evaluation, frozen_evaluation)
    compare_frame(metrics, frozen_metrics, ["fold"], "EXP-008 metrics")
    compare_frame(diagnostics, frozen_diagnostics, ["fold", "sentiment_state"], "EXP-008 diagnostics")

    require(evaluation.get("decision") == "DO_NOT_ADVANCE_SENTIMENT_EXTREMES_UNDER_EXP_008", "EXP-008 frozen decision drift")
    require(evaluation.get("experiment_viability_pass") is False, "EXP-008 must remain rejected")
    require(evaluation.get("v3_019_eligible") is False, "EXP-008 must not unlock V3-019")
    require(float(evaluation.get("current_sizing_multiplier")) == 1.0, "EXP-008 must not change sizing")

    viability = evaluation.get("viability", {})
    require(int(viability.get("hypothesized_prevalence_ordering_folds", -1)) == 0, "EXP-008 prevalence-ordering evidence drift")
    require(viability.get("sample_hashes_match_exp006") is True, "EXP-008 must preserve EXP-006 realized-date samples")

    result = manifest.get("result", {})
    require(result.get("decision") == evaluation.get("decision"), "EXP-008 manifest decision mismatch")
    require(result.get("experiment_viability_pass") == evaluation.get("experiment_viability_pass"), "EXP-008 manifest viability mismatch")

    print("EXP-008 immutable evidence + semantic reproduction: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
