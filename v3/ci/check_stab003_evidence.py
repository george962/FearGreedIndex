#!/usr/bin/env python3
"""Verify STAB-003 immutable evidence and semantic reproduction."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "v3" / "methodology" / "STAB-003" / "manifest.json"
EVALUATION = ROOT / "v3" / "reports" / "stab003_evaluation.json"
METRICS = ROOT / "v3" / "reports" / "stab003_metrics.csv"
FEATURES = ROOT / "v3" / "reports" / "stab003_consensus_features.csv"
CHECKPOINTS = ROOT / "v3" / "evidence" / "forward_checkpoints.json"
ATOL = 1e-8


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def head_bytes(path: Path) -> bytes:
    relative = path.relative_to(ROOT).as_posix()
    return subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def compare_frame(current: pd.DataFrame, frozen: pd.DataFrame, name: str) -> None:
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
    # The generated full dataset can gain rows after the frozen 2026-08-18
    # research cutoff. Do not compare its whole-file hash. Exact frozen sample
    # hashes, fold evidence, decision state, and governance are checked below.
    exact_keys = (
        "method_id",
        "as_of",
        "status",
        "feature_version",
        "feature_count",
        "target",
        "target_source_experiment",
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
        require(current.get(key) == frozen.get(key), f"STAB-003 evaluation drift in {key}")

    current_viability = current.get("viability", {})
    frozen_viability = frozen.get("viability", {})
    require(set(current_viability) == set(frozen_viability), "STAB-003 viability-key drift")
    for key, frozen_value in frozen_viability.items():
        current_value = current_viability[key]
        if isinstance(frozen_value, (bool, int)):
            require(current_value == frozen_value, f"STAB-003 viability drift in {key}")
        else:
            require(
                np.isclose(float(current_value), float(frozen_value), rtol=0.0, atol=ATOL),
                f"STAB-003 viability numeric drift in {key}",
            )


def main() -> int:
    for path in (MANIFEST, EVALUATION, METRICS, FEATURES, CHECKPOINTS):
        require(path.exists(), f"Missing STAB-003 artifact: {path}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(manifest.get("method_id") == "STAB-003", "STAB-003 id drift")
    require(manifest.get("status") == "complete_reject", "STAB-003 status drift")
    require(manifest.get("pre_registered_issue") == 70, "STAB-003 pre-registration drift")
    require(
        manifest.get("feature_set_version") == "v3-features-004-treasury",
        "STAB-003 feature version drift",
    )
    require(manifest.get("feature_count") == 53, "STAB-003 feature-count drift")
    require(
        manifest.get("tuning_after_result_allowed_under_same_version") is False,
        "STAB-003 post-result tuning must remain disabled",
    )

    frozen_eval_bytes = head_bytes(EVALUATION)
    frozen_metrics_bytes = head_bytes(METRICS)
    frozen_features_bytes = head_bytes(FEATURES)
    evidence = manifest.get("evidence", {})
    require(
        evidence.get("v3/reports/stab003_evaluation.json")
        == sha256_bytes(frozen_eval_bytes),
        "STAB-003 committed evaluation hash mismatch",
    )
    require(
        evidence.get("v3/reports/stab003_metrics.csv")
        == sha256_bytes(frozen_metrics_bytes),
        "STAB-003 committed metrics hash mismatch",
    )
    require(
        evidence.get("v3/reports/stab003_consensus_features.csv")
        == sha256_bytes(frozen_features_bytes),
        "STAB-003 committed consensus-feature hash mismatch",
    )

    current_eval = json.loads(EVALUATION.read_text(encoding="utf-8"))
    frozen_eval = json.loads(frozen_eval_bytes.decode("utf-8"))
    current_metrics = pd.read_csv(METRICS)
    frozen_metrics = pd.read_csv(io.BytesIO(frozen_metrics_bytes))
    current_features = pd.read_csv(FEATURES)
    frozen_features = pd.read_csv(io.BytesIO(frozen_features_bytes))

    compare_evaluation(current_eval, frozen_eval)
    compare_frame(current_metrics, frozen_metrics, "STAB-003 metrics")
    compare_frame(current_features, frozen_features, "STAB-003 consensus features")

    require(
        current_eval.get("decision")
        == "DO_NOT_ADVANCE_CONSENSUS_ABSTENTION_UNDER_STAB_003",
        "STAB-003 decision drift",
    )
    require(current_eval.get("method_viability_pass") is False, "STAB-003 unexpectedly passed")
    require(current_eval.get("evid001_outcomes_opened") is False, "STAB-003 opened EVID-001")
    require(current_eval.get("v3_019_eligible") is False, "STAB-003 unlocked V3-019")
    require(
        float(current_eval.get("current_sizing_multiplier")) == 1.0,
        "STAB-003 sizing changed",
    )

    viability = current_eval.get("viability", {})
    require(int(viability.get("support_folds", -1)) == 1, "STAB-003 support conclusion changed")
    require(
        np.isclose(float(viability.get("mean_roc_auc")), 0.5521529678911737, rtol=0.0, atol=ATOL),
        "STAB-003 mean-AUC conclusion changed",
    )
    require(
        int(viability.get("favorable_lift_above_0_05_folds", -1)) == 0,
        "STAB-003 favorable-lift conclusion changed",
    )
    require(
        int(viability.get("unfavorable_separation_above_0_05_folds", -1)) == 0,
        "STAB-003 unfavorable-separation conclusion changed",
    )
    require(
        bool(viability.get("sample_hashes_match_exp006")) is True,
        "STAB-003 sample hashes changed",
    )
    require(
        bool(viability.get("evid001_outcomes_sealed")) is True,
        "STAB-003 EVID-001 seal changed",
    )

    metrics_by_fold = current_metrics.set_index("fold")
    require(set(metrics_by_fold.index) == {"2024", "2025", "2026_ytd"}, "STAB-003 fold set drift")
    require(bool(metrics_by_fold.loc["2024", "support_pass"]) is True, "STAB-003 2024 support changed")
    require(bool(metrics_by_fold.loc["2025", "support_pass"]) is False, "STAB-003 2025 support changed")
    require(bool(metrics_by_fold.loc["2026_ytd", "support_pass"]) is False, "STAB-003 2026 support changed")
    require(
        np.isclose(float(metrics_by_fold.loc["2024", "roc_auc"]), 0.6564589036735213, rtol=0.0, atol=ATOL),
        "STAB-003 2024 ranking changed",
    )
    require(
        int(metrics_by_fold.loc["2024", "strong_favorable_count"]) == 0,
        "STAB-003 2024 favorable-call conclusion changed",
    )
    require(
        int(metrics_by_fold.loc["2024", "strong_unfavorable_count"]) == 130,
        "STAB-003 2024 unfavorable-call conclusion changed",
    )

    result = manifest.get("result", {})
    require(result.get("decision") == current_eval.get("decision"), "STAB-003 manifest/result decision mismatch")
    require(result.get("method_viability_pass") is False, "STAB-003 manifest unexpectedly passes")
    require(int(result.get("support_folds", -1)) == int(viability.get("support_folds", -2)), "STAB-003 manifest support mismatch")
    require(
        np.isclose(float(result.get("mean_roc_auc")), float(viability.get("mean_roc_auc")), rtol=0.0, atol=ATOL),
        "STAB-003 manifest mean-AUC mismatch",
    )

    checkpoints = json.loads(CHECKPOINTS.read_text(encoding="utf-8"))
    require(checkpoints.get("checkpoints") == [], "EVID-001 checkpoint outcomes were opened")

    print("STAB-003 immutable evidence + semantic reproduction: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
