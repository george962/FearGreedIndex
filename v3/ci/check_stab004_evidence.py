#!/usr/bin/env python3
"""Verify frozen STAB-004 evidence and semantic reproduction."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "v3" / "methodology" / "STAB-004" / "manifest.json"
EVALUATION = ROOT / "v3" / "reports" / "stab004_evaluation.json"
METRICS = ROOT / "v3" / "reports" / "stab004_metrics.csv"
CLUSTERS = ROOT / "v3" / "reports" / "stab004_redundancy_clusters.csv"
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
    exact_keys = (
        "method_id",
        "as_of",
        "status",
        "feature_version",
        "feature_count",
        "target",
        "target_source_experiment",
        "relationship_source_method",
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
        require(current.get(key) == frozen.get(key), f"STAB-004 evaluation drift in {key}")

    current_viability = current.get("viability", {})
    frozen_viability = frozen.get("viability", {})
    require(set(current_viability) == set(frozen_viability), "STAB-004 viability-key drift")
    for key, frozen_value in frozen_viability.items():
        current_value = current_viability[key]
        if isinstance(frozen_value, (bool, int)):
            require(current_value == frozen_value, f"STAB-004 viability drift in {key}")
        else:
            require(
                np.isclose(float(current_value), float(frozen_value), rtol=0.0, atol=ATOL),
                f"STAB-004 viability numeric drift in {key}",
            )


def main() -> int:
    for path in (MANIFEST, EVALUATION, METRICS, CLUSTERS, CHECKPOINTS):
        require(path.exists(), f"Missing STAB-004 artifact: {path}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(manifest.get("method_id") == "STAB-004", "STAB-004 id drift")
    require(manifest.get("status") == "complete_reject", "STAB-004 status drift")
    require(manifest.get("pre_registered_issue") == 72, "STAB-004 pre-registration drift")
    require(manifest.get("feature_set_version") == "v3-features-004-treasury", "STAB-004 feature version drift")
    require(manifest.get("feature_count") == 53, "STAB-004 feature-count drift")
    require(manifest.get("tuning_after_result_allowed_under_same_version") is False, "STAB-004 post-result tuning must remain disabled")

    frozen_eval_bytes = head_bytes(EVALUATION)
    frozen_metrics_bytes = head_bytes(METRICS)
    frozen_clusters_bytes = head_bytes(CLUSTERS)
    evidence = manifest.get("evidence", {})
    require(evidence.get("v3/reports/stab004_evaluation.json") == sha256_bytes(frozen_eval_bytes), "STAB-004 committed evaluation hash mismatch")
    require(evidence.get("v3/reports/stab004_metrics.csv") == sha256_bytes(frozen_metrics_bytes), "STAB-004 committed metrics hash mismatch")
    require(evidence.get("v3/reports/stab004_redundancy_clusters.csv") == sha256_bytes(frozen_clusters_bytes), "STAB-004 committed cluster hash mismatch")

    current_eval = json.loads(EVALUATION.read_text(encoding="utf-8"))
    frozen_eval = json.loads(frozen_eval_bytes.decode("utf-8"))
    current_metrics = pd.read_csv(METRICS)
    frozen_metrics = pd.read_csv(io.BytesIO(frozen_metrics_bytes))
    current_clusters = pd.read_csv(CLUSTERS)
    frozen_clusters = pd.read_csv(io.BytesIO(frozen_clusters_bytes))

    compare_evaluation(current_eval, frozen_eval)
    compare_frame(current_metrics, frozen_metrics, "STAB-004 metrics")
    compare_frame(current_clusters, frozen_clusters, "STAB-004 clusters")

    require(current_eval.get("decision") == "DO_NOT_ADVANCE_CAUSAL_ROLLING_NORMALIZATION_UNDER_STAB_004", "STAB-004 decision drift")
    require(current_eval.get("method_viability_pass") is False, "STAB-004 unexpectedly passed")
    require(current_eval.get("evid001_outcomes_opened") is False, "STAB-004 opened EVID-001")
    require(current_eval.get("v3_019_eligible") is False, "STAB-004 unlocked V3-019")
    require(float(current_eval.get("current_sizing_multiplier")) == 1.0, "STAB-004 sizing changed")

    viability = current_eval.get("viability", {})
    require(int(viability.get("support_folds", -1)) == 3, "STAB-004 structural-support conclusion changed")
    require(np.isclose(float(viability.get("mean_roc_auc")), 0.6716860053321619, rtol=0.0, atol=ATOL), "STAB-004 mean-AUC conclusion changed")
    require(np.isclose(float(viability.get("minimum_fold_roc_auc")), 0.6214426688737507, rtol=0.0, atol=ATOL), "STAB-004 minimum-AUC conclusion changed")
    require(int(viability.get("roc_auc_above_0_52_folds", -1)) == 3, "STAB-004 AUC-support conclusion changed")
    require(int(viability.get("favorable_enrichment_above_0_05_folds", -1)) == 2, "STAB-004 favorable-enrichment conclusion changed")
    require(int(viability.get("unfavorable_depletion_above_0_05_folds", -1)) == 2, "STAB-004 unfavorable-depletion conclusion changed")
    require(bool(viability.get("supported_fold_call_count_gate_pass")) is True, "STAB-004 call-count conclusion changed")
    require(bool(viability.get("supported_fold_coverage_gate_pass")) is False, "STAB-004 coverage conclusion changed")
    require(bool(viability.get("sample_hashes_match_exp006")) is True, "STAB-004 sample hashes changed")
    require(bool(viability.get("evid001_outcomes_sealed")) is True, "STAB-004 EVID-001 seal changed")

    metrics_by_fold = current_metrics.set_index("fold")
    require(set(metrics_by_fold.index) == {"2024", "2025", "2026_ytd"}, "STAB-004 fold set drift")
    require(np.isclose(float(metrics_by_fold.loc["2025", "non_abstain_coverage"]), 0.552, rtol=0.0, atol=ATOL), "STAB-004 2025 coverage changed")
    require(float(metrics_by_fold.loc["2025", "non_abstain_coverage"]) > 0.55, "STAB-004 frozen coverage failure disappeared")
    require(np.isclose(float(metrics_by_fold.loc["2026_ytd", "roc_auc"]), 0.7312734082397003, rtol=0.0, atol=ATOL), "STAB-004 2026 ranking changed")

    result = manifest.get("result", {})
    require(result.get("decision") == current_eval.get("decision"), "STAB-004 manifest/result decision mismatch")
    require(result.get("method_viability_pass") is False, "STAB-004 manifest unexpectedly passes")
    require(result.get("coverage_gate_pass") is False, "STAB-004 manifest coverage conclusion changed")

    checkpoints = json.loads(CHECKPOINTS.read_text(encoding="utf-8"))
    require(checkpoints.get("checkpoints") == [], "EVID-001 checkpoint outcomes were opened")

    print("STAB-004 immutable evidence + semantic reproduction: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
