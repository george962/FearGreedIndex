#!/usr/bin/env python3
"""Fail-closed static integrity checks for the repaired V3 repository state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "v3" / "reports"
WORKFLOWS = ROOT / ".github" / "workflows"

STALE_FINALIZERS = (
    "v3_dollar_finalize.yml",
    "v3_policy_finalize.yml",
    "v3_retained_combined_finalize.yml",
    "v3_sizing_finalize.yml",
    "v3_treasury_finalize.yml",
    "v3_integrity_preserve.yml",
)

REQUIRED_FILES = (
    "v3/data/qqq_spy_daily.csv.gz",
    "v3/data/qqq_spy_source.json",
    "v3/data/treasury_daily.csv",
    "v3/data/treasury_source.json",
    "v3/data/dollar_daily.csv",
    "v3/data/dollar_source.json",
    "v3/reports/integrity_rebuild_summary.json",
    "v3/reports/relative_strength_ablation.json",
    "v3/reports/relative_strength_result_REJECT.txt",
    "v3/reports/treasury_ablation.json",
    "v3/reports/treasury_result_KEEP.txt",
    "v3/reports/dollar_ablation.json",
    "v3/reports/dollar_result_REJECT.txt",
    "v3/reports/retained_combined_ablation.json",
    "v3/reports/retained_combined_result_REJECT.txt",
    "v3/reports/decision_policy_manifest.json",
    "v3/reports/decision_policy_result_PASS.txt",
    "v3/reports/sizing_policy_manifest.json",
    "v3/reports/sizing_policy_result_PASS.txt",
)

EXPECTED_DECISIONS = {
    "relative_strength": (False, 1),
    "treasury": (True, 2),
    "dollar": (False, 1),
    "retained_combined": (False, 1),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_files() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        raise RuntimeError(f"Missing V3 immutable checkpoint files: {missing}")


def require_no_temporary_finalizers() -> None:
    leaked = [name for name in STALE_FINALIZERS if (WORKFLOWS / name).exists()]
    if leaked:
        raise RuntimeError(f"Temporary write-enabled workflows present: {leaked}")


def verify_source_hashes() -> None:
    qqq_manifest = json.loads(
        (ROOT / "v3/data/qqq_spy_source.json").read_text(encoding="utf-8")
    )
    qqq_expected = str(qqq_manifest.get("snapshot_sha256", ""))
    qqq_actual = sha256(ROOT / "v3/data/qqq_spy_daily.csv.gz")
    if qqq_expected != qqq_actual:
        raise RuntimeError("QQQ/SPY snapshot hash does not match manifest")

    treasury_manifest = json.loads(
        (ROOT / "v3/data/treasury_source.json").read_text(encoding="utf-8")
    )
    treasury_expected = str(treasury_manifest.get("normalized_sha256", ""))
    treasury_actual = sha256(ROOT / "v3/data/treasury_daily.csv")
    if treasury_expected != treasury_actual:
        raise RuntimeError("Treasury normalized snapshot hash does not match manifest")

    dollar_manifest = json.loads(
        (ROOT / "v3/data/dollar_source.json").read_text(encoding="utf-8")
    )
    dollar_expected = str(dollar_manifest.get("normalized_sha256", ""))
    dollar_actual = sha256(ROOT / "v3/data/dollar_daily.csv")
    if dollar_expected != dollar_actual:
        raise RuntimeError("Dollar normalized snapshot hash does not match manifest")


def verify_rebuild_summary() -> dict[str, object]:
    summary = json.loads(
        (REPORTS / "integrity_rebuild_summary.json").read_text(encoding="utf-8")
    )
    if summary.get("status") != "PASS":
        raise RuntimeError("V3 integrity rebuild summary is not PASS")
    if not bool(summary.get("v2_1_reproducible")):
        raise RuntimeError("Frozen v2.1 reproducibility is not confirmed")
    if bool(summary.get("temporary_finalize_workflows_present")):
        raise RuntimeError("Integrity summary records leaked temporary finalizers")

    decisions = summary.get("decisions")
    if not isinstance(decisions, dict):
        raise RuntimeError("Integrity summary decisions are missing")
    for family, (expected_retain, expected_lanes) in EXPECTED_DECISIONS.items():
        value = decisions.get(family)
        if not isinstance(value, dict):
            raise RuntimeError(f"Missing integrity decision for {family}")
        if bool(value.get("retain")) != expected_retain:
            raise RuntimeError(f"Unexpected retained-state change for {family}")
        if int(value.get("robust_lane_count", -1)) != expected_lanes:
            raise RuntimeError(f"Unexpected robust-lane count for {family}")
        if not bool(value.get("sample_hashes_match")):
            raise RuntimeError(f"Sample hashes do not match for {family}")
        if value.get("promotion_ready") != []:
            raise RuntimeError(f"Unexpected promotion-ready experiment for {family}")
    return summary


def verify_policy_manifests() -> None:
    policy = json.loads((ROOT / "v3/policy/policy_v1.json").read_text(encoding="utf-8"))
    policy_manifest = json.loads(
        (REPORTS / "decision_policy_manifest.json").read_text(encoding="utf-8")
    )
    if policy_manifest.get("policy_version") != policy.get("policy_version"):
        raise RuntimeError("Decision-policy version mismatch")
    if policy_manifest.get("policy_config_sha256") != sha256(ROOT / "v3/policy/policy_v1.json"):
        raise RuntimeError("Decision-policy config hash mismatch")
    if policy_manifest.get("policy_engine_sha256") != sha256(ROOT / "v3/policy/decision_policy.py"):
        raise RuntimeError("Decision-policy engine hash mismatch")

    sizing = json.loads((ROOT / "v3/policy/sizing_v1.json").read_text(encoding="utf-8"))
    sizing_manifest = json.loads(
        (REPORTS / "sizing_policy_manifest.json").read_text(encoding="utf-8")
    )
    if sizing_manifest.get("sizing_version") != sizing.get("sizing_version"):
        raise RuntimeError("Sizing-policy version mismatch")
    if sizing_manifest.get("config_sha256") != sha256(ROOT / "v3/policy/sizing_v1.json"):
        raise RuntimeError("Sizing-policy config hash mismatch")
    if sizing_manifest.get("engine_sha256") != sha256(ROOT / "v3/policy/sizing_policy.py"):
        raise RuntimeError("Sizing-policy engine hash mismatch")
    if sizing_manifest.get("current_candidate_activation") != "BLOCKED":
        raise RuntimeError("Sizing activation must remain blocked before champion promotion")


def main() -> int:
    require_files()
    require_no_temporary_finalizers()
    verify_source_hashes()
    summary = verify_rebuild_summary()
    verify_policy_manifests()
    result = {
        "stage": "V3_REPOSITORY_INTEGRITY",
        "status": "PASS",
        "strongest_retained_research_candidate": "UST-EXP-004",
        "champion_selected": False,
        "current_sizing_multiplier": 1.00,
        "next": "V3-018 champion acceptance gates",
        "repaired_families": summary["decisions"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
