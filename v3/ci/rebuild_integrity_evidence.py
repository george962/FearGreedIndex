#!/usr/bin/env python3
"""Rebuild and verify the V3 checkpoints that feed champion selection.

This is intentionally a read-only repository validation driver: it generates
research evidence in the working tree but never commits or pushes. CI may upload
those outputs as an artifact, and reviewed evidence can then be committed
explicitly.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "v3" / "reports"
WORKFLOWS = ROOT / ".github" / "workflows"

STALE_FINALIZERS = (
    "v3_dollar_finalize.yml",
    "v3_policy_finalize.yml",
    "v3_retained_combined_finalize.yml",
    "v3_sizing_finalize.yml",
    "v3_treasury_finalize.yml",
)

REQUIRED_RETAINED_COMPONENTS = (
    ROOT / "v3" / "data" / "qqq_spy_daily.csv.gz",
    ROOT / "v3" / "data" / "qqq_spy_source.json",
    ROOT / "v3" / "features" / "build_relative_strength_features.py",
    ROOT / "v3" / "features" / "relative_strength_features.json",
    ROOT / "v3" / "data" / "treasury_daily.csv",
    ROOT / "v3" / "data" / "treasury_source.json",
    ROOT / "v3" / "features" / "build_treasury_features.py",
    ROOT / "v3" / "features" / "build_dollar_features.py",
    ROOT / "v3" / "features" / "build_retained_features.py",
)


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_repository_structure() -> None:
    missing = [
        str(path.relative_to(ROOT))
        for path in REQUIRED_RETAINED_COMPONENTS
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(f"Missing retained-feature components: {missing}")

    leaked = [name for name in STALE_FINALIZERS if (WORKFLOWS / name).exists()]
    if leaked:
        raise RuntimeError(
            f"Temporary write-enabled finalizers leaked into repository: {leaked}"
        )


def verify_qqq_spy_snapshot() -> None:
    snapshot = ROOT / "v3" / "data" / "qqq_spy_daily.csv.gz"
    manifest_path = ROOT / "v3" / "data" / "qqq_spy_source.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = str(manifest.get("snapshot_sha256", ""))
    actual = sha256(snapshot)
    if not expected or expected != actual:
        raise RuntimeError(
            f"QQQ/SPY frozen snapshot hash mismatch: expected={expected!r} actual={actual!r}"
        )


def write_result_marker(
    prefix: str,
    report_name: str,
    retention_key: str,
) -> dict[str, Any]:
    report_path = REPORTS / report_name
    report = json.loads(report_path.read_text(encoding="utf-8"))
    decision = report["feature_family_decision"]
    keep = bool(decision[retention_key])
    marker = REPORTS / f"{prefix}_result_{'KEEP' if keep else 'REJECT'}.txt"
    opposite = REPORTS / f"{prefix}_result_{'REJECT' if keep else 'KEEP'}.txt"
    if opposite.exists():
        opposite.unlink()
    marker.write_text(
        "\n".join(
            [
                f"decision={decision['decision']}",
                f"robust_lane_count={decision['robust_lane_count']}",
                f"sample_hashes_match={report['sample_hashes_match']}",
                f"best_ranked_full_candidate={report['best_ranked_full_candidate_in_ablation_tournament']}",
                f"promotion_ready={report['promotion_ready_experiments_from_absolute_gates']}",
                f"ablation_as_of={report['ablation_as_of']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "decision": decision["decision"],
        "retain": keep,
        "robust_lane_count": int(decision["robust_lane_count"]),
        "sample_hashes_match": bool(report["sample_hashes_match"]),
        "best_ranked_full_candidate": report[
            "best_ranked_full_candidate_in_ablation_tournament"
        ],
        "promotion_ready": report[
            "promotion_ready_experiments_from_absolute_gates"
        ],
        "ablation_as_of": report["ablation_as_of"],
    }


def main() -> int:
    require_repository_structure()
    verify_qqq_spy_snapshot()

    run(
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "v3/tests",
        "-p",
        "test_*.py",
        "-v",
    )

    run(sys.executable, "-m", "v3.features.build_features")
    run(sys.executable, "-m", "v3.labels.build_labels")
    run(sys.executable, "-m", "v3.evaluation.validate_dataset")

    run(sys.executable, "-m", "v3.evaluation.relative_strength_ablation")

    run(sys.executable, "-m", "v3.ci.run_treasury_feature_stage")
    run(sys.executable, "-m", "v3.evaluation.treasury_ablation")

    run(sys.executable, "-m", "v3.ci.run_dollar_feature_stage")
    run(sys.executable, "-m", "v3.evaluation.dollar_ablation")

    run(sys.executable, "-m", "v3.evaluation.retained_combined_ablation")

    run(sys.executable, "-m", "unittest", "-v", "v3.tests.test_decision_policy")
    run(sys.executable, "-m", "v3.policy.build_policy_manifest")
    run(sys.executable, "-m", "unittest", "-v", "v3.tests.test_sizing_policy")
    run(sys.executable, "-m", "v3.policy.build_sizing_manifest")

    decisions = {
        "relative_strength": write_result_marker(
            "relative_strength",
            "relative_strength_ablation.json",
            "retain_relative_strength",
        ),
        "treasury": write_result_marker(
            "treasury", "treasury_ablation.json", "retain_treasury"
        ),
        "dollar": write_result_marker(
            "dollar", "dollar_ablation.json", "retain_dollar"
        ),
        "retained_combined": write_result_marker(
            "retained_combined",
            "retained_combined_ablation.json",
            "retain_combined",
        ),
    }

    (REPORTS / "decision_policy_result_PASS.txt").write_text(
        "status=PASS\n"
        "policy_version=v3-decision-policy-001\n"
        "research_only=true\n"
        "sizing_defined=false\n"
        "sell_or_underweight_allowed=false\n",
        encoding="utf-8",
    )
    (REPORTS / "sizing_policy_result_PASS.txt").write_text(
        "status=PASS\n"
        "sizing_version=v3-sizing-policy-001\n"
        "current_candidate_multiplier=1.00\n"
        "promotion_gate_required=true\n"
        "empirical_1_10x_test=BLOCKED_UNTIL_PREDICTION_PROMOTION\n",
        encoding="utf-8",
    )

    run(sys.executable, "-m", "v3.baseline.freeze_v2_1")
    run("git", "diff", "--exit-code", "--", "reports/baseline_v2_1/")

    summary = {
        "status": "PASS",
        "repair_scope": [
            "V3-013",
            "V3-015A",
            "V3-015B",
            "V3-015C",
            "V3-016",
            "V3-017",
        ],
        "decisions": decisions,
        "policy_manifest_sha256": sha256(
            REPORTS / "decision_policy_manifest.json"
        ),
        "sizing_manifest_sha256": sha256(REPORTS / "sizing_policy_manifest.json"),
        "v2_1_reproducible": True,
        "temporary_finalize_workflows_present": False,
        "qqq_spy_snapshot_sha256": sha256(
            ROOT / "v3" / "data" / "qqq_spy_daily.csv.gz"
        ),
    }
    (REPORTS / "integrity_rebuild_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
