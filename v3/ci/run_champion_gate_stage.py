#!/usr/bin/env python3
"""Build and evaluate V3-018 gates for the strongest retained candidate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v3.ci.check_repository_integrity import main as check_repository_integrity  # noqa: E402
from v3.evaluation.build_champion_gate_manifest import main as build_manifest  # noqa: E402
from v3.evaluation.build_current_champion_evidence import (  # noqa: E402
    build_evidence,
)
from v3.evaluation.champion_gates import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_CURRENT_EVIDENCE,
    DEFAULT_REPORT,
    evaluate_candidate,
    load_json,
)


def main() -> int:
    if check_repository_integrity() != 0:
        raise SystemExit("V3 repository integrity checkpoint failed")

    if build_manifest() != 0:
        raise SystemExit("V3-018 gate manifest build failed")

    evidence = build_evidence()
    DEFAULT_CURRENT_EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_CURRENT_EVIDENCE.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    config = load_json(DEFAULT_CONFIG)
    report = evaluate_candidate(evidence, config)
    DEFAULT_REPORT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if report["candidate_id"] != "UST-EXP-004":
        raise SystemExit(
            f"Unexpected strongest retained candidate: {report['candidate_id']}"
        )
    if report["status"] != "NOT_PROMOTION_READY":
        raise SystemExit(
            "Current V3-018 candidate unexpectedly became promotion-ready; "
            "review all evidence before allowing V3-019"
        )
    if report["promotion_ready"] is not False or report["v3_019_eligible"] is not False:
        raise SystemExit("V3-019 must remain blocked for the current candidate")
    if report["gates"]["prediction_prerequisite"]["status"] != "FAIL":
        raise SystemExit("Current candidate should fail the pre-existing prediction prerequisite")
    if evidence.get("sizing_activation") != "BLOCKED" or evidence.get("current_multiplier") != 1.0:
        raise SystemExit("V3-017 sizing must remain blocked at 1.00x")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
