#!/usr/bin/env python3
"""Run the currently implemented v3 research stage on real repository data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v3.evaluation.relative_strength_ablation import (  # noqa: E402
    run_relative_strength_ablation,
)


def main() -> int:
    report = run_relative_strength_ablation()
    if report.get("status") != "RELATIVE_STRENGTH_ABLATION_COMPLETE":
        raise SystemExit("V3-013 relative-strength ablation did not complete")
    decision = report["feature_family_decision"]
    print(
        json.dumps(
            {
                "stage": "V3-013",
                "status": report["status"],
                "feature_family_decision": decision["decision"],
                "robust_lane_count": decision["robust_lane_count"],
                "sample_hashes_match": report["sample_hashes_match"],
                "best_ranked_full_candidate": report[
                    "best_ranked_full_candidate_in_ablation_tournament"
                ],
                "champion_selected": report["champion_selected"],
                "next": report["next"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
