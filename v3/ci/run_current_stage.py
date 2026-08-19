#!/usr/bin/env python3
"""Run the currently implemented v3 research stage on real repository data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v3.evaluation.vix_ablation import run_vix_ablation  # noqa: E402


def main() -> int:
    report = run_vix_ablation()
    if report.get("status") != "VIX_ABLATION_COMPLETE":
        raise SystemExit("V3-012 VIX ablation did not complete successfully")
    print(
        json.dumps(
            {
                "stage": "V3-012",
                "status": report["status"],
                "feature_family_decision": report["feature_family_decision"]["decision"],
                "robust_lane_count": report["feature_family_decision"]["robust_lane_count"],
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
