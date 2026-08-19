#!/usr/bin/env python3
"""Run the currently implemented v3 research stage on real repository data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v3.evaluation.tournament import run_tournament  # noqa: E402


def main() -> int:
    report = run_tournament(regenerate_common_evaluation=True)
    if report.get("status") != "TOURNAMENT_COMPLETE":
        raise SystemExit("V3-010 tournament did not complete successfully")
    print(
        json.dumps(
            {
                "stage": "V3-010",
                "status": report["status"],
                "best_ranked_full_candidate": report["best_ranked_full_candidate"],
                "promotion_ready_experiments": report["promotion_ready_experiments"],
                "champion": report["champion"],
                "trading_score_status": report["trading_score_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
