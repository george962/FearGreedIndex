#!/usr/bin/env python3
"""Run the currently implemented v3 research stage on real repository data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v3.evaluation.walk_forward import run_common_evaluation  # noqa: E402


def main() -> int:
    report = run_common_evaluation()
    if report.get("status") != "COMMON_EVALUATION_COMPLETE":
        raise SystemExit("V3-009 common evaluation did not complete successfully")
    if int(report.get("metric_rows", 0)) <= 0:
        raise SystemExit("V3-009 common evaluation generated no metrics")
    print(
        json.dumps(
            {
                "stage": "V3-009",
                "status": report["status"],
                "experiments": report["experiments"],
                "prediction_rows": report["prediction_rows"],
                "metric_rows": report["metric_rows"],
                "trading_evaluation_status": report["trading_evaluation_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
