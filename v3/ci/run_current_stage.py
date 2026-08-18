#!/usr/bin/env python3
"""Run the currently implemented v3 research stage on real repository data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v3.models.random_forest import run_candidate  # noqa: E402


def main() -> int:
    report = run_candidate()
    if report.get("status") != "BENCHMARK_GENERATED":
        raise SystemExit("V3-008 random forest benchmark did not generate successfully")
    if int(report.get("prediction_rows", 0)) <= 0:
        raise SystemExit("V3-008 random forest benchmark generated no predictions")
    print(
        json.dumps(
            {
                "stage": "V3-008",
                "status": report["status"],
                "model_name": report["model_name"],
                "prediction_rows": report["prediction_rows"],
                "feature_count": report["feature_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
