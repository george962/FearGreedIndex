#!/usr/bin/env python3
"""Run the currently implemented v3 research stage on real repository data."""

from __future__ import annotations

import json

from v3.models.logistic_baseline import run_baseline


def main() -> int:
    report = run_baseline()
    if report.get("status") != "BASELINE_GENERATED":
        raise SystemExit("V3-005 logistic baseline did not generate successfully")
    if int(report.get("prediction_rows", 0)) <= 0:
        raise SystemExit("V3-005 logistic baseline generated no predictions")
    print(
        json.dumps(
            {
                "stage": "V3-005",
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
