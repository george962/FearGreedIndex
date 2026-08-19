#!/usr/bin/env python3
"""Validate the v3 point-in-time feature/label dataset before model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES = ROOT / "v3" / "data" / "features_daily.parquet"
DEFAULT_LABELS = ROOT / "v3" / "data" / "labels_daily.parquet"
DEFAULT_REPORT = ROOT / "v3" / "reports" / "dataset_validation.json"

FORBIDDEN_FEATURE_TOKENS = (
    "forward_",
    "future_",
    "target",
    "label",
    "outcome",
    "known_date",
    "entry_date",
    "entry_price",
    "max_drawdown_",
    "further_5pct_decline",
)
SOURCE_DATE_COLUMNS = (
    ("fear_greed_date", "Fear & Greed"),
    ("vix_date", "VIX"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def _check_sorted_unique(frame: pd.DataFrame, name: str) -> list[str]:
    errors: list[str] = []
    if "decision_date" not in frame:
        return [f"{name}: missing decision_date"]
    dates = pd.to_datetime(frame["decision_date"], errors="coerce")
    if dates.isna().any():
        errors.append(f"{name}: invalid decision_date values")
    if dates.duplicated().any():
        errors.append(f"{name}: duplicate decision_date values")
    if not dates.is_monotonic_increasing:
        errors.append(f"{name}: decision_date is not sorted")
    return errors


def validate_frames(features: pd.DataFrame, labels: pd.DataFrame) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    errors.extend(_check_sorted_unique(features, "features"))
    errors.extend(_check_sorted_unique(labels, "labels"))

    forbidden_columns = [
        column
        for column in features.columns
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden_columns:
        errors.append(f"features: forward/target columns present: {forbidden_columns}")

    if "decision_date" in features:
        decision = pd.to_datetime(features["decision_date"], errors="coerce")
        for source_column, label in SOURCE_DATE_COLUMNS:
            if source_column not in features:
                continue
            source = pd.to_datetime(features[source_column], errors="coerce")
            future_source = source.notna() & decision.notna() & source.gt(decision)
            if future_source.any():
                errors.append(f"features: future {label} source date detected")

    if "entry_date" in labels:
        decision = pd.to_datetime(labels["decision_date"], errors="coerce")
        entry = pd.to_datetime(labels["entry_date"], errors="coerce")
        invalid = entry.notna() & decision.notna() & entry.le(decision)
        if invalid.any():
            errors.append("labels: entry_date must be after decision_date")

    for horizon in (5, 20, 60):
        return_column = f"forward_return_{horizon}d"
        positive_column = f"forward_positive_{horizon}d"
        known_column = f"_forward_{horizon}d_known_date"
        if return_column not in labels or known_column not in labels:
            errors.append(f"labels: missing horizon {horizon} columns")
            continue
        known = pd.to_datetime(labels[known_column], errors="coerce")
        entry = pd.to_datetime(labels["entry_date"], errors="coerce")
        invalid_known = known.notna() & entry.notna() & known.lt(entry)
        if invalid_known.any():
            errors.append(f"labels: {known_column} precedes entry_date")
        if positive_column in labels:
            returns = pd.to_numeric(labels[return_column], errors="coerce")
            positive = labels[positive_column].astype("boolean")
            comparable = returns.notna() & positive.notna()
            mismatch = comparable & ((returns.gt(0)).astype("boolean") != positive)
            if mismatch.any():
                errors.append(f"labels: {positive_column} inconsistent with {return_column}")

    numeric_features = features.select_dtypes(include=[np.number])
    infinite_columns = [
        column for column in numeric_features.columns if np.isinf(numeric_features[column]).any()
    ]
    if infinite_columns:
        errors.append(f"features: infinite numeric values: {infinite_columns}")

    metadata_dates = {"decision_date", *(column for column, _ in SOURCE_DATE_COLUMNS)}
    high_missing = {
        column: float(features[column].isna().mean())
        for column in features.columns
        if column not in metadata_dates and float(features[column].isna().mean()) > 0.80
    }
    if high_missing:
        warnings.append(f"features: >80% missing: {high_missing}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "feature_rows": int(len(features)),
        "label_rows": int(len(labels)),
        "feature_columns": int(len(features.columns)),
        "label_columns": int(len(labels.columns)),
    }


def main() -> int:
    args = parse_args()
    features = pd.read_parquet(args.features, engine="pyarrow")
    labels = pd.read_parquet(args.labels, engine="pyarrow")
    report = validate_frames(features, labels)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
