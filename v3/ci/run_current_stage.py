#!/usr/bin/env python3
"""Run the currently implemented v3 research stage on real repository data."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v3.data_sources.fetch_relative_strength import (  # noqa: E402
    DEFAULT_END_INCLUSIVE,
    DEFAULT_MANIFEST as SOURCE_MANIFEST,
    DEFAULT_OUTPUT as SOURCE_SNAPSHOT,
    DEFAULT_START,
    fetch_snapshot_frame,
    write_snapshot,
)
from v3.evaluation.validate_dataset import validate_frames  # noqa: E402
from v3.features.build_relative_strength_features import (  # noqa: E402
    DEFAULT_BASE_FEATURES,
    DEFAULT_OUTPUT as RELATIVE_FEATURES,
    DEFAULT_OUTPUT_REGISTRY as RELATIVE_REGISTRY,
    DEFAULT_REPORT as RELATIVE_REPORT,
    DEFAULT_SOURCE,
    run_build,
)
from v3.labels.build_labels import build_labels, load_market  # noqa: E402

RELATIVE_LABELS = ROOT / "v3" / "data" / "labels_daily_relative_strength.parquet"
RELATIVE_MODEL_DATASET = ROOT / "v3" / "data" / "model_dataset_relative_strength.parquet"
RELATIVE_VALIDATION = ROOT / "v3" / "reports" / "dataset_validation_relative_strength.json"
MARKET_DATA = ROOT / "data" / "spx_daily.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_source_snapshot() -> dict[str, object]:
    if not SOURCE_SNAPSHOT.exists() or not SOURCE_MANIFEST.exists():
        frame = fetch_snapshot_frame(
            start=DEFAULT_START,
            end_inclusive=DEFAULT_END_INCLUSIVE,
        )
        return write_snapshot(
            frame,
            output=SOURCE_SNAPSHOT,
            manifest_output=SOURCE_MANIFEST,
            start=DEFAULT_START,
            end_inclusive=DEFAULT_END_INCLUSIVE,
        )

    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    expected = str(manifest.get("snapshot_sha256", ""))
    actual = _sha256(SOURCE_SNAPSHOT)
    if not expected or expected != actual:
        raise SystemExit(
            "Checked-in QQQ/SPY snapshot hash does not match its manifest; "
            "refresh intentionally rather than silently refetching it."
        )
    return manifest


def build_labels_and_validate() -> dict[str, object]:
    base = pd.read_parquet(DEFAULT_BASE_FEATURES, engine="pyarrow")
    expanded = pd.read_parquet(RELATIVE_FEATURES, engine="pyarrow")
    assert_frame_equal(
        expanded[base.columns].reset_index(drop=True),
        base.reset_index(drop=True),
        check_dtype=True,
    )

    future = (
        expanded["relative_strength_date"].notna()
        & expanded["relative_strength_date"].gt(expanded["decision_date"])
    )
    if future.any():
        raise SystemExit("V3-013 detected future QQQ/SPY source dates")

    market = load_market(MARKET_DATA)
    labels = build_labels(expanded["decision_date"], market)
    model_dataset = expanded.merge(
        labels,
        on="decision_date",
        how="left",
        validate="one_to_one",
    )
    RELATIVE_LABELS.parent.mkdir(parents=True, exist_ok=True)
    RELATIVE_VALIDATION.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(RELATIVE_LABELS, index=False, engine="pyarrow")
    model_dataset.to_parquet(RELATIVE_MODEL_DATASET, index=False, engine="pyarrow")

    validation = validate_frames(expanded, labels)
    RELATIVE_VALIDATION.write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )
    if validation.get("status") != "PASS":
        raise SystemExit(f"V3-013 dataset validation failed: {validation}")
    return validation


def main() -> int:
    source_manifest = ensure_source_snapshot()
    feature_report = run_build(
        base_features_path=DEFAULT_BASE_FEATURES,
        source_path=DEFAULT_SOURCE,
        output_path=RELATIVE_FEATURES,
        output_registry_path=RELATIVE_REGISTRY,
        report_path=RELATIVE_REPORT,
    )
    validation = build_labels_and_validate()

    if int(feature_report.get("future_source_rows", -1)) != 0:
        raise SystemExit("V3-013 source-date leakage check failed")
    if int(feature_report.get("relative_strength_feature_count", 0)) != 12:
        raise SystemExit("V3-013 expected exactly twelve relative-strength features")

    print(
        json.dumps(
            {
                "stage": "V3-013_FEATURES",
                "status": "PASS",
                "feature_version": feature_report.get("feature_version"),
                "rows": feature_report.get("rows"),
                "baseline_feature_count": feature_report.get("baseline_feature_count"),
                "relative_strength_feature_count": feature_report.get(
                    "relative_strength_feature_count"
                ),
                "total_feature_count": feature_report.get("total_feature_count"),
                "source_start": source_manifest.get("start"),
                "source_end": source_manifest.get("end"),
                "snapshot_sha256": source_manifest.get("snapshot_sha256"),
                "dataset_validation": validation.get("status"),
                "next": "V3-013 controlled relative-strength ablation",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
