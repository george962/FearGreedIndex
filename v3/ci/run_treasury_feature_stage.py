#!/usr/bin/env python3
"""Build and validate the V3-015 Treasury feature family on real repository data."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v3.data_sources.fetch_treasury import (  # noqa: E402
    DEFAULT_END_INCLUSIVE,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    DEFAULT_START,
    compressed_snapshot_bytes,
    fetch_snapshot_frame,
    normalized_csv_bytes,
    write_snapshot,
)
from v3.evaluation.validate_dataset import validate_frames  # noqa: E402
from v3.features.build_treasury_features import (  # noqa: E402
    DEFAULT_BASE_FEATURES,
    DEFAULT_OUTPUT as TREASURY_FEATURES_PATH,
    DEFAULT_OUTPUT_REGISTRY,
    DEFAULT_REPORT,
    TREASURY_FEATURES,
    run_build,
)
from v3.labels.build_labels import build_labels, load_market  # noqa: E402

PLAIN_SNAPSHOT = ROOT / "v3" / "data" / "treasury_daily.csv"
TREASURY_LABELS = ROOT / "v3" / "data" / "labels_daily_treasury.parquet"
TREASURY_MODEL_DATASET = ROOT / "v3" / "data" / "model_dataset_treasury.parquet"
VALIDATION_REPORT = ROOT / "v3" / "reports" / "dataset_validation_treasury.json"
MARKET_DATA = ROOT / "data" / "spx_daily.csv"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def ensure_snapshot() -> dict[str, object]:
    """Materialize deterministic gzip from a checked-in plain CSV, or bootstrap it."""
    if PLAIN_SNAPSHOT.exists() and DEFAULT_MANIFEST.exists():
        normalized_payload = PLAIN_SNAPSHOT.read_bytes()
        snapshot_payload = compressed_snapshot_bytes(normalized_payload)
        manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        if _sha256(normalized_payload) != manifest.get("normalized_sha256"):
            raise SystemExit("Checked-in Treasury CSV hash does not match manifest")
        if _sha256(snapshot_payload) != manifest.get("snapshot_sha256"):
            raise SystemExit("Deterministic Treasury gzip hash does not match manifest")
        DEFAULT_OUTPUT.write_bytes(snapshot_payload)
        return manifest

    frame, source_hashes = fetch_snapshot_frame(
        start=DEFAULT_START,
        end_inclusive=DEFAULT_END_INCLUSIVE,
    )
    manifest = write_snapshot(
        frame,
        source_hashes=source_hashes,
        output=DEFAULT_OUTPUT,
        manifest_output=DEFAULT_MANIFEST,
        start=DEFAULT_START,
        end_inclusive=DEFAULT_END_INCLUSIVE,
    )
    PLAIN_SNAPSHOT.write_bytes(gzip.decompress(DEFAULT_OUTPUT.read_bytes()))
    return manifest


def build_labels_and_validate() -> dict[str, object]:
    base = pd.read_parquet(DEFAULT_BASE_FEATURES, engine="pyarrow")
    expanded = pd.read_parquet(TREASURY_FEATURES_PATH, engine="pyarrow")
    assert_frame_equal(
        expanded[base.columns].reset_index(drop=True),
        base.reset_index(drop=True),
        check_dtype=True,
    )

    future = (
        expanded["treasury_date"].notna()
        & expanded["treasury_date"].gt(expanded["decision_date"])
    )
    if future.any():
        raise SystemExit("V3-015 Treasury features contain future source dates")

    market = load_market(MARKET_DATA)
    labels = build_labels(expanded["decision_date"], market)
    model_dataset = expanded.merge(
        labels,
        on="decision_date",
        how="left",
        validate="one_to_one",
    )
    TREASURY_LABELS.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(TREASURY_LABELS, index=False, engine="pyarrow")
    model_dataset.to_parquet(TREASURY_MODEL_DATASET, index=False, engine="pyarrow")
    validation = validate_frames(expanded, labels)
    VALIDATION_REPORT.write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )
    if validation.get("status") != "PASS":
        raise SystemExit(f"Treasury dataset validation failed: {validation}")
    return validation


def main() -> int:
    manifest = ensure_snapshot()
    report = run_build(
        base_features_path=DEFAULT_BASE_FEATURES,
        source_path=DEFAULT_OUTPUT,
        output_path=TREASURY_FEATURES_PATH,
        output_registry_path=DEFAULT_OUTPUT_REGISTRY,
        report_path=DEFAULT_REPORT,
    )
    validation = build_labels_and_validate()

    if int(report.get("future_source_rows", -1)) != 0:
        raise SystemExit("Treasury future-source gate failed")
    if int(report.get("treasury_feature_count", 0)) != len(TREASURY_FEATURES):
        raise SystemExit("Unexpected Treasury feature count")

    print(
        json.dumps(
            {
                "stage": "V3-015_TREASURY_FEATURES",
                "status": "PASS",
                "feature_version": report.get("feature_version"),
                "rows": report.get("rows"),
                "baseline_feature_count": report.get("baseline_feature_count"),
                "treasury_feature_count": report.get("treasury_feature_count"),
                "total_feature_count": report.get("total_feature_count"),
                "source_start": manifest.get("start"),
                "source_end": manifest.get("end"),
                "normalized_sha256": manifest.get("normalized_sha256"),
                "dataset_validation": validation.get("status"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
