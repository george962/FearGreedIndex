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

from v3.data_sources.fetch_vix import (  # noqa: E402
    DEFAULT_MANIFEST as VIX_SOURCE_MANIFEST,
    DEFAULT_OUTPUT as VIX_SNAPSHOT,
    DEFAULT_SOURCE_URL,
    fetch_bytes,
    write_snapshot,
)
from v3.evaluation.validate_dataset import validate_frames  # noqa: E402
from v3.features.build_vix_features import (  # noqa: E402
    DEFAULT_BASE_FEATURES,
    DEFAULT_OUTPUT as VIX_FEATURES,
    DEFAULT_REGISTRY as VIX_REGISTRY,
    DEFAULT_REPORT as VIX_MISSINGNESS_REPORT,
    run_build,
)
from v3.labels.build_labels import build_labels, load_market  # noqa: E402

VIX_LABELS = ROOT / "v3" / "data" / "labels_daily_vix.parquet"
VIX_MODEL_DATASET = ROOT / "v3" / "data" / "model_dataset_vix.parquet"
VIX_VALIDATION_REPORT = ROOT / "v3" / "reports" / "dataset_validation_vix.json"
MARKET_DATA = ROOT / "data" / "spx_daily.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_vix_snapshot() -> dict[str, object]:
    if not VIX_SNAPSHOT.exists() or not VIX_SOURCE_MANIFEST.exists():
        payload = fetch_bytes(DEFAULT_SOURCE_URL)
        return write_snapshot(
            payload,
            source_url=DEFAULT_SOURCE_URL,
            output=VIX_SNAPSHOT,
            manifest_output=VIX_SOURCE_MANIFEST,
        )

    manifest = json.loads(VIX_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    expected = str(manifest.get("snapshot_sha256", ""))
    actual = _sha256(VIX_SNAPSHOT)
    if not expected or expected != actual:
        raise SystemExit(
            "Checked-in VIX snapshot hash does not match vix_source.json; "
            "refresh the snapshot intentionally rather than silently refetching it."
        )
    return manifest


def build_vix_labels_and_validate() -> dict[str, object]:
    base = pd.read_parquet(DEFAULT_BASE_FEATURES, engine="pyarrow")
    expanded = pd.read_parquet(VIX_FEATURES, engine="pyarrow")

    # V3-011 must be a pure feature-family addition. It may not alter any
    # baseline row, baseline feature value, or decision date.
    assert_frame_equal(
        expanded[base.columns].reset_index(drop=True),
        base.reset_index(drop=True),
        check_dtype=True,
    )

    market = load_market(MARKET_DATA)
    labels = build_labels(expanded["decision_date"], market)
    model_dataset = expanded.merge(
        labels,
        on="decision_date",
        how="left",
        validate="one_to_one",
    )

    VIX_LABELS.parent.mkdir(parents=True, exist_ok=True)
    VIX_VALIDATION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(VIX_LABELS, index=False, engine="pyarrow")
    model_dataset.to_parquet(VIX_MODEL_DATASET, index=False, engine="pyarrow")

    validation = validate_frames(expanded, labels)
    VIX_VALIDATION_REPORT.write_text(
        json.dumps(validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if validation.get("status") != "PASS":
        raise SystemExit(f"V3-011 expanded dataset validation failed: {validation}")
    return validation


def main() -> int:
    source_manifest = ensure_vix_snapshot()
    feature_report = run_build(
        base_features_path=DEFAULT_BASE_FEATURES,
        vix_path=VIX_SNAPSHOT,
        output_path=VIX_FEATURES,
        registry_path=VIX_REGISTRY,
        report_path=VIX_MISSINGNESS_REPORT,
    )
    validation = build_vix_labels_and_validate()

    if int(feature_report.get("vix_source_date_future_rows", -1)) != 0:
        raise SystemExit("V3-011 detected future VIX source dates")
    if int(feature_report.get("vix_feature_count", 0)) != 8:
        raise SystemExit("V3-011 expected exactly eight VIX candidate features")

    print(
        json.dumps(
            {
                "stage": "V3-011",
                "status": "PASS",
                "feature_version": feature_report.get("feature_version"),
                "rows": feature_report.get("rows"),
                "baseline_feature_count": feature_report.get("baseline_feature_count"),
                "vix_feature_count": feature_report.get("vix_feature_count"),
                "total_feature_count": feature_report.get("total_feature_count"),
                "vix_source_start": source_manifest.get("start"),
                "vix_source_end": source_manifest.get("end"),
                "vix_normalized_sha256": source_manifest.get("normalized_sha256"),
                "vix_snapshot_sha256": source_manifest.get("snapshot_sha256"),
                "dataset_validation": validation.get("status"),
                "next": "V3-012 controlled VIX ablation",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
