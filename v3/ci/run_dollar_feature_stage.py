#!/usr/bin/env python3
"""Build and validate V3-015B broad-dollar features on real repository data."""

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

from v3.data_sources.fetch_dollar import DEFAULT_END_INCLUSIVE, DEFAULT_MANIFEST, DEFAULT_OUTPUT, DEFAULT_START, compressed_snapshot_bytes, fetch_bytes, normalize_csv, sha256_bytes, source_url, write_snapshot  # noqa: E402
from v3.evaluation.validate_dataset import validate_frames  # noqa: E402
from v3.features.build_dollar_features import DEFAULT_BASE_FEATURES, DEFAULT_OUTPUT as FEATURE_PATH, DEFAULT_OUTPUT_REGISTRY, DEFAULT_REPORT, DOLLAR_FEATURES, run_build  # noqa: E402
from v3.labels.build_labels import build_labels, load_market  # noqa: E402

PLAIN_SNAPSHOT = ROOT / "v3" / "data" / "dollar_daily.csv"
MODEL_DATASET = ROOT / "v3" / "data" / "model_dataset_dollar.parquet"
LABELS = ROOT / "v3" / "data" / "labels_daily_dollar.parquet"
VALIDATION_REPORT = ROOT / "v3" / "reports" / "dataset_validation_dollar.json"
MARKET_DATA = ROOT / "data" / "spx_daily.csv"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def ensure_snapshot() -> dict[str, object]:
    if PLAIN_SNAPSHOT.exists() and DEFAULT_MANIFEST.exists():
        normalized = PLAIN_SNAPSHOT.read_bytes()
        compressed = compressed_snapshot_bytes(normalized)
        manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        if _sha(normalized) != manifest.get("normalized_sha256"):
            raise SystemExit("Checked-in dollar CSV hash does not match manifest")
        if _sha(compressed) != manifest.get("snapshot_sha256"):
            raise SystemExit("Deterministic dollar gzip hash does not match manifest")
        DEFAULT_OUTPUT.write_bytes(compressed)
        return manifest

    url = source_url(start=DEFAULT_START, end_inclusive=DEFAULT_END_INCLUSIVE)
    payload = fetch_bytes(url)
    frame = normalize_csv(payload)
    frame = frame.loc[frame["date"].le(pd.Timestamp(DEFAULT_END_INCLUSIVE))].copy()
    manifest = write_snapshot(frame, raw_source_sha256=sha256_bytes(payload))
    PLAIN_SNAPSHOT.write_bytes(gzip.decompress(DEFAULT_OUTPUT.read_bytes()))
    return manifest


def main() -> int:
    manifest = ensure_snapshot()
    report = run_build()
    base = pd.read_parquet(DEFAULT_BASE_FEATURES, engine="pyarrow")
    expanded = pd.read_parquet(FEATURE_PATH, engine="pyarrow")
    assert_frame_equal(expanded[base.columns].reset_index(drop=True), base.reset_index(drop=True), check_dtype=True)
    if int(report["future_available_rows"]) != 0 or int(report["same_day_observation_rows"]) != 0:
        raise SystemExit("Dollar availability-lag gate failed")
    if int(report["dollar_feature_count"]) != len(DOLLAR_FEATURES):
        raise SystemExit("Unexpected dollar feature count")

    labels = build_labels(expanded["decision_date"], load_market(MARKET_DATA))
    model = expanded.merge(labels, on="decision_date", how="left", validate="one_to_one")
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(LABELS, index=False, engine="pyarrow")
    model.to_parquet(MODEL_DATASET, index=False, engine="pyarrow")
    validation = validate_frames(expanded, labels)
    VALIDATION_REPORT.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    if validation.get("status") != "PASS":
        raise SystemExit(f"Dollar dataset validation failed: {validation}")

    print(json.dumps({
        "stage": "V3-015B_DOLLAR_FEATURES",
        "status": "PASS",
        "feature_version": report["feature_version"],
        "rows": report["rows"],
        "dollar_feature_count": report["dollar_feature_count"],
        "total_feature_count": report["total_feature_count"],
        "source_start": manifest.get("start"),
        "source_end": manifest.get("end"),
        "availability_rule": manifest.get("availability_rule"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
