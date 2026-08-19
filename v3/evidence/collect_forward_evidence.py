#!/usr/bin/env python3
"""Seal the latest post-DIAG point-in-time feature row without touching outcomes."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from v3.evidence.append_forward_snapshot import (
    DEFAULT_LEDGER,
    DEFAULT_MANIFEST,
    DEFAULT_REGISTRY,
    append_snapshot,
    load_manifest,
    read_ledger,
)
from v3.evidence.collect_forward_treasury import (
    DEFAULT_FORWARD_SOURCE,
    DEFAULT_FROZEN_SOURCE,
    collect_forward_treasury,
    materialize_source_for_decision,
)
from v3.evidence.verify_forward_lane import DEFAULT_CHECKPOINTS, verify_lane
from v3.features.build_features import DEFAULT_OUTPUT as BASE_FEATURES, run_build as build_base_features
from v3.features.build_treasury_features import run_build as build_treasury_features

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_REGISTRY = ROOT / "v3" / "features" / "feature_registry.json"
DEFAULT_FAMILY_REGISTRY = ROOT / "v3" / "features" / "treasury_features.json"


def _registry_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_base_decision_date(base_features_path: Path = BASE_FEATURES) -> str:
    frame = pd.read_parquet(base_features_path, engine="pyarrow")
    if "decision_date" not in frame.columns or frame.empty:
        raise ValueError("Base feature table has no decision dates")
    dates = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
    return dates.max().strftime("%Y-%m-%d")


def collect_latest_forward_evidence(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    ledger_path: Path = DEFAULT_LEDGER,
    registry_path: Path = DEFAULT_REGISTRY,
    checkpoints_path: Path = DEFAULT_CHECKPOINTS,
    frozen_treasury_path: Path = DEFAULT_FROZEN_SOURCE,
    forward_treasury_path: Path = DEFAULT_FORWARD_SOURCE,
    base_features_path: Path = BASE_FEATURES,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)

    # Rebuild only point-in-time base features. Labels/outcomes are intentionally
    # absent from the forward collection path.
    build_base_features(output_path=base_features_path)
    decision_date = latest_base_decision_date(base_features_path)
    cutoff = str(manifest["research_exposed_through"])
    if decision_date <= cutoff:
        report = verify_lane(
            manifest_path=manifest_path,
            registry_path=registry_path,
            ledger_path=ledger_path,
            checkpoints_path=checkpoints_path,
            treasury_forward_source_path=forward_treasury_path,
            treasury_frozen_source_path=frozen_treasury_path,
        )
        return {
            "status": "NO_POST_CUTOFF_DECISION_AVAILABLE",
            "decision_date": decision_date,
            "treasury_observations_appended": 0,
            "feature_ledger_action": "NONE",
            "lane": report,
        }

    _, feature_names = __import__(
        "v3.evidence.append_forward_snapshot", fromlist=["load_registry"]
    ).load_registry(registry_path, manifest)
    existing_rows = read_ledger(ledger_path, feature_names)
    if existing_rows and decision_date < existing_rows[-1]["decision_date"]:
        raise ValueError("Current base feature date predates forward ledger head")

    treasury_appended = collect_forward_treasury(
        capture_date=decision_date,
        frozen_source_path=frozen_treasury_path,
        forward_source_path=forward_treasury_path,
        manifest_path=manifest_path,
    )

    with tempfile.TemporaryDirectory(prefix="evid001-") as directory:
        work = Path(directory)
        source_path = materialize_source_for_decision(
            decision_date=decision_date,
            output_path=work / "treasury_asof.csv",
            frozen_source_path=frozen_treasury_path,
            forward_source_path=forward_treasury_path,
        )
        treasury_features = work / "features_daily_treasury.parquet"
        generated_registry = work / "feature_registry_treasury.json"
        generated_report = work / "treasury_features_missingness.json"
        build_treasury_features(
            base_features_path=base_features_path,
            source_path=source_path,
            output_path=treasury_features,
            output_registry_path=generated_registry,
            report_path=generated_report,
        )
        if _registry_payload(generated_registry) != _registry_payload(registry_path):
            raise ValueError("Forward generated Treasury registry differs from frozen registry")

        action = append_snapshot(
            decision_date=decision_date,
            features_path=treasury_features,
            registry_path=registry_path,
            manifest_path=manifest_path,
            ledger_path=ledger_path,
        )

    lane = verify_lane(
        manifest_path=manifest_path,
        registry_path=registry_path,
        ledger_path=ledger_path,
        checkpoints_path=checkpoints_path,
        treasury_forward_source_path=forward_treasury_path,
        treasury_frozen_source_path=frozen_treasury_path,
    )
    return {
        "status": "PASS",
        "decision_date": decision_date,
        "treasury_observations_appended": treasury_appended,
        "feature_ledger_action": action,
        "lane": lane,
        "note": "Only the latest observable decision date is sealed. Missed dates are never backfilled.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--frozen-treasury", type=Path, default=DEFAULT_FROZEN_SOURCE)
    parser.add_argument("--forward-treasury", type=Path, default=DEFAULT_FORWARD_SOURCE)
    parser.add_argument("--base-features", type=Path, default=BASE_FEATURES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = collect_latest_forward_evidence(
        manifest_path=args.manifest,
        ledger_path=args.ledger,
        registry_path=args.registry,
        checkpoints_path=args.checkpoints,
        frozen_treasury_path=args.frozen_treasury,
        forward_treasury_path=args.forward_treasury,
        base_features_path=args.base_features,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
