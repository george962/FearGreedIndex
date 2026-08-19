#!/usr/bin/env python3
"""Fail-closed integrity checks for the untouched forward evidence lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v3.evidence.append_forward_snapshot import (
    DEFAULT_LEDGER,
    DEFAULT_MANIFEST,
    DEFAULT_REGISTRY,
    GENESIS_HASH,
    LEDGER_PREFIX,
    canonical_json,
    expected_columns,
    load_manifest,
    load_registry,
    read_ledger,
    sha256_bytes,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINTS = ROOT / "v3" / "evidence" / "forward_checkpoints.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_checkpoints(path: Path, manifest: dict[str, Any], ledger_dates: list[str]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("version") == int(manifest["checkpoint_registry_version"]), "Forward checkpoint version drift")
    require(payload.get("lane_id") == manifest["lane_id"], "Forward checkpoint lane id drift")
    require(payload.get("research_exposed_through") == manifest["research_exposed_through"], "Forward checkpoint cutoff drift")
    checkpoints = payload.get("checkpoints", [])
    require(isinstance(checkpoints, list), "Forward checkpoints must be a list")

    prior_end: str | None = None
    ids: set[str] = set()
    for item in checkpoints:
        require(isinstance(item, dict), "Forward checkpoint entry must be an object")
        for field in ("checkpoint_id", "start_date", "end_date", "status"):
            require(bool(item.get(field)), f"Forward checkpoint missing {field}")
        checkpoint_id = str(item["checkpoint_id"])
        require(checkpoint_id not in ids, "Forward checkpoint id duplicated")
        ids.add(checkpoint_id)
        start = pd.Timestamp(item["start_date"]).normalize().strftime("%Y-%m-%d")
        end = pd.Timestamp(item["end_date"]).normalize().strftime("%Y-%m-%d")
        require(start > manifest["research_exposed_through"], "Forward checkpoint starts inside exposed history")
        require(start <= end, "Forward checkpoint start exceeds end")
        if prior_end is not None:
            require(start > prior_end, "Forward checkpoints overlap or are out of order")
        prior_end = end
        require(item["status"] in {"SEALED", "OPENED"}, "Unknown forward checkpoint status")
        if item["status"] == "OPENED":
            require(bool(item.get("opened_for_experiment")), "Opened checkpoint must record consuming experiment")
            require(bool(item.get("opened_on")), "Opened checkpoint must record open date")
            covered = [date for date in ledger_dates if start <= date <= end]
            require(bool(covered), "Opened checkpoint has no collected ledger evidence")
            require(covered[0] >= start and covered[-1] <= end, "Opened checkpoint ledger coverage drift")
            require(end <= ledger_dates[-1], "Opened checkpoint extends beyond collected ledger evidence")


def verify_lane(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    registry_path: Path = DEFAULT_REGISTRY,
    ledger_path: Path = DEFAULT_LEDGER,
    checkpoints_path: Path = DEFAULT_CHECKPOINTS,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    _, feature_names = load_registry(registry_path, manifest)
    rows = read_ledger(ledger_path, feature_names)
    require(expected_columns(feature_names) == LEDGER_PREFIX + feature_names + ["row_sha256"], "Forward schema construction drift")

    cutoff = str(manifest["research_exposed_through"])
    prior_date: str | None = None
    expected_previous = GENESIS_HASH
    for row in rows:
        decision = row["decision_date"]
        require(decision > cutoff, "Forward ledger contains exposed-history date")
        require(row["research_cutoff"] == cutoff, "Forward row cutoff drift")
        require(row["feature_set_version"] == manifest["feature_set_version"], "Forward row feature version drift")
        require(int(row["feature_count"]) == len(feature_names), "Forward row feature count drift")
        require(row["collector_version"] == manifest["collector_version"], "Forward collector version drift")
        require(row["previous_row_sha256"] == expected_previous, "Forward hash chain is broken")
        if prior_date is not None:
            require(decision > prior_date, "Forward ledger dates are not strictly increasing")
        require(row["fear_greed_date"] <= decision, "Forward Fear & Greed source is in the future")
        require(row["treasury_date"] <= decision, "Forward Treasury source is in the future")

        vector_payload = [[name, row[name]] for name in feature_names]
        require(row["feature_vector_sha256"] == sha256_bytes(canonical_json(vector_payload)), "Forward feature-vector hash mismatch")
        source_payload = {
            "decision_date": decision,
            "fear_greed_date": row["fear_greed_date"],
            "treasury_date": row["treasury_date"],
            "feature_set_version": row["feature_set_version"],
            "feature_registry_sha256": row["feature_registry_sha256"],
            "feature_vector_sha256": row["feature_vector_sha256"],
        }
        require(row["source_feature_sha256"] == sha256_bytes(canonical_json(source_payload)), "Forward source-feature hash mismatch")
        row_payload = {key: row[key] for key in LEDGER_PREFIX + feature_names}
        require(row["row_sha256"] == sha256_bytes(canonical_json(row_payload)), "Forward row hash mismatch")
        expected_previous = row["row_sha256"]
        prior_date = decision

    ledger_dates = [row["decision_date"] for row in rows]
    require(len(ledger_dates) == len(set(ledger_dates)), "Forward ledger duplicate dates")
    verify_checkpoints(checkpoints_path, manifest, ledger_dates)

    report = {
        "lane_id": manifest["lane_id"],
        "status": "PASS",
        "research_exposed_through": cutoff,
        "feature_set_version": manifest["feature_set_version"],
        "feature_count": len(feature_names),
        "ledger_rows": len(rows),
        "first_ledger_date": ledger_dates[0] if ledger_dates else None,
        "last_ledger_date": ledger_dates[-1] if ledger_dates else None,
        "chain_head": rows[-1]["row_sha256"] if rows else GENESIS_HASH,
        "outcomes_present": False,
        "champion_selected": False,
        "v3_019_eligible": False,
        "sizing_multiplier": 1.0,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = verify_lane(
        manifest_path=args.manifest,
        registry_path=args.registry,
        ledger_path=args.ledger,
        checkpoints_path=args.checkpoints,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
