#!/usr/bin/env python3
"""Collect and verify immutable research-only STAB-004 shadow predictions.

The prediction lane is downstream of EVID-001. It may consume the frozen historical
research dataset and append-only forward feature rows, but it never consumes
post-cutoff realized outcomes and it can never change the production decision or
sizing policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v3.evaluation.exp006_opportunity import AS_OF, add_opportunity_targets, eligible_training_mask
from v3.evaluation.stab001_past_only import select_stable_features
from v3.evaluation.stab003_consensus_abstention import (
    build_consensus,
    consensus_score,
    evid001_outcomes_are_sealed,
    select_short_memory_features,
)
from v3.evaluation.stab004_rolling_normalization import (
    MIN_REPRESENTATIVES,
    redundancy_clusters,
    rolling_score_percentiles,
)
from v3.evidence.append_forward_snapshot import GENESIS_HASH, canonical_json, canonical_scalar, sha256_bytes
from v3.models.common import load_feature_registry, validate_feature_columns

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_DATASET = ROOT / "v3" / "data" / "model_dataset_treasury.parquet"
FEATURE_REGISTRY = ROOT / "v3" / "reports" / "feature_registry_treasury.json"
FORWARD_FEATURE_LEDGER = ROOT / "v3" / "evidence" / "forward_feature_ledger.csv"
PREDICTION_LEDGER = ROOT / "v3" / "evidence" / "shadow_prediction_ledger.csv"
STAB004_EVALUATION = ROOT / "v3" / "reports" / "stab004_evaluation.json"
STAB004_MANIFEST = ROOT / "v3" / "methodology" / "STAB-004" / "manifest.json"

METHOD_ID = "STAB-004"
COLLECTOR_VERSION = "v3-shadow-prediction-v1"
PRODUCTION_EFFECT = "NONE"
SIZING_MULTIPLIER = "1"

BASE_FIELDS = [
    "decision_date",
    "research_cutoff",
    "method_id",
    "method_status",
    "feature_set_version",
    "forward_feature_row_sha256",
    "feature_vector_sha256",
    "source_feature_sha256",
    "feature_registry_sha256",
    "method_evaluation_sha256",
    "method_manifest_sha256",
    "representative_set_sha256",
    "representative_feature_count",
    "representative_features",
    "raw_score",
    "rolling_percentile",
    "reference_count",
    "call_state",
    "collector_version",
    "production_effect",
    "sizing_multiplier",
]
LEDGER_FIELDS = BASE_FIELDS + ["previous_row_sha256", "row_sha256"]


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_float(value: Any) -> str:
    numeric = float(value)
    if not np.isfinite(numeric):
        return "NA"
    return format(numeric, ".17g")


def read_prediction_ledger(path: Path = PREDICTION_LEDGER) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != LEDGER_FIELDS:
            raise ValueError("Shadow prediction ledger header does not match frozen schema")
        return [{key: (value or "") for key, value in row.items()} for row in reader]


def write_prediction_ledger(rows: list[dict[str, str]], path: Path = PREDICTION_LEDGER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def representative_payload(representatives: list[dict[str, Any]]) -> list[dict[str, str]]:
    fields = [
        "feature",
        "family",
        "direction",
        "long_weight",
        "short_weight",
        "consensus_weight",
        "cluster_id",
        "cluster_size",
    ]
    payload: list[dict[str, str]] = []
    for item in representatives:
        payload.append({field: canonical_scalar(item.get(field)) for field in fields})
    return payload


def _load_frozen_context() -> tuple[dict[str, Any], str, list[str], pd.DataFrame, list[dict[str, Any]]]:
    _require(evid001_outcomes_are_sealed(), "Shadow prediction collection refuses to run after EVID-001 outcomes are opened")

    evaluation = json.loads(STAB004_EVALUATION.read_text(encoding="utf-8"))
    _require(evaluation.get("method_id") == METHOD_ID, "Unexpected shadow method id")
    _require(evaluation.get("status") == "complete_reject", "Frozen STAB-004 status drift")
    _require(evaluation.get("evid001_outcomes_opened") is False, "Frozen STAB-004 evidence says EVID-001 was opened")
    _require(evaluation.get("champion_selected") is False, "Shadow challenger cannot be a champion")
    _require(evaluation.get("v3_019_eligible") is False, "Shadow challenger cannot be V3-019 eligible")
    _require(float(evaluation.get("current_sizing_multiplier", 0.0)) == 1.0, "Shadow challenger cannot change sizing")

    feature_version, features = load_feature_registry(FEATURE_REGISTRY)
    historical = pd.read_parquet(HISTORICAL_DATASET, engine="pyarrow").copy()
    historical["decision_date"] = pd.to_datetime(historical["decision_date"], errors="raise").dt.normalize()
    historical = historical.loc[historical["decision_date"].le(AS_OF)].sort_values("decision_date").reset_index(drop=True)
    historical = add_opportunity_targets(historical)
    validate_feature_columns(historical, features)

    train = historical.loc[eligible_training_mask(historical, AS_OF)].copy()
    _require(not train.empty, "No legally mature pre-cutoff training rows for shadow scoring")
    long_selected, _ = select_stable_features(train, features)
    short_selected, _ = select_short_memory_features(train, features)
    consensus = build_consensus(long_selected, short_selected)
    representatives, _ = redundancy_clusters(train, consensus)
    _require(len(representatives) >= MIN_REPRESENTATIVES, "Shadow scoring lacks frozen STAB-004 structural support")
    return evaluation, feature_version, features, historical, representatives


def build_prediction_candidates() -> list[dict[str, str]]:
    """Recompute all legal forward predictions from frozen relationships and sealed features."""
    evaluation, feature_version, features, historical, representatives = _load_frozen_context()

    forward = pd.read_csv(FORWARD_FEATURE_LEDGER)
    if forward.empty:
        return []
    forward["decision_date"] = pd.to_datetime(forward["decision_date"], errors="raise").dt.normalize()
    _require(forward["decision_date"].gt(AS_OF).all(), "Forward feature ledger contains exposed-history date")
    _require(not forward["decision_date"].duplicated().any(), "Forward feature ledger contains duplicate dates")
    _require(forward["decision_date"].is_monotonic_increasing, "Forward feature ledger dates are not increasing")
    _require(forward["feature_set_version"].astype(str).eq(feature_version).all(), "Forward feature version drift")
    _require(pd.to_numeric(forward["feature_count"], errors="raise").eq(len(features)).all(), "Forward feature count drift")
    validate_feature_columns(forward, features)

    registry_hash = sha256_path(FEATURE_REGISTRY)
    _require(forward["feature_registry_sha256"].astype(str).eq(registry_hash).all(), "Forward feature registry hash drift")

    train = historical.loc[eligible_training_mask(historical, AS_OF)].copy()
    long_selected, _ = select_stable_features(train, features)
    short_selected, _ = select_short_memory_features(train, features)
    consensus = build_consensus(long_selected, short_selected)

    historical_scores = historical[["decision_date", *features]].copy()
    forward_scores = forward[["decision_date", *features]].copy()
    scoring_frame = (
        pd.concat([historical_scores, forward_scores], ignore_index=True)
        .sort_values("decision_date")
        .drop_duplicates(subset="decision_date", keep="last")
        .reset_index(drop=True)
    )
    raw_scores = consensus_score(train, scoring_frame, representatives)
    target_dates = set(forward["decision_date"].tolist())
    rolling = rolling_score_percentiles(scoring_frame["decision_date"], raw_scores, target_dates)
    scored = forward[
        [
            "decision_date",
            "row_sha256",
            "feature_vector_sha256",
            "source_feature_sha256",
            "feature_registry_sha256",
        ]
    ].merge(rolling, on="decision_date", how="left", validate="one_to_one")
    _require(scored["call_state"].notna().all(), "Shadow scoring failed to produce a call state for every forward row")

    rep_payload = representative_payload(representatives)
    rep_hash = sha256_bytes(canonical_json(rep_payload))
    rep_names = "|".join(str(item["feature"]) for item in representatives)
    evaluation_hash = sha256_path(STAB004_EVALUATION)
    manifest_hash = sha256_path(STAB004_MANIFEST)
    cutoff = pd.Timestamp(AS_OF).strftime("%Y-%m-%d")

    candidates: list[dict[str, str]] = []
    for _, row in scored.sort_values("decision_date").iterrows():
        candidates.append(
            {
                "decision_date": pd.Timestamp(row["decision_date"]).strftime("%Y-%m-%d"),
                "research_cutoff": cutoff,
                "method_id": METHOD_ID,
                "method_status": str(evaluation["status"]),
                "feature_set_version": feature_version,
                "forward_feature_row_sha256": str(row["row_sha256"]),
                "feature_vector_sha256": str(row["feature_vector_sha256"]),
                "source_feature_sha256": str(row["source_feature_sha256"]),
                "feature_registry_sha256": str(row["feature_registry_sha256"]),
                "method_evaluation_sha256": evaluation_hash,
                "method_manifest_sha256": manifest_hash,
                "representative_set_sha256": rep_hash,
                "representative_feature_count": str(len(representatives)),
                "representative_features": rep_names,
                "raw_score": _canonical_float(row["raw_score"]),
                "rolling_percentile": _canonical_float(row["rolling_percentile"]),
                "reference_count": str(int(row["reference_count"])),
                "call_state": str(row["call_state"]),
                "collector_version": COLLECTOR_VERSION,
                "production_effect": PRODUCTION_EFFECT,
                "sizing_multiplier": SIZING_MULTIPLIER,
            }
        )
    return candidates


def finalize_candidate(candidate: dict[str, str], previous_row_sha256: str) -> dict[str, str]:
    _require(set(candidate) == set(BASE_FIELDS), "Shadow prediction candidate schema drift")
    row = {field: str(candidate[field]) for field in BASE_FIELDS}
    row["previous_row_sha256"] = previous_row_sha256
    payload = {field: row[field] for field in BASE_FIELDS + ["previous_row_sha256"]}
    row["row_sha256"] = sha256_bytes(canonical_json(payload))
    return row


def verify_hash_chain(rows: list[dict[str, str]]) -> None:
    previous = GENESIS_HASH
    prior_date: str | None = None
    for row in rows:
        _require(row.get("previous_row_sha256") == previous, "Shadow prediction hash chain is broken")
        if prior_date is not None:
            _require(row["decision_date"] > prior_date, "Shadow prediction dates are not strictly increasing")
        payload = {field: row[field] for field in BASE_FIELDS + ["previous_row_sha256"]}
        _require(row.get("row_sha256") == sha256_bytes(canonical_json(payload)), "Shadow prediction row hash mismatch")
        previous = row["row_sha256"]
        prior_date = row["decision_date"]


def reconcile_prediction_rows(
    existing: list[dict[str, str]], candidates: list[dict[str, str]]
) -> tuple[list[dict[str, str]], int]:
    """Preserve an immutable prefix and append only new chronological candidates."""
    verify_hash_chain(existing)
    existing_dates = [row["decision_date"] for row in existing]
    candidate_dates = [row["decision_date"] for row in candidates]
    _require(len(candidate_dates) == len(set(candidate_dates)), "Shadow candidates contain duplicate dates")
    _require(existing_dates == candidate_dates[: len(existing_dates)], "Shadow ledger is not an exact prefix of legal candidates")

    for index, row in enumerate(existing):
        candidate = candidates[index]
        for field in BASE_FIELDS:
            _require(row[field] == candidate[field], f"Immutable shadow prediction drift for {row['decision_date']} field {field}")

    output = [dict(row) for row in existing]
    previous = output[-1]["row_sha256"] if output else GENESIS_HASH
    appended = 0
    for candidate in candidates[len(output) :]:
        row = finalize_candidate(candidate, previous)
        output.append(row)
        previous = row["row_sha256"]
        appended += 1
    verify_hash_chain(output)
    return output, appended


def collect_shadow_predictions(path: Path = PREDICTION_LEDGER) -> dict[str, Any]:
    existing = read_prediction_ledger(path)
    candidates = build_prediction_candidates()
    rows, appended = reconcile_prediction_rows(existing, candidates)
    write_prediction_ledger(rows, path)
    return {
        "status": "PASS",
        "method_id": METHOD_ID,
        "collector_version": COLLECTOR_VERSION,
        "rows": len(rows),
        "appended": appended,
        "first_prediction_date": rows[0]["decision_date"] if rows else None,
        "last_prediction_date": rows[-1]["decision_date"] if rows else None,
        "chain_head": rows[-1]["row_sha256"] if rows else GENESIS_HASH,
        "production_effect": PRODUCTION_EFFECT,
        "sizing_multiplier": 1.0,
        "evid001_outcomes_opened": False,
    }


def verify_shadow_predictions(path: Path = PREDICTION_LEDGER) -> dict[str, Any]:
    rows = read_prediction_ledger(path)
    candidates = build_prediction_candidates()
    verify_hash_chain(rows)
    candidate_dates = [row["decision_date"] for row in candidates]
    ledger_dates = [row["decision_date"] for row in rows]
    _require(ledger_dates == candidate_dates[: len(ledger_dates)], "Shadow prediction ledger is not a gap-free prefix of forward evidence")
    for index, row in enumerate(rows):
        candidate = candidates[index]
        for field in BASE_FIELDS:
            _require(row[field] == candidate[field], f"Shadow prediction semantic mismatch for {row['decision_date']} field {field}")
        _require(row["production_effect"] == PRODUCTION_EFFECT, "Shadow prediction gained production effect")
        _require(row["sizing_multiplier"] == SIZING_MULTIPLIER, "Shadow prediction changed sizing")

    return {
        "status": "PASS",
        "method_id": METHOD_ID,
        "ledger_rows": len(rows),
        "available_forward_rows": len(candidates),
        "lag_rows": len(candidates) - len(rows),
        "first_prediction_date": rows[0]["decision_date"] if rows else None,
        "last_prediction_date": rows[-1]["decision_date"] if rows else None,
        "chain_head": rows[-1]["row_sha256"] if rows else GENESIS_HASH,
        "outcomes_present": False,
        "champion_selected": False,
        "v3_019_eligible": False,
        "sizing_multiplier": 1.0,
        "production_effect": PRODUCTION_EFFECT,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["collect", "verify"])
    parser.add_argument("--ledger", type=Path, default=PREDICTION_LEDGER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = collect_shadow_predictions(args.ledger) if args.action == "collect" else verify_shadow_predictions(args.ledger)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
