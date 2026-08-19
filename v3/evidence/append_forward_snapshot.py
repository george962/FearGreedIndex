#!/usr/bin/env python3
"""Append one sealed point-in-time feature snapshot to the untouched forward lane."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "v3" / "evidence" / "forward_lane_manifest.json"
DEFAULT_LEDGER = ROOT / "v3" / "evidence" / "forward_feature_ledger.csv"
DEFAULT_FEATURES = ROOT / "v3" / "data" / "features_daily_treasury.parquet"
DEFAULT_REGISTRY = ROOT / "v3" / "reports" / "feature_registry_treasury.json"
GENESIS_HASH = "0" * 64

LEDGER_PREFIX = [
    "decision_date",
    "research_cutoff",
    "feature_set_version",
    "feature_count",
    "feature_registry_sha256",
    "fear_greed_date",
    "treasury_date",
    "feature_vector_sha256",
    "source_feature_sha256",
    "collector_version",
    "previous_row_sha256",
]
LEDGER_SUFFIX = ["row_sha256"]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_scalar(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, pd.Timestamp):
        return value.normalize().strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "lane_id",
        "research_exposed_through",
        "first_eligible_decision_date",
        "feature_set_version",
        "feature_count",
        "collector_version",
        "forbidden_column_fragments",
        "checkpoint_registry_version",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"Forward manifest missing fields: {missing}")
    return manifest


def load_registry(path: Path, manifest: dict[str, Any]) -> tuple[str, list[str]]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    version = str(registry.get("version", ""))
    features = [str(item["name"]) for item in registry.get("features", [])]
    if version != manifest["feature_set_version"]:
        raise ValueError("Forward lane feature version does not match registry")
    if len(features) != int(manifest["feature_count"]):
        raise ValueError("Forward lane feature count does not match registry")
    if len(features) != len(set(features)):
        raise ValueError("Forward registry has duplicate feature names")
    forbidden = [str(item).lower() for item in manifest["forbidden_column_fragments"]]
    bad = [name for name in features if any(fragment in name.lower() for fragment in forbidden)]
    if bad:
        raise ValueError(f"Forward registry contains forbidden outcome-like features: {bad}")
    return version, features


def expected_columns(features: list[str]) -> list[str]:
    return LEDGER_PREFIX + features + LEDGER_SUFFIX


def read_ledger(path: Path, features: list[str]) -> list[dict[str, str]]:
    expected = expected_columns(features)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected:
            raise ValueError("Forward ledger header does not match frozen schema")
        return [{key: (value or "") for key, value in row.items()} for row in reader]


def write_ledger(path: Path, features: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = expected_columns(features)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _normalize_date(value: Any, name: str) -> str:
    if value is None or pd.isna(value):
        raise ValueError(f"{name} is missing")
    stamp = pd.to_datetime(value, errors="raise")
    if isinstance(stamp, pd.DatetimeIndex):
        raise ValueError(f"{name} must be scalar")
    return pd.Timestamp(stamp).normalize().strftime("%Y-%m-%d")


def build_snapshot_row(
    source_row: pd.Series,
    *,
    decision_date: str,
    manifest: dict[str, Any],
    registry_path: Path,
    feature_names: list[str],
    previous_row_sha256: str,
) -> dict[str, str]:
    cutoff = str(manifest["research_exposed_through"])
    decision = _normalize_date(decision_date, "decision_date")
    if decision <= cutoff:
        raise ValueError("Forward decision date must be strictly after research cutoff")
    if decision < str(manifest["first_eligible_decision_date"]):
        raise ValueError("Forward decision date precedes first eligible date")

    fear_date = _normalize_date(source_row.get("fear_greed_date"), "fear_greed_date")
    treasury_date = _normalize_date(source_row.get("treasury_date"), "treasury_date")
    if fear_date > decision or treasury_date > decision:
        raise ValueError("Forward source date exceeds decision date")

    values = {name: canonical_scalar(source_row[name]) for name in feature_names}
    vector_payload = [[name, values[name]] for name in feature_names]
    vector_hash = sha256_bytes(canonical_json(vector_payload))
    registry_hash = sha256_path(registry_path)
    source_hash = sha256_bytes(
        canonical_json(
            {
                "decision_date": decision,
                "fear_greed_date": fear_date,
                "treasury_date": treasury_date,
                "feature_set_version": manifest["feature_set_version"],
                "feature_registry_sha256": registry_hash,
                "feature_vector_sha256": vector_hash,
            }
        )
    )

    row: dict[str, str] = {
        "decision_date": decision,
        "research_cutoff": cutoff,
        "feature_set_version": str(manifest["feature_set_version"]),
        "feature_count": str(len(feature_names)),
        "feature_registry_sha256": registry_hash,
        "fear_greed_date": fear_date,
        "treasury_date": treasury_date,
        "feature_vector_sha256": vector_hash,
        "source_feature_sha256": source_hash,
        "collector_version": str(manifest["collector_version"]),
        "previous_row_sha256": previous_row_sha256,
    }
    row.update(values)
    row_hash_payload = {key: row[key] for key in LEDGER_PREFIX + feature_names}
    row["row_sha256"] = sha256_bytes(canonical_json(row_hash_payload))
    return row


def append_snapshot(
    *,
    decision_date: str,
    features_path: Path = DEFAULT_FEATURES,
    registry_path: Path = DEFAULT_REGISTRY,
    manifest_path: Path = DEFAULT_MANIFEST,
    ledger_path: Path = DEFAULT_LEDGER,
) -> str:
    manifest = load_manifest(manifest_path)
    _, feature_names = load_registry(registry_path, manifest)
    frame = pd.read_parquet(features_path, engine="pyarrow").copy()

    forbidden = [str(item).lower() for item in manifest["forbidden_column_fragments"]]
    bad_columns = [column for column in frame.columns if any(fragment in str(column).lower() for fragment in forbidden)]
    if bad_columns:
        raise ValueError(f"Forward source table contains forbidden outcome columns: {bad_columns}")

    required = {"decision_date", "fear_greed_date", "treasury_date", *feature_names}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Forward source table missing columns: {missing}")

    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
    wanted = pd.Timestamp(decision_date).normalize()
    selected = frame.loc[frame["decision_date"].eq(wanted)]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one source row for {wanted.date()}, found {len(selected)}")

    rows = read_ledger(ledger_path, feature_names)
    if rows:
        dates = [row["decision_date"] for row in rows]
        if len(dates) != len(set(dates)):
            raise ValueError("Forward ledger contains duplicate dates")
        existing = [row for row in rows if row["decision_date"] == wanted.strftime("%Y-%m-%d")]
        previous = rows[-1]["row_sha256"]
    else:
        existing = []
        previous = GENESIS_HASH

    candidate = build_snapshot_row(
        selected.iloc[0],
        decision_date=wanted.strftime("%Y-%m-%d"),
        manifest=manifest,
        registry_path=registry_path,
        feature_names=feature_names,
        previous_row_sha256=(existing[0]["previous_row_sha256"] if existing else previous),
    )

    if existing:
        if existing[0] == candidate:
            return "IDEMPOTENT"
        raise ValueError("Forward ledger already contains this decision date with different evidence")

    if rows and candidate["decision_date"] <= rows[-1]["decision_date"]:
        raise ValueError("Forward ledger only permits strictly increasing decision dates")

    rows.append(candidate)
    write_ledger(ledger_path, feature_names, rows)
    return "APPENDED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-date", required=True)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status = append_snapshot(
        decision_date=args.decision_date,
        features_path=args.features,
        registry_path=args.registry,
        manifest_path=args.manifest,
        ledger_path=args.ledger,
    )
    print(f"Forward evidence {args.decision_date}: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
