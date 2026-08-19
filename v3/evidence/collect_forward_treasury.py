#!/usr/bin/env python3
"""Capture post-cutoff Treasury observations without mutating frozen research data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from v3.data_sources.fetch_treasury import fetch_snapshot_frame

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FROZEN_SOURCE = ROOT / "v3" / "data" / "treasury_daily.csv"
DEFAULT_FORWARD_SOURCE = ROOT / "v3" / "evidence" / "forward_treasury_source.csv"
DEFAULT_MANIFEST = ROOT / "v3" / "evidence" / "forward_lane_manifest.json"
GENESIS_HASH = "0" * 64
SOURCE_COLUMNS = [
    "observation_date",
    "dgs2",
    "dgs10",
    "captured_on_date",
    "dgs2_source_sha256",
    "dgs10_source_sha256",
    "previous_row_sha256",
    "row_sha256",
]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _canonical_float(value: Any) -> str:
    if value is None or pd.isna(value):
        raise ValueError("Treasury value is missing")
    return format(float(value), ".17g")


def _iso_date(value: Any, field: str) -> str:
    if value is None or pd.isna(value):
        raise ValueError(f"{field} is missing")
    return pd.Timestamp(value).normalize().strftime("%Y-%m-%d")


def read_forward_source(path: Path = DEFAULT_FORWARD_SOURCE) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SOURCE_COLUMNS:
            raise ValueError("Forward Treasury source header drift")
        return [{key: (value or "") for key, value in row.items()} for row in reader]


def write_forward_source(rows: list[dict[str, str]], path: Path = DEFAULT_FORWARD_SOURCE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _row_hash(row: dict[str, str]) -> str:
    payload = {key: row[key] for key in SOURCE_COLUMNS if key != "row_sha256"}
    return _sha256(_canonical_json(payload))


def verify_forward_treasury_source(
    path: Path = DEFAULT_FORWARD_SOURCE,
    *,
    frozen_source_path: Path = DEFAULT_FROZEN_SOURCE,
) -> dict[str, Any]:
    frozen = pd.read_csv(frozen_source_path)
    frozen["date"] = pd.to_datetime(frozen["date"], errors="raise").dt.normalize()
    if frozen.empty:
        raise ValueError("Frozen Treasury source is empty")
    frozen_end = frozen["date"].max().strftime("%Y-%m-%d")

    rows = read_forward_source(path)
    prior = GENESIS_HASH
    prior_date: str | None = None
    for row in rows:
        observation = _iso_date(row["observation_date"], "observation_date")
        captured = _iso_date(row["captured_on_date"], "captured_on_date")
        if observation <= frozen_end:
            raise ValueError("Forward Treasury source overlaps frozen research snapshot")
        if observation > captured:
            raise ValueError("Treasury observation was captured before its observation date")
        if prior_date is not None and observation <= prior_date:
            raise ValueError("Forward Treasury observations are not strictly increasing")
        if row["previous_row_sha256"] != prior:
            raise ValueError("Forward Treasury hash chain is broken")
        if row["row_sha256"] != _row_hash(row):
            raise ValueError("Forward Treasury row hash mismatch")
        prior = row["row_sha256"]
        prior_date = observation

    return {
        "status": "PASS",
        "rows": len(rows),
        "frozen_end": frozen_end,
        "first_observation_date": rows[0]["observation_date"] if rows else None,
        "last_observation_date": rows[-1]["observation_date"] if rows else None,
        "chain_head": rows[-1]["row_sha256"] if rows else GENESIS_HASH,
    }


def collect_forward_treasury(
    *,
    capture_date: str,
    frozen_source_path: Path = DEFAULT_FROZEN_SOURCE,
    forward_source_path: Path = DEFAULT_FORWARD_SOURCE,
    manifest_path: Path = DEFAULT_MANIFEST,
    fetcher: Callable[..., tuple[pd.DataFrame, dict[str, str]]] = fetch_snapshot_frame,
) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture = _iso_date(capture_date, "capture_date")
    if capture <= str(manifest["research_exposed_through"]):
        raise ValueError("Forward Treasury capture must be after research cutoff")

    frozen = pd.read_csv(frozen_source_path)
    frozen["date"] = pd.to_datetime(frozen["date"], errors="raise").dt.normalize()
    frozen_end_ts = frozen["date"].max()
    rows = read_forward_source(forward_source_path)
    if rows:
        latest_ts = pd.Timestamp(rows[-1]["observation_date"])
        prior_hash = rows[-1]["row_sha256"]
    else:
        latest_ts = frozen_end_ts
        prior_hash = GENESIS_HASH

    # Include a small already-known lookback so the FRED intersection is non-empty
    # even when no new business-day observation has been published yet.
    query_start = (latest_ts - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    fetched, source_hashes = fetcher(start=query_start, end_inclusive=capture)
    fetched = fetched.copy()
    fetched["date"] = pd.to_datetime(fetched["date"], errors="raise").dt.normalize()
    fetched = fetched.loc[
        fetched["date"].gt(latest_ts) & fetched["date"].le(pd.Timestamp(capture))
    ].sort_values("date")

    appended = 0
    for item in fetched.itertuples(index=False):
        row: dict[str, str] = {
            "observation_date": _iso_date(item.date, "observation_date"),
            "dgs2": _canonical_float(item.dgs2),
            "dgs10": _canonical_float(item.dgs10),
            "captured_on_date": capture,
            "dgs2_source_sha256": str(source_hashes.get("DGS2", "")),
            "dgs10_source_sha256": str(source_hashes.get("DGS10", "")),
            "previous_row_sha256": prior_hash,
            "row_sha256": "",
        }
        if not row["dgs2_source_sha256"] or not row["dgs10_source_sha256"]:
            raise ValueError("Forward Treasury capture missing raw source hashes")
        row["row_sha256"] = _row_hash(row)
        rows.append(row)
        prior_hash = row["row_sha256"]
        appended += 1

    write_forward_source(rows, forward_source_path)
    verify_forward_treasury_source(forward_source_path, frozen_source_path=frozen_source_path)
    return appended


def materialize_source_for_decision(
    *,
    decision_date: str,
    output_path: Path,
    frozen_source_path: Path = DEFAULT_FROZEN_SOURCE,
    forward_source_path: Path = DEFAULT_FORWARD_SOURCE,
) -> Path:
    decision = _iso_date(decision_date, "decision_date")
    frozen = pd.read_csv(frozen_source_path)[["date", "dgs2", "dgs10"]].copy()
    frozen["date"] = pd.to_datetime(frozen["date"], errors="raise").dt.normalize()

    forward_rows = read_forward_source(forward_source_path)
    eligible = [row for row in forward_rows if row["captured_on_date"] <= decision]
    if eligible:
        forward = pd.DataFrame(
            {
                "date": [row["observation_date"] for row in eligible],
                "dgs2": [float(row["dgs2"]) for row in eligible],
                "dgs10": [float(row["dgs10"]) for row in eligible],
            }
        )
        forward["date"] = pd.to_datetime(forward["date"], errors="raise").dt.normalize()
        combined = pd.concat([frozen, forward], ignore_index=True)
    else:
        combined = frozen

    combined = combined.loc[combined["date"].le(pd.Timestamp(decision))].copy()
    combined = combined.sort_values("date").drop_duplicates("date", keep="first")
    if combined["date"].duplicated().any():
        raise ValueError("Combined Treasury source has duplicate dates")
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, lineterminator="\n", float_format="%.10g")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-date", required=True)
    parser.add_argument("--frozen-source", type=Path, default=DEFAULT_FROZEN_SOURCE)
    parser.add_argument("--forward-source", type=Path, default=DEFAULT_FORWARD_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count = collect_forward_treasury(
        capture_date=args.capture_date,
        frozen_source_path=args.frozen_source,
        forward_source_path=args.forward_source,
        manifest_path=args.manifest,
    )
    print(f"Forward Treasury observations appended: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
