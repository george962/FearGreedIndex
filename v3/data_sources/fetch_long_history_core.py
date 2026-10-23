#!/usr/bin/env python3
"""Freeze and verify the DATA-001 long-history core source snapshots.

This module intentionally reuses the repository's existing acquisition mechanisms:
- S&P 500 (^GSPC): Yahoo Finance through the pinned yfinance path in
  ``FearGreedMarketData.fetch_spx_history``;
- VIX: first-party Cboe historical daily CSV;
- DGS2 / DGS10: Federal Reserve H.15 series through FRED graph CSV endpoints.

Network access is used only to create the initial frozen snapshots. Downstream
research consumes the checked-in deterministic gzip CSV files and verifies their
hashes; it never silently refreshes history.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from FearGreedMarketData import fetch_spx_history
from v3.data_sources.fetch_treasury import fetch_snapshot_frame as fetch_treasury_frame
from v3.data_sources.fetch_vix import (
    DEFAULT_SOURCE_URL as CBOE_VIX_SOURCE_URL,
    fetch_bytes as fetch_vix_bytes,
    normalize_vix_csv,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "v3" / "data" / "long_history"
SPX_SNAPSHOT = DATA_DIR / "spx_daily.csv.gz"
VIX_SNAPSHOT = DATA_DIR / "vix_daily.csv.gz"
TREASURY_SNAPSHOT = DATA_DIR / "treasury_daily.csv.gz"
MANIFEST = DATA_DIR / "source_manifest.json"

DATASET_ID = "DATA-001"
SOURCE_CONTRACT_VERSION = "v3-long-history-source-001"
START = "1990-01-02"
RESEARCH_CUTOFF = "2026-08-18"
SPX_SYMBOL = "^GSPC"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalized_csv_bytes(frame: pd.DataFrame, columns: list[str]) -> bytes:
    normalized = frame.loc[:, columns].copy()
    if "date" in normalized:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.strftime(
            "%Y-%m-%d"
        )
    return normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        na_rep="",
    ).encode("utf-8")


def deterministic_gzip(payload: bytes) -> bytes:
    return gzip.compress(payload, compresslevel=9, mtime=0)


def _write_snapshot(path: Path, normalized_payload: bytes) -> dict[str, str]:
    compressed = deterministic_gzip(normalized_payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    return {
        "normalized_sha256": sha256_bytes(normalized_payload),
        "snapshot_sha256": sha256_bytes(compressed),
    }


def _read_snapshot(path: Path) -> tuple[pd.DataFrame, bytes, bytes]:
    compressed = path.read_bytes()
    normalized = gzip.decompress(compressed)
    frame = pd.read_csv(path)
    return frame, normalized, compressed


def _frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    return {
        "rows": int(len(frame)),
        "start": dates.min().date().isoformat(),
        "end": dates.max().date().isoformat(),
    }


def _freeze_spx() -> tuple[pd.DataFrame, dict[str, Any]]:
    start_date = pd.Timestamp(START).date()
    end_exclusive = pd.Timestamp(RESEARCH_CUTOFF).date() + timedelta(days=1)
    acquired = fetch_spx_history(SPX_SYMBOL, start_date, end_exclusive, timeout=30)
    acquired = acquired.copy()
    acquired["date"] = pd.to_datetime(acquired["date"], errors="raise").dt.normalize()
    acquired = acquired.loc[
        acquired["date"].between(pd.Timestamp(START), pd.Timestamp(RESEARCH_CUTOFF))
    ].copy()
    acquired = acquired.sort_values("date").drop_duplicates("date", keep="last")
    if acquired.empty:
        raise ValueError("DATA-001 SPX acquisition returned no rows")

    acquisition_columns = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "data_source",
    ]
    acquisition_payload = normalized_csv_bytes(acquired, acquisition_columns)
    snapshot_columns = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    normalized_payload = normalized_csv_bytes(acquired, snapshot_columns)
    hashes = _write_snapshot(SPX_SNAPSHOT, normalized_payload)
    summary = _frame_summary(acquired)
    summary.update(
        {
            "source": "Yahoo Finance via pinned yfinance acquisition path",
            "symbol": SPX_SYMBOL,
            "acquisition_frame_sha256": sha256_bytes(acquisition_payload),
            **hashes,
            "columns": snapshot_columns,
        }
    )
    return acquired.loc[:, snapshot_columns], summary


def _freeze_vix() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_payload = fetch_vix_bytes(CBOE_VIX_SOURCE_URL, timeout=30.0, attempts=3)
    frame = normalize_vix_csv(raw_payload)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame = frame.loc[
        frame["date"].between(pd.Timestamp(START), pd.Timestamp(RESEARCH_CUTOFF))
    ].copy()
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    if frame.empty:
        raise ValueError("DATA-001 VIX acquisition returned no rows")

    columns = ["date", "vix_close"]
    normalized_payload = normalized_csv_bytes(frame, columns)
    hashes = _write_snapshot(VIX_SNAPSHOT, normalized_payload)
    summary = _frame_summary(frame)
    summary.update(
        {
            "source": "Cboe VIX Index historical daily data",
            "source_url": CBOE_VIX_SOURCE_URL,
            "source_sha256": sha256_bytes(raw_payload),
            **hashes,
            "columns": columns,
        }
    )
    return frame.loc[:, columns], summary


def _freeze_treasury() -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, source_hashes = fetch_treasury_frame(
        start=START,
        end_inclusive=RESEARCH_CUTOFF,
        timeout=30.0,
        attempts=3,
    )
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame = frame.loc[
        frame["date"].between(pd.Timestamp(START), pd.Timestamp(RESEARCH_CUTOFF))
    ].copy()
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    if frame.empty:
        raise ValueError("DATA-001 Treasury acquisition returned no rows")

    columns = ["date", "dgs2", "dgs10"]
    normalized_payload = normalized_csv_bytes(frame, columns)
    hashes = _write_snapshot(TREASURY_SNAPSHOT, normalized_payload)
    summary = _frame_summary(frame)
    summary.update(
        {
            "source": "Federal Reserve Board H.15 via FRED",
            "series": ["DGS2", "DGS10"],
            "source_sha256": source_hashes,
            **hashes,
            "columns": columns,
        }
    )
    return frame.loc[:, columns], summary


def freeze(*, force: bool = False) -> dict[str, Any]:
    targets = [SPX_SNAPSHOT, VIX_SNAPSHOT, TREASURY_SNAPSHOT, MANIFEST]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "DATA-001 frozen source files already exist; refusing to overwrite: "
            + ", ".join(str(path.relative_to(ROOT)) for path in existing)
        )

    _, spx = _freeze_spx()
    _, vix = _freeze_vix()
    _, treasury = _freeze_treasury()
    manifest: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "source_contract_version": SOURCE_CONTRACT_VERSION,
        "research_start": START,
        "research_cutoff": RESEARCH_CUTOFF,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "spx": spx,
            "vix": vix,
            "treasury": treasury,
        },
        "governance": {
            "cnn_fear_greed_included": False,
            "live_source_dependency_in_experiments": False,
            "champion_selected": False,
            "v3_019_eligible": False,
            "sizing_multiplier": 1.0,
        },
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _verify_one(name: str, path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame, normalized, compressed = _read_snapshot(path)
    if sha256_bytes(normalized) != expected["normalized_sha256"]:
        raise ValueError(f"{name} normalized SHA-256 mismatch")
    if sha256_bytes(compressed) != expected["snapshot_sha256"]:
        raise ValueError(f"{name} snapshot SHA-256 mismatch")
    if list(frame.columns) != expected["columns"]:
        raise ValueError(f"{name} columns differ from frozen manifest")
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f"{name} dates must be unique and increasing")
    if int(len(frame)) != int(expected["rows"]):
        raise ValueError(f"{name} row count differs from frozen manifest")
    if dates.min().date().isoformat() != expected["start"]:
        raise ValueError(f"{name} start date differs from frozen manifest")
    if dates.max().date().isoformat() != expected["end"]:
        raise ValueError(f"{name} end date differs from frozen manifest")
    if dates.min() < pd.Timestamp(START) or dates.max() > pd.Timestamp(RESEARCH_CUTOFF):
        raise ValueError(f"{name} snapshot exceeds DATA-001 frozen research window")
    return _frame_summary(frame)


def verify() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("DATA-001 source manifest has wrong dataset_id")
    if manifest.get("source_contract_version") != SOURCE_CONTRACT_VERSION:
        raise ValueError("DATA-001 source manifest has wrong contract version")
    if manifest.get("research_start") != START or manifest.get("research_cutoff") != RESEARCH_CUTOFF:
        raise ValueError("DATA-001 research window changed")
    governance = manifest.get("governance", {})
    if governance.get("cnn_fear_greed_included") is not False:
        raise ValueError("DATA-001 source contract must exclude CNN Fear & Greed")
    if governance.get("champion_selected") is not False or governance.get("v3_019_eligible") is not False:
        raise ValueError("DATA-001 cannot change champion eligibility")
    if float(governance.get("sizing_multiplier", -1.0)) != 1.0:
        raise ValueError("DATA-001 cannot change sizing")

    sources = manifest["sources"]
    return {
        "dataset_id": DATASET_ID,
        "status": "PASS",
        "research_start": START,
        "research_cutoff": RESEARCH_CUTOFF,
        "spx": _verify_one("spx", SPX_SNAPSHOT, sources["spx"]),
        "vix": _verify_one("vix", VIX_SNAPSHOT, sources["vix"]),
        "treasury": _verify_one("treasury", TREASURY_SNAPSHOT, sources["treasury"]),
        "cnn_fear_greed_included": False,
        "champion_selected": False,
        "v3_019_eligible": False,
        "sizing_multiplier": 1.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "verify"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = freeze(force=args.force) if args.command == "freeze" else verify()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
