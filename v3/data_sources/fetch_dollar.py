#!/usr/bin/env python3
"""Fetch and freeze the Federal Reserve broad U.S. dollar index for V3-015B."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "v3" / "data" / "dollar_daily.csv.gz"
DEFAULT_MANIFEST = ROOT / "v3" / "data" / "dollar_source.json"
DEFAULT_START = "2019-01-01"
DEFAULT_END_INCLUSIVE = "2026-08-18"
SERIES_ID = "DTWEXBGS"
FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end-inclusive", default=DEFAULT_END_INCLUSIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--attempts", type=int, default=3)
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def source_url(*, start: str, end_inclusive: str) -> str:
    return f"{FRED_GRAPH_URL}?" + urlencode(
        {
            "id": SERIES_ID,
            "cosd": pd.Timestamp(start).date().isoformat(),
            "coed": pd.Timestamp(end_inclusive).date().isoformat(),
        }
    )


def fetch_bytes(
    url: str,
    *,
    timeout: float = 30.0,
    attempts: int = 3,
) -> bytes:
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = Request(
            url,
            headers={
                "User-Agent": "FearGreedIndex-v3-research/1.0",
                "Accept": "text/csv,text/plain,*/*",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                payload = response.read()
            if not payload:
                raise RuntimeError("FRED dollar-index response was empty")
            return payload
        except (HTTPError, URLError, TimeoutError, RuntimeError) as error:
            last_error = error
            if attempt < attempts:
                time.sleep(float(attempt))
    raise RuntimeError(
        f"Unable to fetch FRED {SERIES_ID} after {attempts} attempts"
    ) from last_error


def normalize_csv(payload: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(payload))
    normalized_names = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in frame.columns
    }
    date_source = next(
        (
            normalized_names[name]
            for name in ("observation_date", "date")
            if name in normalized_names
        ),
        None,
    )
    value_source = normalized_names.get(SERIES_ID.lower())
    if date_source is None or value_source is None:
        raise ValueError(
            f"FRED {SERIES_ID} CSV must contain date and {SERIES_ID}; "
            f"received={list(frame.columns)}"
        )

    result = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_source], errors="raise").dt.normalize(),
            "dollar_index": pd.to_numeric(frame[value_source], errors="coerce"),
        }
    ).dropna()
    result["dollar_index"] = result["dollar_index"].astype(float)
    result = result.sort_values("date").reset_index(drop=True)
    if result.empty:
        raise ValueError("Normalized broad-dollar dataset is empty")
    if result["date"].duplicated().any():
        raise ValueError("Broad-dollar dataset contains duplicate dates")
    if (result["dollar_index"] <= 0.0).any():
        raise ValueError("Broad-dollar index values must be positive")
    return result


def normalized_csv_bytes(frame: pd.DataFrame) -> bytes:
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    return normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    ).encode("utf-8")


def compressed_snapshot_bytes(normalized_payload: bytes) -> bytes:
    return gzip.compress(normalized_payload, compresslevel=9, mtime=0)


def write_snapshot(
    frame: pd.DataFrame,
    *,
    raw_source_sha256: str,
    output: Path = DEFAULT_OUTPUT,
    manifest_output: Path = DEFAULT_MANIFEST,
    start: str = DEFAULT_START,
    end_inclusive: str = DEFAULT_END_INCLUSIVE,
) -> dict[str, object]:
    normalized_payload = normalized_csv_bytes(frame)
    snapshot_payload = compressed_snapshot_bytes(normalized_payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(snapshot_payload)

    manifest: dict[str, object] = {
        "source": "Federal Reserve Board broad U.S. dollar index via FRED",
        "series": SERIES_ID,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": pd.Timestamp(start).date().isoformat(),
        "requested_end_inclusive": pd.Timestamp(end_inclusive).date().isoformat(),
        "rows": int(len(frame)),
        "start": frame["date"].min().date().isoformat(),
        "end": frame["date"].max().date().isoformat(),
        "normalized_columns": ["date", "dollar_index"],
        "source_sha256": raw_source_sha256,
        "normalized_sha256": sha256_bytes(normalized_payload),
        "snapshot_sha256": sha256_bytes(snapshot_payload),
        "snapshot_format": "gzip csv; deterministic mtime=0",
        "availability_rule": (
            "Conservative research rule: an observation dated T is eligible no earlier "
            "than T+1 calendar day. This avoids assuming same-day H.10 availability."
        ),
        "usage": (
            "Frozen V3-015B broad-dollar research snapshot. Experiments consume "
            "the checked-in normalized data rather than current FRED history."
        ),
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    args = parse_args()
    url = source_url(start=args.start, end_inclusive=args.end_inclusive)
    payload = fetch_bytes(url, timeout=args.timeout, attempts=args.attempts)
    frame = normalize_csv(payload)
    frame = frame.loc[
        frame["date"].le(pd.Timestamp(args.end_inclusive).normalize())
    ].copy()
    manifest = write_snapshot(
        frame,
        raw_source_sha256=sha256_bytes(payload),
        output=args.output,
        manifest_output=args.manifest_output,
        start=args.start,
        end_inclusive=args.end_inclusive,
    )
    print(
        f"Wrote {args.output} ({manifest['rows']} rows, "
        f"{manifest['start']} through {manifest['end']})"
    )
    print(f"Wrote {args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
