#!/usr/bin/env python3
"""Fetch and freeze daily U.S. Treasury constant-maturity rates for V3-015.

The research snapshot uses first-party Federal Reserve/FRED graph CSV endpoints
for DGS2 and DGS10. Network access is only used to create/refresh the checked-in
snapshot intentionally; experiments consume the frozen deterministic gzip CSV.
"""

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
DEFAULT_OUTPUT = ROOT / "v3" / "data" / "treasury_daily.csv.gz"
DEFAULT_MANIFEST = ROOT / "v3" / "data" / "treasury_source.json"
DEFAULT_START = "2019-01-01"
DEFAULT_END_INCLUSIVE = "2026-08-18"
SERIES = ("DGS2", "DGS10")
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
                raise RuntimeError("FRED response was empty")
            return payload
        except (HTTPError, URLError, TimeoutError, RuntimeError) as error:
            last_error = error
            if attempt < attempts:
                time.sleep(float(attempt))
    raise RuntimeError(f"Unable to fetch FRED data after {attempts} attempts") from last_error


def series_url(series_id: str, *, start: str, end_inclusive: str) -> str:
    query = urlencode(
        {
            "id": series_id,
            "cosd": pd.Timestamp(start).date().isoformat(),
            "coed": pd.Timestamp(end_inclusive).date().isoformat(),
        }
    )
    return f"{FRED_GRAPH_URL}?{query}"


def normalize_series_csv(payload: bytes, series_id: str) -> pd.DataFrame:
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
    value_source = next(
        (
            column
            for key, column in normalized_names.items()
            if key == series_id.lower()
        ),
        None,
    )
    if date_source is None or value_source is None:
        raise ValueError(
            f"FRED {series_id} CSV must contain date and {series_id}; "
            f"received={list(frame.columns)}"
        )

    result = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_source], errors="raise").dt.normalize(),
            series_id.lower(): pd.to_numeric(frame[value_source], errors="coerce"),
        }
    ).dropna()
    result[series_id.lower()] = result[series_id.lower()].astype(float)
    result = result.sort_values("date").reset_index(drop=True)
    if result.empty:
        raise ValueError(f"Normalized FRED {series_id} dataset is empty")
    if result["date"].duplicated().any():
        raise ValueError(f"FRED {series_id} contains duplicate dates")
    return result


def fetch_snapshot_frame(
    *,
    start: str = DEFAULT_START,
    end_inclusive: str = DEFAULT_END_INCLUSIVE,
    timeout: float = 30.0,
    attempts: int = 3,
) -> tuple[pd.DataFrame, dict[str, str]]:
    frames: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}
    for series_id in SERIES:
        payload = fetch_bytes(
            series_url(series_id, start=start, end_inclusive=end_inclusive),
            timeout=timeout,
            attempts=attempts,
        )
        source_hashes[series_id] = sha256_bytes(payload)
        frames.append(normalize_series_csv(payload, series_id))

    frame = frames[0]
    for other in frames[1:]:
        frame = frame.merge(other, on="date", how="inner", validate="one_to_one")
    frame = frame.sort_values("date").reset_index(drop=True)
    cutoff = pd.Timestamp(end_inclusive).normalize()
    frame = frame.loc[frame["date"].le(cutoff)].copy()
    if frame.empty:
        raise ValueError("Treasury snapshot intersection is empty")
    if frame["date"].duplicated().any():
        raise ValueError("Treasury snapshot contains duplicate dates")
    return frame[["date", "dgs2", "dgs10"]], source_hashes


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
    source_hashes: dict[str, str],
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
        "source": "Federal Reserve Board H.15 via FRED",
        "series": list(SERIES),
        "units": "percent",
        "frequency": "daily business-day observations",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": pd.Timestamp(start).date().isoformat(),
        "requested_end_inclusive": pd.Timestamp(end_inclusive).date().isoformat(),
        "rows": int(len(frame)),
        "start": frame["date"].min().date().isoformat(),
        "end": frame["date"].max().date().isoformat(),
        "normalized_columns": ["date", "dgs2", "dgs10"],
        "source_sha256": source_hashes,
        "normalized_sha256": sha256_bytes(normalized_payload),
        "snapshot_sha256": sha256_bytes(snapshot_payload),
        "snapshot_format": "gzip csv; deterministic mtime=0",
        "observation_semantics": (
            "Treasury market yields are joined by observation date at or before the "
            "end-of-day decision date; portfolio execution remains next-session open."
        ),
        "usage": (
            "Frozen V3-015 Treasury research snapshot. Experiments consume this "
            "checked-in file rather than current FRED history."
        ),
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    args = parse_args()
    frame, source_hashes = fetch_snapshot_frame(
        start=args.start,
        end_inclusive=args.end_inclusive,
        timeout=args.timeout,
        attempts=args.attempts,
    )
    manifest = write_snapshot(
        frame,
        source_hashes=source_hashes,
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
