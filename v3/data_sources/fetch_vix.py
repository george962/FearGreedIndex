#!/usr/bin/env python3
"""Fetch and normalize the first-party Cboe VIX daily-history snapshot.

The normalized snapshot intentionally keeps only the fields required by V3-011:
trading date and the published VIX close. The checked-in snapshot, not a live
network call, is the reproducible input used by later model experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "v3" / "data" / "vix_daily.csv"
DEFAULT_MANIFEST = ROOT / "v3" / "data" / "vix_source.json"
DEFAULT_SOURCE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
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
                raise RuntimeError("Cboe VIX response was empty")
            return payload
        except (HTTPError, URLError, TimeoutError, RuntimeError) as error:
            last_error = error
            if attempt < attempts:
                time.sleep(float(attempt))
    raise RuntimeError(f"Unable to fetch Cboe VIX history after {attempts} attempts") from last_error


def normalize_vix_csv(payload: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(payload))
    normalized_names = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in frame.columns
    }

    date_source = next(
        (
            normalized_names[name]
            for name in ("date", "trade_date", "trading_date")
            if name in normalized_names
        ),
        None,
    )
    close_source = next(
        (
            normalized_names[name]
            for name in ("close", "vix_close", "closing_value", "value")
            if name in normalized_names
        ),
        None,
    )
    if date_source is None or close_source is None:
        raise ValueError(
            "Cboe VIX CSV must contain recognizable date and close columns; "
            f"received={list(frame.columns)}"
        )

    result = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_source], errors="raise").dt.normalize(),
            "vix_close": pd.to_numeric(frame[close_source], errors="raise").astype(float),
        }
    ).dropna(subset=["date", "vix_close"])
    result = result.sort_values("date").reset_index(drop=True)

    if result["date"].duplicated().any():
        duplicate_dates = result.loc[
            result["date"].duplicated(keep=False), "date"
        ].dt.date.astype(str).unique()
        raise ValueError(f"Duplicate VIX dates: {duplicate_dates[:5].tolist()}")
    if (result["vix_close"] <= 0.0).any():
        raise ValueError("VIX close values must be positive")
    if result.empty:
        raise ValueError("Normalized VIX dataset is empty")
    return result


def normalized_csv_bytes(frame: pd.DataFrame) -> bytes:
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    text = normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )
    return text.encode("utf-8")


def write_snapshot(
    payload: bytes,
    *,
    source_url: str,
    output: Path,
    manifest_output: Path,
) -> dict[str, object]:
    frame = normalize_vix_csv(payload)
    normalized_payload = normalized_csv_bytes(frame)

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(normalized_payload)

    manifest: dict[str, object] = {
        "source": "Cboe VIX Index historical daily data",
        "source_url": source_url,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256_bytes(payload),
        "normalized_sha256": sha256_bytes(normalized_payload),
        "rows": int(len(frame)),
        "start": frame["date"].min().date().isoformat(),
        "end": frame["date"].max().date().isoformat(),
        "normalized_columns": ["date", "vix_close"],
        "usage": (
            "Research snapshot for V3-011/V3-012. Later experiments consume this "
            "checked-in normalized snapshot rather than refetching live history."
        ),
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    args = parse_args()
    payload = fetch_bytes(
        args.source_url,
        timeout=args.timeout,
        attempts=args.attempts,
    )
    manifest = write_snapshot(
        payload,
        source_url=args.source_url,
        output=args.output,
        manifest_output=args.manifest_output,
    )
    print(
        f"Wrote {args.output} ({manifest['rows']} rows, "
        f"{manifest['start']} through {manifest['end']})"
    )
    print(f"Wrote {args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
