#!/usr/bin/env python3
"""Print the latest CNN Fear & Greed Index value."""

from __future__ import annotations

import argparse
import json
import ssl
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi


# MacroMicro chart 50108 tracks CNN's index. Fetching CNN's underlying feed
# directly avoids MacroMicro's interactive Cloudflare and login checks.
DATA_URL = (
    "https://production.dataviz.cnn.io/index/"
    "fearandgreed/graphdata/2021-02-01"
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.cnn.com",
    "Referer": "https://www.cnn.com/",
}


def fetch_latest(timeout: int = 30) -> dict[str, Any]:
    """Fetch the current index record from CNN's JSON feed."""
    request = Request(DATA_URL, headers=HEADERS)
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    try:
        with urlopen(
            request,
            timeout=timeout,
            context=ssl_context,
        ) as response:
            payload = json.load(response)
    except HTTPError as error:
        raise RuntimeError(
            f"CNN returned HTTP {error.code} ({error.reason})"
        ) from error
    except URLError as error:
        raise RuntimeError(f"Could not reach CNN: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError("CNN returned invalid JSON") from error

    record = payload.get("fear_and_greed")
    if not isinstance(record, dict):
        raise RuntimeError("CNN response is missing fear_and_greed")

    return record


def parse_record(record: dict[str, Any]) -> tuple[str, float, str]:
    """Return (date, score, rating) from a CNN index record."""
    try:
        score = float(record["score"])
        timestamp = str(record["timestamp"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("CNN returned an unexpected data format") from error

    try:
        date = datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")
        ).date().isoformat()
    except ValueError as error:
        raise ValueError("CNN returned an invalid timestamp") from error

    rating = str(record.get("rating", "")).strip()
    return date, score, rating


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Get the latest CNN Fear & Greed Index value.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds (default: 30)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.timeout <= 0:
        print("Error: --timeout must be greater than zero")
        return 2

    try:
        date, value, rating = parse_record(fetch_latest(args.timeout))
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return 1

    if args.json:
        print(
            json.dumps(
                {"date": date, "value": value, "rating": rating}
            )
        )
    else:
        label = rating.replace("_", " ").title()
        print(f"Fear & Greed Index: {value:g} ({label}, {date})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
