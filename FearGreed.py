#!/usr/bin/env python3
"""Fetch the latest CNN Fear & Greed Index for GitHub Actions."""

from __future__ import annotations

import argparse
import json
import os
import ssl
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi


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

DEFAULT_LOW_THRESHOLD = 25.0
DEFAULT_HIGH_THRESHOLD = 75.0


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
        raise RuntimeError(
            f"Could not reach CNN: {error.reason}"
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "CNN returned invalid JSON"
        ) from error

    record = payload.get("fear_and_greed")

    if not isinstance(record, dict):
        raise RuntimeError(
            "CNN response is missing fear_and_greed"
        )

    return record


def parse_record(
    record: dict[str, Any],
) -> tuple[str, float, str]:
    """Return date, score, and rating from a CNN index record."""
    try:
        score = float(record["score"])
        timestamp = str(record["timestamp"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "CNN returned an unexpected data format"
        ) from error

    try:
        date = datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")
        ).date().isoformat()
    except ValueError as error:
        raise ValueError(
            "CNN returned an invalid timestamp"
        ) from error

    rating = str(record.get("rating", "")).strip()

    return date, score, rating


def determine_alert_type(
    value: float,
    low_threshold: float,
    high_threshold: float,
) -> str:
    """Return low, high, or normal."""
    if value <= low_threshold:
        return "low"

    if value >= high_threshold:
        return "high"

    return "normal"


def write_github_output(name: str, value: str) -> None:
    """Write a step output when running inside GitHub Actions."""
    output_file = os.environ.get("GITHUB_OUTPUT")

    if not output_file:
        return

    with Path(output_file).open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(f"{name}={value}\n")


def set_github_outputs(
    date: str,
    value: float,
    rating: str,
    alert_type: str,
    low_threshold: float,
    high_threshold: float,
) -> None:
    """Expose results to later GitHub Actions steps."""
    label = rating.replace("_", " ").title() or "Unknown"

    if alert_type == "low":
        title = f"Fear & Greed LOW Alert: {value:g}"
        condition = (
            f"The index is at or below the low threshold "
            f"of {low_threshold:g}."
        )
    elif alert_type == "high":
        title = f"Fear & Greed HIGH Alert: {value:g}"
        condition = (
            f"The index is at or above the high threshold "
            f"of {high_threshold:g}."
        )
    else:
        title = f"Fear & Greed Normal: {value:g}"
        condition = (
            "The index is currently inside the normal range."
        )

    issue_body = "\n".join(
        [
            f"<!-- feargreed-alert:{alert_type} -->",
            "## CNN Fear & Greed Index Alert",
            "",
            f"- **Current value:** {value:g}",
            f"- **Rating:** {label}",
            f"- **Data date:** {date}",
            f"- **Low threshold:** {low_threshold:g}",
            f"- **High threshold:** {high_threshold:g}",
            "",
            condition,
            "",
            (
                "This issue was created automatically "
                "by GitHub Actions."
            ),
        ]
    )

    write_github_output("alert_type", alert_type)
    write_github_output("date", date)
    write_github_output("value", f"{value:g}")
    write_github_output("rating", label)
    write_github_output("issue_title", title)
    write_github_output("issue_body", issue_body)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Get the latest CNN Fear & Greed Index "
            "and expose the result to GitHub Actions."
        )
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON",
    )

    parser.add_argument(
        "--github-output",
        action="store_true",
        help="write values to the GitHub Actions output file",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds (default: 30)",
    )

    parser.add_argument(
        "--low",
        type=float,
        default=DEFAULT_LOW_THRESHOLD,
        help=(
            "low alert threshold "
            f"(default: {DEFAULT_LOW_THRESHOLD:g})"
        ),
    )

    parser.add_argument(
        "--high",
        type=float,
        default=DEFAULT_HIGH_THRESHOLD,
        help=(
            "high alert threshold "
            f"(default: {DEFAULT_HIGH_THRESHOLD:g})"
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Run the Fear & Greed check."""
    args = parse_args()

    if args.timeout <= 0:
        print("Error: --timeout must be greater than zero")
        return 2

    if not 0 <= args.low <= 100:
        print("Error: --low must be between 0 and 100")
        return 2

    if not 0 <= args.high <= 100:
        print("Error: --high must be between 0 and 100")
        return 2

    if args.low >= args.high:
        print("Error: --low must be lower than --high")
        return 2

    try:
        date, value, rating = parse_record(
            fetch_latest(args.timeout)
        )
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return 1

    alert_type = determine_alert_type(
        value=value,
        low_threshold=args.low,
        high_threshold=args.high,
    )

    label = rating.replace("_", " ").title() or "Unknown"

    if args.github_output:
        try:
            set_github_outputs(
                date=date,
                value=value,
                rating=rating,
                alert_type=alert_type,
                low_threshold=args.low,
                high_threshold=args.high,
            )
        except OSError as error:
            print(
                f"Error writing GitHub Actions output: {error}"
            )
            return 1

    result = {
        "date": date,
        "value": value,
        "rating": rating,
        "alert_type": alert_type,
        "low_threshold": args.low,
        "high_threshold": args.high,
    }

    if args.json:
        print(json.dumps(result))
    else:
        print(
            f"Fear & Greed Index: "
            f"{value:g} ({label}, {date})"
        )
        print(f"Alert status: {alert_type}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())