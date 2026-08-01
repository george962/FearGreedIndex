#!/usr/bin/env python3
"""Build a one-row-per-day CNN Fear & Greed history CSV."""

from __future__ import annotations

import argparse
import csv
import io
import json
import ssl
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi


HISTORY_BASE_URL = (
    "https://production.dataviz.cnn.io/index/"
    "fearandgreed/graphdata"
)
DEFAULT_START_DATE = date(2021, 2, 1)
DEFAULT_OUTPUT = Path("data/fear_greed_daily.csv")
CSV_HEADERS = [
    "Date",
    "Value",
    "Rating",
    "Source Timestamp UTC",
]
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


@dataclass(frozen=True)
class DailyRecord:
    """A single daily Fear & Greed observation."""

    day: date
    value: float
    rating: str
    source_time: datetime


def parse_date(value: str) -> date:
    """Parse an ISO date for argparse."""
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "date must use YYYY-MM-DD format"
        ) from error


def normalize_datetime(value: datetime) -> datetime:
    """Return an aware UTC datetime with second precision."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc).replace(microsecond=0)


def parse_iso_timestamp(value: Any) -> datetime:
    """Parse a CNN ISO timestamp."""
    try:
        parsed = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ValueError(
            f"Invalid CNN timestamp: {value!r}"
        ) from error

    return normalize_datetime(parsed)


def parse_millisecond_timestamp(value: Any) -> datetime:
    """Parse a Unix timestamp expressed in milliseconds."""
    try:
        milliseconds = float(value)
        parsed = datetime.fromtimestamp(
            milliseconds / 1000,
            tz=timezone.utc,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Invalid CNN millisecond timestamp: {value!r}"
        ) from error

    return normalize_datetime(parsed)


def normalize_score(value: Any) -> float:
    """Convert a score to a validated float."""
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid Fear & Greed score: {value!r}"
        ) from error

    if not 0 <= score <= 100:
        raise ValueError(
            f"Fear & Greed score is outside 0-100: {score}"
        )

    return score


def rating_from_score(score: float) -> str:
    """Return a CNN-style rating when the feed omits one."""
    if score <= 24:
        return "extreme fear"
    if score <= 44:
        return "fear"
    if score <= 55:
        return "neutral"
    if score <= 75:
        return "greed"
    return "extreme greed"


def normalize_rating(value: Any, score: float) -> str:
    """Normalize CNN's rating text or derive it from the score."""
    rating = str(value or "").strip().lower().replace("_", " ")
    return rating or rating_from_score(score)


def fetch_payload(
    start_date: date,
    timeout: int,
) -> dict[str, Any]:
    """Fetch CNN's historical Fear & Greed JSON payload."""
    url = f"{HISTORY_BASE_URL}/{start_date.isoformat()}"
    request = Request(url, headers=HEADERS)
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

    if not isinstance(payload, dict):
        raise RuntimeError("CNN returned an unexpected payload")

    return payload


def record_from_historical_item(
    item: dict[str, Any],
) -> DailyRecord:
    """Convert one historical CNN item to a daily record."""
    source_time = parse_millisecond_timestamp(item.get("x"))
    score = normalize_score(item.get("y"))

    return DailyRecord(
        day=source_time.date(),
        value=score,
        rating=normalize_rating(item.get("rating"), score),
        source_time=source_time,
    )


def record_from_current_item(
    item: dict[str, Any],
) -> DailyRecord:
    """Convert CNN's current reading to a daily record."""
    source_time = parse_iso_timestamp(item.get("timestamp"))
    score = normalize_score(item.get("score"))

    return DailyRecord(
        day=source_time.date(),
        value=score,
        rating=normalize_rating(item.get("rating"), score),
        source_time=source_time,
    )


def parse_payload(
    payload: dict[str, Any],
    start_date: date,
) -> dict[date, DailyRecord]:
    """Return the newest CNN observation for each UTC date."""
    daily: dict[date, DailyRecord] = {}
    historical = payload.get("fear_and_greed_historical")

    if not isinstance(historical, dict):
        raise RuntimeError(
            "CNN response is missing fear_and_greed_historical"
        )

    items = historical.get("data")

    if not isinstance(items, list):
        raise RuntimeError(
            "CNN historical response is missing its data list"
        )

    for item in items:
        if not isinstance(item, dict):
            continue

        record = record_from_historical_item(item)

        if record.day >= start_date:
            keep_newest_record(daily, record)

    current = payload.get("fear_and_greed")

    if isinstance(current, dict):
        record = record_from_current_item(current)

        if record.day >= start_date:
            keep_newest_record(daily, record)

    if not daily:
        raise RuntimeError("CNN returned no usable historical records")

    return daily


def keep_newest_record(
    records: dict[date, DailyRecord],
    candidate: DailyRecord,
) -> None:
    """Keep only the latest source timestamp for each date."""
    current = records.get(candidate.day)

    if current is None or candidate.source_time >= current.source_time:
        records[candidate.day] = candidate


def load_existing_csv(path: Path) -> dict[date, DailyRecord]:
    """Load existing history so older rows survive truncated API results."""
    if not path.exists():
        return {}

    records: dict[date, DailyRecord] = {}

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != CSV_HEADERS:
            raise RuntimeError(
                f"{path} does not have the expected CSV columns"
            )

        for row in reader:
            try:
                day = date.fromisoformat(row["Date"])
                score = normalize_score(row["Value"])
                source_time = parse_iso_timestamp(
                    row["Source Timestamp UTC"]
                )
            except (KeyError, ValueError) as error:
                raise RuntimeError(
                    f"Invalid row in {path}: {row}"
                ) from error

            record = DailyRecord(
                day=day,
                value=score,
                rating=normalize_rating(row.get("Rating"), score),
                source_time=source_time,
            )
            keep_newest_record(records, record)

    return records


def format_score(value: float) -> str:
    """Format scores compactly while retaining useful precision."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def format_timestamp(value: datetime) -> str:
    """Format an aware datetime as an ISO UTC timestamp."""
    return normalize_datetime(value).isoformat().replace("+00:00", "Z")


def render_csv(records: dict[date, DailyRecord]) -> str:
    """Render records as deterministic CSV text."""
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(CSV_HEADERS)

    for day in sorted(records):
        record = records[day]
        writer.writerow(
            [
                record.day.isoformat(),
                format_score(record.value),
                record.rating,
                format_timestamp(record.source_time),
            ]
        )

    return output.getvalue()


def write_if_changed(path: Path, content: str) -> bool:
    """Write the CSV only when its complete content changed."""
    existing = ""

    if path.exists():
        existing = path.read_text(encoding="utf-8")

    if existing == content:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")
    return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fetch CNN Fear & Greed history and maintain one CSV row "
            "per UTC date."
        )
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=DEFAULT_START_DATE,
        help="first CNN date to request (default: 2021-02-01)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV output path",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds",
    )
    return parser.parse_args()


def main() -> int:
    """Update the daily historical CSV."""
    args = parse_args()

    if args.timeout <= 0:
        print("Error: --timeout must be greater than zero")
        return 2

    try:
        existing = load_existing_csv(args.output)
        fetched = parse_payload(
            fetch_payload(args.start_date, args.timeout),
            args.start_date,
        )

        for record in fetched.values():
            keep_newest_record(existing, record)

        changed = write_if_changed(
            args.output,
            render_csv(existing),
        )
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 1

    dates = sorted(existing)
    status = "updated" if changed else "unchanged"
    print(
        f"{args.output} {status}: {len(dates)} daily records "
        f"from {dates[0]} through {dates[-1]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
