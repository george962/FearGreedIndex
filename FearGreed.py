#!/usr/bin/env python3
"""Fetch CNN Fear & Greed data for GitHub Actions and Google Sheets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import ssl
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request

import certifi

from http_retry import fetch_json_with_retry


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
DEFAULT_SPREADSHEET = "FearAndGreed"
DEFAULT_WORKSHEET = "Sheet1"

SHEET_HEADERS = [
    "Date",
    "Time",
    "Value",
    "Site Updated",
]


def fetch_latest(timeout: int = 30, attempts: int = 4) -> dict[str, Any]:
    """Fetch the current index record from CNN's JSON feed."""
    request = Request(DATA_URL, headers=HEADERS)
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    payload = fetch_json_with_retry(
        request,
        timeout=timeout,
        context=ssl_context,
        attempts=attempts,
    )

    record = payload.get("fear_and_greed")

    if not isinstance(record, dict):
        raise RuntimeError(
            "CNN response is missing fear_and_greed"
        )

    return record


def load_cached_latest(path: Path) -> dict[str, Any]:
    """Load the newest repository observation as an explicit fallback."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Fallback dataset is empty: {path}")

    def source_time(row: dict[str, str]) -> datetime:
        raw = row.get("Source Timestamp UTC") or row.get("Date") or ""
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))

    newest = max(rows, key=source_time)
    timestamp = newest.get("Source Timestamp UTC") or newest.get("Date")
    return {
        "score": newest.get("Value"),
        "rating": newest.get("Rating", ""),
        "timestamp": timestamp,
    }


def parse_record(
    record: dict[str, Any],
) -> tuple[str, float, str, datetime]:
    """Return data date, score, rating, and source timestamp."""
    try:
        score = float(record["score"])
        timestamp = str(record["timestamp"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "CNN returned an unexpected data format"
        ) from error

    try:
        source_time = datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")
        )

        if source_time.tzinfo is None:
            source_time = source_time.replace(
                tzinfo=timezone.utc
            )

        data_date = source_time.date().isoformat()
    except ValueError as error:
        raise ValueError(
            "CNN returned an invalid timestamp"
        ) from error

    rating = str(record.get("rating", "")).strip()

    return data_date, score, rating, source_time


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


def write_github_output(
    name: str,
    value: str,
) -> None:
    """Write a single-line or multiline GitHub Actions output."""
    output_file = os.environ.get("GITHUB_OUTPUT")

    if not output_file:
        return

    output_path = Path(output_file)

    with output_path.open(
        "a",
        encoding="utf-8",
    ) as file:
        if "\n" in value or "\r" in value:
            delimiter = f"EOF_{uuid.uuid4().hex}"

            file.write(f"{name}<<{delimiter}\n")
            file.write(value)
            file.write("\n")
            file.write(f"{delimiter}\n")
        else:
            file.write(f"{name}={value}\n")


def set_github_outputs(
    data_date: str,
    value: float,
    rating: str,
    alert_type: str,
    low_threshold: float,
    high_threshold: float,
) -> None:
    """Expose results to later GitHub Actions steps."""
    label = (
        rating.replace("_", " ").title()
        or "Unknown"
    )

    if alert_type == "low":
        title = f"Fear & Greed LOW Alert: {value:g}"
        condition = (
            "The index is at or below the low threshold "
            f"of {low_threshold:g}."
        )
    elif alert_type == "high":
        title = f"Fear & Greed HIGH Alert: {value:g}"
        condition = (
            "The index is at or above the high threshold "
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
            f"- **Data date:** {data_date}",
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

    write_github_output(
        "alert_type",
        alert_type,
    )
    write_github_output(
        "date",
        data_date,
    )
    write_github_output(
        "value",
        f"{value:g}",
    )
    write_github_output(
        "rating",
        label,
    )
    write_github_output(
        "issue_title",
        title,
    )
    write_github_output(
        "issue_body",
        issue_body,
    )


def load_service_account_info() -> dict[str, Any]:
    """Load Google service-account credentials from a GitHub secret."""
    raw_json = os.environ.get(
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "",
    ).strip()

    if not raw_json:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON environment variable "
            "is missing"
        )

    try:
        info = json.loads(raw_json)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON"
        ) from error

    if not isinstance(info, dict):
        raise RuntimeError(
            "Google service-account JSON must be an object"
        )

    return info


def format_age(
    source_time: datetime,
    checked_time: datetime,
) -> str:
    """Format how old the CNN reading is."""
    seconds = max(
        0,
        int(
            (
                checked_time - source_time
            ).total_seconds()
        ),
    )

    minutes = seconds // 60

    if minutes < 1:
        return "a minute ago"

    if minutes == 1:
        return "1 minute ago"

    if minutes < 60:
        return f"{minutes} minutes ago"

    hours = minutes // 60

    if hours == 1:
        return "1 hour ago"

    if hours < 24:
        return f"{hours} hours ago"

    days = hours // 24

    if days == 1:
        return "1 day ago"

    return f"{days} days ago"


def ensure_sheet_headers(worksheet: Any) -> None:
    """Ensure row 1 contains the expected four-column headers."""
    current_headers = worksheet.row_values(1)

    normalized_headers = [
        str(value).strip()
        for value in current_headers[:4]
    ]

    if normalized_headers != SHEET_HEADERS:
        worksheet.update(
            range_name="A1:D1",
            values=[SHEET_HEADERS],
            value_input_option="USER_ENTERED",
        )


def get_latest_recorded_value(
    worksheet: Any,
) -> int | None:
    """
    Return the newest recorded whole-number value.

    Since newest records are inserted into row 2,
    the latest Value cell is C2.
    """
    cell_value = worksheet.acell("C2").value

    if cell_value is None:
        return None

    cell_value = str(cell_value).strip()

    if not cell_value:
        return None

    try:
        return int(float(cell_value) + 0.5)
    except (TypeError, ValueError):
        return None


def update_google_sheet(
    checked_time: datetime,
    source_time: datetime,
    value: float,
    spreadsheet_name: str,
    worksheet_name: str,
    append_unchanged: bool,
) -> bool:
    """
    Insert Fear & Greed data into the four-column sheet.

    New rows are inserted at row 2 so the newest entry
    always appears directly below the header.
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError as error:
        raise RuntimeError(
            "Google Sheets packages are missing; "
            "install gspread and google-auth"
        ) from error

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    credentials = Credentials.from_service_account_info(
        load_service_account_info(),
        scopes=scopes,
    )

    client = gspread.authorize(credentials)

    try:
        spreadsheet = client.open(
            spreadsheet_name
        )
    except gspread.SpreadsheetNotFound as error:
        raise RuntimeError(
            f'Google Sheet "{spreadsheet_name}" was not found '
            "or not shared with the service account"
        ) from error

    try:
        worksheet = spreadsheet.worksheet(
            worksheet_name
        )
    except gspread.WorksheetNotFound as error:
        raise RuntimeError(
            f'Worksheet "{worksheet_name}" was not found in '
            f'"{spreadsheet_name}"'
        ) from error

    ensure_sheet_headers(worksheet)

    display_value = int(value + 0.5)

    if not append_unchanged:
        latest_value = get_latest_recorded_value(
            worksheet
        )

        if latest_value == display_value:
            print(
                "Google Sheet unchanged: latest recorded "
                f"whole-number value is {display_value}"
            )
            return False

    checked_date = checked_time.strftime(
        "%m/%d/%Y"
    )
    checked_clock_time = checked_time.strftime(
        "%H:%M:%S"
    )
    site_updated = format_age(
        source_time,
        checked_time,
    )

    worksheet.insert_row(
        [
            checked_date,
            checked_clock_time,
            display_value,
            site_updated,
        ],
        index=2,
        value_input_option="USER_ENTERED",
        inherit_from_before=False,
    )

    print(
        f'Inserted Fear & Greed {display_value} into '
        f'"{spreadsheet_name}" / "{worksheet_name}" '
        f'at {checked_date} {checked_clock_time} '
        f'({site_updated})'
    )

    return True


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Get CNN Fear & Greed data for alerts "
            "or Google Sheets."
        )
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
    )
    parser.add_argument(
        "--update-sheet",
        action="store_true",
    )
    parser.add_argument(
        "--append-unchanged",
        action="store_true",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
    )
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--fallback-file",
        type=Path,
        default=Path("data/fear_greed_daily.csv"),
    )
    parser.add_argument("--max-data-age-hours", type=float, default=96.0)
    parser.add_argument(
        "--low",
        type=float,
        default=DEFAULT_LOW_THRESHOLD,
    )
    parser.add_argument(
        "--high",
        type=float,
        default=DEFAULT_HIGH_THRESHOLD,
    )
    parser.add_argument(
        "--spreadsheet",
        default=os.environ.get(
            "GOOGLE_SPREADSHEET_NAME",
            DEFAULT_SPREADSHEET,
        ),
    )
    parser.add_argument(
        "--worksheet",
        default=os.environ.get(
            "GOOGLE_WORKSHEET_NAME",
            DEFAULT_WORKSHEET,
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Run the Fear & Greed check."""
    args = parse_args()

    if args.timeout <= 0:
        print(
            "Error: --timeout must be greater than zero"
        )
        return 2
    if args.retries <= 0 or args.max_data_age_hours <= 0:
        print("Error: --retries and --max-data-age-hours must be greater than zero")
        return 2

    if not 0 <= args.low <= 100:
        print(
            "Error: --low must be between 0 and 100"
        )
        return 2

    if not 0 <= args.high <= 100:
        print(
            "Error: --high must be between 0 and 100"
        )
        return 2

    if args.low >= args.high:
        print(
            "Error: --low must be lower than --high"
        )
        return 2

    checked_time = datetime.now(timezone.utc).replace(microsecond=0)
    data_source = "cnn_live"
    fetch_warning = None
    try:
        try:
            record = fetch_latest(args.timeout, args.retries)
        except RuntimeError as error:
            record = load_cached_latest(args.fallback_file)
            data_source = "repository_fallback"
            fetch_warning = str(error)
        (
            data_date,
            value,
            rating,
            source_time,
        ) = parse_record(
            record
        )

        data_age_hours = max(
            0.0,
            (checked_time - source_time.astimezone(timezone.utc)).total_seconds() / 3600.0,
        )
        if data_age_hours > args.max_data_age_hours:
            data_status = "stale"
        elif data_source == "repository_fallback":
            data_status = "cached_fallback"
        else:
            data_status = "fresh"

        alert_type = (
            determine_alert_type(value, args.low, args.high)
            if data_status == "fresh"
            else "unavailable"
        )

        checked_at_utc = checked_time.isoformat()

        if args.github_output:
            set_github_outputs(
                data_date,
                value,
                rating,
                alert_type,
                args.low,
                args.high,
            )
            write_github_output("data_status", data_status)
            write_github_output("data_age_hours", f"{data_age_hours:.2f}")
            write_github_output("data_source", data_source)

        sheet_updated = False

        if args.update_sheet and data_status == "fresh":
            sheet_updated = update_google_sheet(
                checked_time=checked_time,
                source_time=source_time,
                value=value,
                spreadsheet_name=args.spreadsheet,
                worksheet_name=args.worksheet,
                append_unchanged=args.append_unchanged,
            )

    except (
        RuntimeError,
        ValueError,
        OSError,
    ) as error:
        print(f"Error: {error}")
        return 1

    label = (
        rating.replace("_", " ").title()
        or "Unknown"
    )

    result: dict[str, Any] = {
        "checked_at_utc": checked_at_utc,
        "date": data_date,
        "value": value,
        "rating": rating,
        "alert_type": alert_type,
        "sheet_updated": sheet_updated,
        "data_status": data_status,
        "data_age_hours": round(data_age_hours, 2),
        "data_source": data_source,
        "fetch_warning": fetch_warning,
    }

    if args.json:
        print(json.dumps(result))
    else:
        print(
            f"Fear & Greed Index: "
            f"{value:g} ({label}, {data_date})"
        )
        print(
            f"Alert status: {alert_type}"
        )
        print(
            f"Data status: {data_status} ({data_age_hours:.1f}h old, {data_source})"
        )
        if fetch_warning:
            print(f"Fetch warning: {fetch_warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
