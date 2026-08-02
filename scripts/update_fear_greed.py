"""Optionally replace or merge the repository Fear & Greed file.

Set a repository Actions secret named FEAR_GREED_SOURCE_URL to a URL you control
that returns CSV, TSV, or TXT with Date, optional Time, and Value columns.

When the secret is absent, the existing data/fear_greed.tsv file is left alone.
This script intentionally does not scrape a third-party website.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import requests


DESTINATION = Path("data/fear_greed.tsv")


def normalize(name: object) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("&", "and")
        .replace(" ", "_")
        .replace("-", "_")
    )


def parse_table(content: bytes) -> pd.DataFrame:
    text = content.decode("utf-8-sig", errors="replace")
    last_error: Exception | None = None

    for separator in (None, "\t", ",", ";"):
        try:
            frame = pd.read_csv(
                io.StringIO(text),
                sep=separator,
                engine="python",
            )
            if frame.shape[1] >= 2:
                frame.columns = [normalize(column) for column in frame.columns]
                return frame
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise ValueError(f"Could not parse source table: {last_error}")


def standardize(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "date": ["date", "day"],
        "time": ["time"],
        "value": [
            "value",
            "fear_greed",
            "fear_and_greed",
            "fear_greed_index",
            "score",
            "index",
        ],
    }

    rename: dict[str, str] = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            if candidate in frame.columns:
                rename[candidate] = target
                break

    frame = frame.rename(columns=rename)

    if "date" not in frame.columns or "value" not in frame.columns:
        raise ValueError("Source must contain Date and Value columns.")

    if "time" not in frame.columns:
        frame["time"] = "16:00:00"

    result = frame[["date", "time", "value"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result = result.dropna(subset=["date", "value"])
    result = result[result["value"].between(0, 100)]
    result["date"] = result["date"].dt.strftime("%-m/%-d/%Y")
    result["time"] = result["time"].astype(str)
    return result


def load_existing() -> pd.DataFrame:
    if not DESTINATION.exists():
        return pd.DataFrame(columns=["date", "time", "value"])
    return standardize(parse_table(DESTINATION.read_bytes()))


def main() -> None:
    source_url = os.getenv("FEAR_GREED_SOURCE_URL", "").strip()
    if not source_url:
        print("FEAR_GREED_SOURCE_URL is not configured; using repository data.")
        return

    response = requests.get(
        source_url,
        timeout=30,
        headers={"User-Agent": "fear-greed-dashboard/1.0"},
    )
    response.raise_for_status()

    incoming = standardize(parse_table(response.content))
    existing = load_existing()

    combined = pd.concat([existing, incoming], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(
        combined["date"] + " " + combined["time"],
        errors="coerce",
    )
    combined = combined.dropna(subset=["timestamp"])
    combined = combined.drop_duplicates(
        subset=["timestamp", "value"],
        keep="last",
    ).sort_values("timestamp", ascending=False)

    output = combined[["date", "time", "value"]].rename(
        columns={"date": "Date", "time": "Time", "value": "Value"}
    )
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(DESTINATION, sep="\t", index=False)
    print(f"Wrote {len(output)} Fear & Greed observations.")


if __name__ == "__main__":
    main()
