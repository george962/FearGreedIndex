#!/usr/bin/env python3
"""Create a reproducible QQQ/SPY daily-close snapshot for V3-013.

Yahoo/yfinance is used only to acquire the research snapshot. Later experiments
consume the checked-in deterministic gzip CSV and never silently refetch history.
The snapshot is frozen through 2026-08-18 to match the V3-013 experiment cutoff.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "v3" / "data" / "qqq_spy_daily.csv.gz"
DEFAULT_MANIFEST = ROOT / "v3" / "data" / "qqq_spy_source.json"
DEFAULT_START = "2019-01-01"
DEFAULT_END_INCLUSIVE = "2026-08-18"
TICKERS = ("QQQ", "SPY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end-inclusive", default=DEFAULT_END_INCLUSIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _download_close(
    ticker: str,
    *,
    start: str,
    end_inclusive: str,
) -> pd.Series:
    end_exclusive = (pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)).date().isoformat()
    frame = yf.download(
        ticker,
        start=start,
        end=end_exclusive,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if frame.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")

    if isinstance(frame.columns, pd.MultiIndex):
        if "Close" not in frame.columns.get_level_values(0):
            raise ValueError(f"{ticker} download has no Close field")
        close_frame = frame["Close"]
        if isinstance(close_frame, pd.Series):
            close = close_frame
        elif ticker in close_frame.columns:
            close = close_frame[ticker]
        elif close_frame.shape[1] == 1:
            close = close_frame.iloc[:, 0]
        else:
            raise ValueError(f"Unable to resolve {ticker} Close column")
    else:
        if "Close" not in frame.columns:
            raise ValueError(f"{ticker} download has no Close field")
        close = frame["Close"]

    close = pd.to_numeric(close, errors="coerce").dropna().astype(float)
    close.index = pd.to_datetime(close.index, errors="raise").tz_localize(None).normalize()
    close = close.sort_index()
    if close.index.duplicated().any():
        raise ValueError(f"{ticker} download contains duplicate dates")
    if (close <= 0.0).any():
        raise ValueError(f"{ticker} download contains non-positive closes")
    return close


def fetch_snapshot_frame(
    *,
    start: str = DEFAULT_START,
    end_inclusive: str = DEFAULT_END_INCLUSIVE,
) -> pd.DataFrame:
    qqq = _download_close("QQQ", start=start, end_inclusive=end_inclusive)
    spy = _download_close("SPY", start=start, end_inclusive=end_inclusive)
    frame = pd.concat([qqq.rename("qqq_close"), spy.rename("spy_close")], axis=1, join="inner")
    frame = frame.dropna().sort_index().reset_index()
    frame = frame.rename(columns={frame.columns[0]: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame.empty:
        raise ValueError("QQQ/SPY intersection is empty")
    if frame["date"].duplicated().any():
        raise ValueError("QQQ/SPY snapshot has duplicate dates")
    return frame[["date", "qqq_close", "spy_close"]]


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
        "source": "Yahoo Finance via pinned yfinance research dependency",
        "tickers": list(TICKERS),
        "price_field": "unadjusted daily Close",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": pd.Timestamp(start).date().isoformat(),
        "requested_end_inclusive": pd.Timestamp(end_inclusive).date().isoformat(),
        "rows": int(len(frame)),
        "start": frame["date"].min().date().isoformat(),
        "end": frame["date"].max().date().isoformat(),
        "normalized_columns": ["date", "qqq_close", "spy_close"],
        "normalized_sha256": sha256_bytes(normalized_payload),
        "snapshot_sha256": sha256_bytes(snapshot_payload),
        "snapshot_format": "gzip csv; deterministic mtime=0",
        "usage": (
            "Frozen V3-013 research snapshot. Experiments consume this checked-in "
            "file rather than downloading current QQQ/SPY history."
        ),
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    args = parse_args()
    frame = fetch_snapshot_frame(start=args.start, end_inclusive=args.end_inclusive)
    manifest = write_snapshot(
        frame,
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
