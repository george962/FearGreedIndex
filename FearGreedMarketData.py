#!/usr/bin/env python3
"""Build an analysis-ready Fear & Greed + S&P 500 daily dataset.

The script deliberately loads and validates the existing Fear & Greed CSV
before importing or calling yfinance. If there are no usable Fear & Greed
observations, it exits successfully without making a market-data request.

Two datasets are maintained:

* data/spx_daily.csv
    Incremental raw daily ^GSPC OHLCV cache.
* data/fear_greed_spx_daily.csv
    Fear & Greed observations aligned to the most recent S&P 500 trading day,
    plus backward-looking market features and explicitly named forward-looking
    outcome columns for later research.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_FEAR_GREED_INPUT = Path("data/fear_greed_daily.csv")
DEFAULT_SPX_CACHE = Path("data/spx_daily.csv")
DEFAULT_OUTPUT = Path("data/fear_greed_spx_daily.csv")
DEFAULT_SYMBOL = "^GSPC"
DEFAULT_CONTEXT_DAYS = 450
DEFAULT_OVERLAP_DAYS = 10
DEFAULT_TIMEOUT = 30
MARKET_TIMEZONE = "America/New_York"
DATA_SOURCE = "Yahoo Finance via yfinance"

FEAR_GREED_REQUIRED_COLUMNS = [
    "Date",
    "Value",
    "Rating",
    "Source Timestamp UTC",
]

SPX_CACHE_COLUMNS = [
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

PAST_RETURN_HORIZONS = (1, 5, 10, 20, 60, 252)
SMA_HORIZONS = (20, 50, 200)
FORWARD_RETURN_HORIZONS = (1, 5, 10, 20, 60)
FORWARD_PATH_HORIZONS = (5, 20, 60)


class DataValidationError(RuntimeError):
    """Raised when an input or downloaded dataset is malformed."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally collect ^GSPC daily data only after usable "
            "Fear & Greed observations exist, then build an analysis-ready "
            "combined CSV."
        )
    )
    parser.add_argument(
        "--fear-greed-input",
        type=Path,
        default=DEFAULT_FEAR_GREED_INPUT,
        help="existing daily Fear & Greed CSV",
    )
    parser.add_argument(
        "--spx-cache",
        type=Path,
        default=DEFAULT_SPX_CACHE,
        help="incremental raw S&P 500 cache CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="combined analysis-ready output CSV",
    )
    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        help="Yahoo Finance symbol (default: ^GSPC)",
    )
    parser.add_argument(
        "--context-days",
        type=int,
        default=DEFAULT_CONTEXT_DAYS,
        help=(
            "calendar days fetched before the earliest Fear & Greed market "
            "date so long lookback features can be calculated"
        ),
    )
    parser.add_argument(
        "--overlap-days",
        type=int,
        default=DEFAULT_OVERLAP_DAYS,
        help="days re-fetched when extending an existing SPX cache",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="yfinance request timeout in seconds",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="refresh the recent overlap even when the cache appears current",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a machine-readable result",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate numeric command-line settings."""
    if args.context_days < 370:
        raise ValueError(
            "--context-days must be at least 370 to support 252-trading-day "
            "features"
        )
    if args.overlap_days < 0:
        raise ValueError("--overlap-days cannot be negative")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")
    if not str(args.symbol).strip():
        raise ValueError("--symbol cannot be empty")


def previous_weekday(day: date) -> date:
    """Return the latest Monday-Friday date on or before *day*."""
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def load_fear_greed(path: Path) -> pd.DataFrame:
    """Load, validate, and normalize the daily Fear & Greed CSV."""
    if not path.exists():
        raise DataValidationError(f"Fear & Greed input does not exist: {path}")

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [
        column
        for column in FEAR_GREED_REQUIRED_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise DataValidationError(
            f"{path} is missing required columns: {', '.join(missing)}"
        )

    frame = frame[FEAR_GREED_REQUIRED_COLUMNS].copy()
    frame["Value"] = frame["Value"].str.strip()
    frame = frame.loc[frame["Value"] != ""].copy()

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "fear_greed_date_utc",
                "fear_greed_market_date",
                "fear_greed_value",
                "fear_greed_rating",
                "fear_greed_source_timestamp_utc",
            ]
        )

    try:
        frame["fear_greed_date_utc"] = pd.to_datetime(
            frame["Date"],
            format="%Y-%m-%d",
            errors="raise",
        ).dt.normalize()
        frame["fear_greed_value"] = pd.to_numeric(
            frame["Value"],
            errors="raise",
        )
        frame["fear_greed_source_timestamp_utc"] = pd.to_datetime(
            frame["Source Timestamp UTC"],
            utc=True,
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise DataValidationError(
            f"{path} contains an invalid date, timestamp, or score"
        ) from error

    invalid_scores = ~frame["fear_greed_value"].between(0, 100)
    if invalid_scores.any():
        bad_values = frame.loc[invalid_scores, "fear_greed_value"].tolist()
        raise DataValidationError(
            f"Fear & Greed scores must be between 0 and 100: {bad_values}"
        )

    source_in_market_tz = frame[
        "fear_greed_source_timestamp_utc"
    ].dt.tz_convert(MARKET_TIMEZONE)
    frame["fear_greed_market_date"] = pd.to_datetime(
        source_in_market_tz.dt.date
    )
    frame["fear_greed_rating"] = (
        frame["Rating"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", " ", regex=False)
    )

    frame = frame.sort_values(
        ["fear_greed_date_utc", "fear_greed_source_timestamp_utc"]
    )
    frame = frame.drop_duplicates(
        subset=["fear_greed_date_utc"],
        keep="last",
    )

    return frame[
        [
            "fear_greed_date_utc",
            "fear_greed_market_date",
            "fear_greed_value",
            "fear_greed_rating",
            "fear_greed_source_timestamp_utc",
        ]
    ].reset_index(drop=True)


def load_spx_cache(path: Path) -> pd.DataFrame:
    """Load a prior SPX cache, returning an empty normalized frame if absent."""
    if not path.exists():
        return pd.DataFrame(columns=SPX_CACHE_COLUMNS)

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in SPX_CACHE_COLUMNS if column not in frame]
    if missing:
        raise DataValidationError(
            f"{path} is missing required columns: {', '.join(missing)}"
        )

    frame = frame[SPX_CACHE_COLUMNS].copy()
    try:
        frame["date"] = pd.to_datetime(
            frame["date"],
            format="%Y-%m-%d",
            errors="raise",
        ).dt.normalize()
        for column in [
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ]:
            frame[column] = pd.to_numeric(
                frame[column].replace("", pd.NA),
                errors="coerce",
            )
    except (TypeError, ValueError) as error:
        raise DataValidationError(f"Invalid data in {path}") from error

    frame = frame.dropna(subset=["date", "close"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    return frame.reset_index(drop=True)


def determine_fetch_window(
    fear_greed: pd.DataFrame,
    cache: pd.DataFrame,
    *,
    context_days: int,
    overlap_days: int,
    today: date,
    force_refresh: bool,
) -> tuple[date, date] | None:
    """Return an efficient yfinance [start, end) window, or None if current."""
    earliest_market_date = fear_greed["fear_greed_market_date"].min().date()
    latest_market_date = fear_greed["fear_greed_market_date"].max().date()
    earliest_required = earliest_market_date - timedelta(days=context_days)
    latest_required = max(previous_weekday(today), previous_weekday(latest_market_date))
    end_exclusive = max(today, latest_required) + timedelta(days=2)

    if cache.empty:
        return earliest_required, end_exclusive

    cache_start = cache["date"].min().date()
    cache_end = cache["date"].max().date()

    if cache_start > earliest_required:
        return earliest_required, end_exclusive

    if cache_end < latest_required:
        start = max(
            earliest_required,
            cache_end - timedelta(days=overlap_days),
        )
        return start, end_exclusive

    if force_refresh:
        start = max(
            earliest_required,
            cache_end - timedelta(days=overlap_days),
        )
        return start, end_exclusive

    return None


def fetch_spx_history(
    symbol: str,
    start: date,
    end: date,
    timeout: int,
) -> pd.DataFrame:
    """Fetch raw daily index OHLCV data from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf
    except ImportError as error:
        raise RuntimeError(
            "yfinance is not installed; run pip install -r requirements.txt"
        ) from error

    try:
        history = yf.Ticker(symbol).history(
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            actions=False,
            auto_adjust=False,
            back_adjust=False,
            repair=True,
            keepna=False,
            prepost=False,
            rounding=False,
            timeout=timeout,
            raise_errors=True,
        )
    except Exception as error:  # yfinance exposes several version-specific errors.
        raise RuntimeError(
            f"yfinance failed for {symbol} from {start} through {end}: {error}"
        ) from error

    if history is None or history.empty:
        raise RuntimeError(
            f"yfinance returned no {symbol} data from {start} through {end}"
        )

    return normalize_downloaded_history(history, symbol)


def normalize_downloaded_history(
    history: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """Normalize a yfinance history frame into the cache schema."""
    frame = history.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            column[0] if isinstance(column, tuple) else column
            for column in frame.columns
        ]

    required = ["Open", "High", "Low", "Close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataValidationError(
            "Downloaded SPX history is missing columns: " + ", ".join(missing)
        )

    if "Adj Close" not in frame.columns:
        frame["Adj Close"] = frame["Close"]
    if "Volume" not in frame.columns:
        frame["Volume"] = pd.NA

    index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    if index.tz is not None:
        index = index.tz_convert(MARKET_TIMEZONE)

    normalized = pd.DataFrame(
        {
            "date": pd.to_datetime(index.date),
            "symbol": symbol,
            "open": pd.to_numeric(frame["Open"], errors="coerce").to_numpy(),
            "high": pd.to_numeric(frame["High"], errors="coerce").to_numpy(),
            "low": pd.to_numeric(frame["Low"], errors="coerce").to_numpy(),
            "close": pd.to_numeric(frame["Close"], errors="coerce").to_numpy(),
            "adj_close": pd.to_numeric(
                frame["Adj Close"], errors="coerce"
            ).to_numpy(),
            "volume": pd.to_numeric(
                frame["Volume"], errors="coerce"
            ).to_numpy(),
            "data_source": DATA_SOURCE,
        }
    )
    normalized = normalized.dropna(subset=["date", "close"])
    normalized = normalized.sort_values("date").drop_duplicates(
        "date", keep="last"
    )
    return normalized[SPX_CACHE_COLUMNS].reset_index(drop=True)


def merge_spx_cache(
    existing: pd.DataFrame,
    fetched: pd.DataFrame,
) -> pd.DataFrame:
    """Merge cache frames with newly fetched rows taking precedence."""
    if existing.empty:
        merged = fetched.copy()
    elif fetched.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, fetched], ignore_index=True)

    merged = merged.sort_values("date").drop_duplicates("date", keep="last")
    return merged[SPX_CACHE_COLUMNS].reset_index(drop=True)


def add_fear_greed_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add backward-looking Fear & Greed trend features."""
    result = frame.sort_values("fear_greed_market_date").copy()
    score = result["fear_greed_value"]

    result["fear_greed_change_1d"] = score.diff(1)
    result["fear_greed_change_5d"] = score - score.shift(5)
    result["fear_greed_sma_5d"] = score.rolling(5, min_periods=5).mean()
    result["fear_greed_sma_20d"] = score.rolling(20, min_periods=20).mean()
    rolling_mean = score.rolling(20, min_periods=20).mean()
    rolling_std = score.rolling(20, min_periods=20).std(ddof=1)
    result["fear_greed_zscore_20d"] = (score - rolling_mean) / rolling_std
    return result


def future_window_stat(
    series: pd.Series,
    horizon: int,
    operation: str,
) -> pd.Series:
    """Calculate a statistic over the next *horizon* rows, excluding today."""
    shifted = pd.concat(
        [series.shift(-step) for step in range(1, horizon + 1)],
        axis=1,
    )
    complete = shifted.notna().sum(axis=1).eq(horizon)

    if operation == "min":
        result = shifted.min(axis=1, skipna=True)
    elif operation == "max":
        result = shifted.max(axis=1, skipna=True)
    else:
        raise ValueError(f"Unsupported future-window operation: {operation}")

    return result.where(complete)


def add_spx_features(cache: pd.DataFrame) -> pd.DataFrame:
    """Add backward-looking features and clearly labeled future outcomes."""
    result = cache.sort_values("date").copy().reset_index(drop=True)
    close = result["close"].astype(float)
    open_price = result["open"].astype(float)
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    previous_close = close.shift(1)
    daily_decimal_return = close.pct_change(fill_method=None)

    result["spx_previous_close"] = previous_close
    result["spx_daily_return_pct"] = daily_decimal_return * 100
    result["spx_gap_pct"] = (open_price / previous_close - 1) * 100
    result["spx_intraday_return_pct"] = (close / open_price - 1) * 100
    result["spx_range_pct"] = ((high - low) / previous_close) * 100

    for horizon in PAST_RETURN_HORIZONS:
        result[f"spx_return_{horizon}d_pct"] = (
            close.pct_change(horizon, fill_method=None) * 100
        )

    for horizon in SMA_HORIZONS:
        sma = close.rolling(horizon, min_periods=horizon).mean()
        result[f"spx_sma_{horizon}d"] = sma
        result[f"spx_close_vs_sma_{horizon}d_pct"] = (close / sma - 1) * 100

    result["spx_volatility_20d_annualized_pct"] = (
        daily_decimal_return.rolling(20, min_periods=20).std(ddof=1)
        * math.sqrt(252)
        * 100
    )
    rolling_high = close.rolling(252, min_periods=252).max()
    result["spx_drawdown_from_252d_high_pct"] = (close / rolling_high - 1) * 100

    for horizon in FORWARD_RETURN_HORIZONS:
        result[f"outcome_forward_return_{horizon}d_pct"] = (
            close.shift(-horizon) / close - 1
        ) * 100

    for horizon in FORWARD_PATH_HORIZONS:
        future_low = future_window_stat(low, horizon, "min")
        future_high = future_window_stat(high, horizon, "max")
        result[f"outcome_forward_max_drawdown_{horizon}d_pct"] = (
            future_low / close - 1
        ) * 100
        result[f"outcome_forward_max_gain_{horizon}d_pct"] = (
            future_high / close - 1
        ) * 100

    result = result.rename(
        columns={
            "date": "spx_date",
            "symbol": "spx_symbol",
            "open": "spx_open",
            "high": "spx_high",
            "low": "spx_low",
            "close": "spx_close",
            "adj_close": "spx_adj_close",
            "volume": "spx_volume",
            "data_source": "spx_data_source",
        }
    )
    return result


def build_analysis_dataset(
    fear_greed: pd.DataFrame,
    cache: pd.DataFrame,
) -> pd.DataFrame:
    """Align Fear & Greed rows to the latest available SPX trading day."""
    if cache.empty:
        raise DataValidationError("SPX cache is empty")

    fear_features = add_fear_greed_features(fear_greed)
    spx_features = add_spx_features(cache)

    joined = pd.merge_asof(
        fear_features.sort_values("fear_greed_market_date"),
        spx_features.sort_values("spx_date"),
        left_on="fear_greed_market_date",
        right_on="spx_date",
        direction="backward",
        allow_exact_matches=True,
    )

    if joined["spx_date"].isna().any():
        missing_dates = joined.loc[
            joined["spx_date"].isna(), "fear_greed_market_date"
        ].dt.strftime("%Y-%m-%d").tolist()
        raise DataValidationError(
            "No SPX trading day could be matched for: " + ", ".join(missing_dates)
        )

    joined["spx_match_lag_calendar_days"] = (
        joined["fear_greed_market_date"] - joined["spx_date"]
    ).dt.days
    joined["spx_same_trading_day"] = joined[
        "spx_match_lag_calendar_days"
    ].eq(0)
    joined["fear_greed_day_of_week"] = joined[
        "fear_greed_market_date"
    ].dt.day_name()
    joined["spx_day_of_week"] = joined["spx_date"].dt.day_name()

    preferred = [
        "fear_greed_date_utc",
        "fear_greed_market_date",
        "fear_greed_source_timestamp_utc",
        "fear_greed_value",
        "fear_greed_rating",
        "fear_greed_change_1d",
        "fear_greed_change_5d",
        "fear_greed_sma_5d",
        "fear_greed_sma_20d",
        "fear_greed_zscore_20d",
        "fear_greed_day_of_week",
        "spx_date",
        "spx_match_lag_calendar_days",
        "spx_same_trading_day",
        "spx_day_of_week",
        "spx_symbol",
        "spx_open",
        "spx_high",
        "spx_low",
        "spx_close",
        "spx_adj_close",
        "spx_volume",
        "spx_previous_close",
        "spx_daily_return_pct",
        "spx_gap_pct",
        "spx_intraday_return_pct",
        "spx_range_pct",
    ]

    for horizon in PAST_RETURN_HORIZONS:
        preferred.append(f"spx_return_{horizon}d_pct")
    for horizon in SMA_HORIZONS:
        preferred.extend(
            [
                f"spx_sma_{horizon}d",
                f"spx_close_vs_sma_{horizon}d_pct",
            ]
        )
    preferred.extend(
        [
            "spx_volatility_20d_annualized_pct",
            "spx_drawdown_from_252d_high_pct",
            "spx_data_source",
        ]
    )
    for horizon in FORWARD_RETURN_HORIZONS:
        preferred.append(f"outcome_forward_return_{horizon}d_pct")
    for horizon in FORWARD_PATH_HORIZONS:
        preferred.extend(
            [
                f"outcome_forward_max_drawdown_{horizon}d_pct",
                f"outcome_forward_max_gain_{horizon}d_pct",
            ]
        )

    return joined[preferred].sort_values(
        ["fear_greed_market_date", "fear_greed_source_timestamp_utc"]
    ).reset_index(drop=True)


def render_csv(frame: pd.DataFrame) -> str:
    """Render a deterministic CSV with ISO dates and compact numeric values."""
    output = frame.copy()

    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            if isinstance(output[column].dtype, pd.DatetimeTZDtype):
                output[column] = output[column].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                if (
                    column == "date"
                    or column.endswith("_date")
                    or column.endswith("_date_utc")
                ):
                    output[column] = output[column].dt.strftime("%Y-%m-%d")
                else:
                    output[column] = output[column].dt.strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    )

    buffer = io.StringIO(newline="")
    output.to_csv(
        buffer,
        index=False,
        lineterminator="\n",
        float_format="%.6f",
        na_rep="",
    )
    return buffer.getvalue()


def write_if_changed(path: Path, content: str) -> bool:
    """Write text only when its contents differ."""
    existing = path.read_text(encoding="utf-8") if path.exists() else None
    if existing == content:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")
    return True


def run(args: argparse.Namespace, *, today: date | None = None) -> dict[str, Any]:
    """Execute the collection and dataset-building workflow."""
    validate_args(args)
    fear_greed = load_fear_greed(args.fear_greed_input)

    if fear_greed.empty:
        return {
            "status": "no_fear_greed_data",
            "market_request_made": False,
            "fear_greed_rows": 0,
            "spx_rows": 0,
            "analysis_rows": 0,
            "spx_cache_updated": False,
            "analysis_output_updated": False,
        }

    cache = load_spx_cache(args.spx_cache)
    effective_today = today or datetime.now(timezone.utc).date()
    fetch_window = determine_fetch_window(
        fear_greed,
        cache,
        context_days=args.context_days,
        overlap_days=args.overlap_days,
        today=effective_today,
        force_refresh=args.force_refresh,
    )

    market_request_made = fetch_window is not None
    if fetch_window is not None:
        fetched = fetch_spx_history(
            symbol=args.symbol.strip(),
            start=fetch_window[0],
            end=fetch_window[1],
            timeout=args.timeout,
        )
        cache = merge_spx_cache(cache, fetched)

    analysis = build_analysis_dataset(fear_greed, cache)
    cache_updated = write_if_changed(args.spx_cache, render_csv(cache))
    analysis_updated = write_if_changed(args.output, render_csv(analysis))

    return {
        "status": "ok",
        "market_request_made": market_request_made,
        "fetch_start": fetch_window[0].isoformat() if fetch_window else None,
        "fetch_end_exclusive": fetch_window[1].isoformat() if fetch_window else None,
        "fear_greed_rows": int(len(fear_greed)),
        "spx_rows": int(len(cache)),
        "analysis_rows": int(len(analysis)),
        "spx_cache_updated": cache_updated,
        "analysis_output_updated": analysis_updated,
        "spx_cache": str(args.spx_cache),
        "analysis_output": str(args.output),
    }


def main() -> int:
    """Command-line entry point."""
    args = parse_args()

    try:
        result = run(args)
    except (DataValidationError, RuntimeError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 1

    if args.json:
        print(json.dumps(result, sort_keys=True))
    elif result["status"] == "no_fear_greed_data":
        print("No usable Fear & Greed rows; no yfinance request was made.")
    else:
        request_status = (
            "requested yfinance data"
            if result["market_request_made"]
            else "used the current SPX cache"
        )
        print(
            f"Built {result['analysis_rows']} combined rows and {request_status}."
        )
        print(
            f"SPX cache: {result['spx_rows']} rows "
            f"({'updated' if result['spx_cache_updated'] else 'unchanged'})"
        )
        print(
            f"Analysis CSV: "
            f"{'updated' if result['analysis_output_updated'] else 'unchanged'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
