#!/usr/bin/env python3
# FAST TIMING VERSION 2 — early buy/trim layer integrated
"""
scripts/build_dashboard.py

Build a static "Fear & Greed vs. S&P 500" research dashboard from the data
already stored in this repository.

The headline action is regime-aware. Historical analogs must resemble the
current Fear & Greed setup AND the current S&P 500 price/risk regime. The
signal is compared with a same-regime baseline rather than the unconditional
market average.

This version also performs a point-in-time replay of the SAME decision engine
for every historical Fear & Greed observation. Future outcomes are withheld
until they would have been knowable on each replay date, so historical action
dates do not use future data.

Public functions imported by test_dashboard.py are intentionally preserved:
    parse_fear_dataset
    parse_market_dataset
    parse_combined_dataset
    add_features
    merge_signals

Expected inputs, relative to the repository root:
    data/fear_greed_spx_daily.csv
    data/fear_greed_daily.csv
    data/spx_daily.csv

Generated output, under site/ by default:
    index.html
    styles.css
    app.js
    event_study.csv
    analogs.csv
    full_analysis.csv
    analysis.json
    version.json
    historical_decisions.csv
    decision_changes.csv
    timing_decision_changes.csv
    timing_evaluation.csv
    historical_decisions.json
    decision_history.html
"""

from __future__ import annotations

import argparse
import html as html_lib
import io
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template
from plotly.subplots import make_subplots

try:
    import yfinance as yf
except ImportError:
    yf = None


ROOT = Path(__file__).resolve().parents[1]

MARKET_CONTEXT_COLUMNS = [
    "market_return_3d",
    "market_return_5d",
    "market_return_20d",
    "distance_from_3d_low",
    "distance_from_3d_high",
    "distance_from_20d_high",
    "distance_from_252d_high",
    "distance_from_record_high",
    "distance_from_sma_50",
    "distance_from_sma_200",
    "volatility_20d",
    "market_regime",
]

HISTORY_OUTPUT_COLUMNS = [
    "decision_date",
    "market_date",
    "fear_greed",
    "fg_change_5",
    "market_regime",
    "market_extension",
    "timing_action",
    "timing_tone",
    "timing_side",
    "timing_score",
    "timing_confirmation_count",
    "timing_confirmation_total",
    "timing_recommendation",
    "timing_rationale",
    "recent_fg_low_5",
    "recent_fg_high_5",
    "distance_from_252d_high",
    "market_return_3d",
    "market_return_5d",
    "market_return_20d",
    "action",
    "confidence",
    "sizing_tier",
    "sizing_label",
    "analog_sample",
    "regime_baseline_sample",
    "required_sample",
    "win_rate_5d",
    "average_5d",
    "median_5d",
    "average_20d",
    "regime_baseline_5d",
    "excess_5d",
    "excess_ci_low_5d",
    "excess_ci_high_5d",
    "average_drawdown_20d",
    "positive_checks_passed",
    "positive_checks_total",
    "analog_method",
    "rationale",
]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Settings:
    combined_dataset: str = "data/fear_greed_spx_daily.csv"
    fear_greed_dataset: str = "data/fear_greed_daily.csv"
    market_dataset: str = "data/spx_daily.csv"
    fallback_ticker: str = "^GSPC"

    cooldown_calendar_days: int = 10
    minimum_action_sample: int = 8  # retained for backward-compatible config files
    minimum_regime_sample: int = 20
    minimum_regime_baseline_sample: int = 30
    refresh_seconds: int = 300
    bootstrap_iterations: int = 5000
    bootstrap_seed: int = 42

    horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 20, 60])
    level_thresholds: list[int] = field(
        default_factory=lambda: [15, 20, 25, 30, 35, 40, 50]
    )
    drop_windows: list[int] = field(default_factory=lambda: [1, 3, 5])
    drop_thresholds: list[int] = field(default_factory=lambda: [5, 10, 15, 20])

    analog_level_band: float = 7.0
    analog_change_band: float = 5.0
    analog_high_distance_band: float = 0.03
    analog_return_20d_band: float = 0.05
    analog_sma_200_band: float = 0.08
    analog_volatility_band: float = 0.08
    max_analog_distance: float = 4.5
    maximum_analogs: int = 50

    normal_minimum_excess_5d: float = 0.0025
    moderate_extension_minimum_excess_5d: float = 0.0030
    high_extension_minimum_excess_5d: float = 0.0040
    negative_excess_5d: float = -0.0025

    normal_maximum_average_drawdown_20d: float = -0.04
    high_extension_maximum_average_drawdown_20d: float = -0.03

    sizing_strong_buy_pct: int = 150
    sizing_modest_buy_low_pct: int = 110
    sizing_modest_buy_high_pct: int = 125
    sizing_strong_buy_min_checks_ratio: float = 0.8

    # Fast timing layer. This layer intentionally reacts before the slower
    # analog/bootstrap confirmation model. It uses only information available
    # on the decision date and never uses future returns to create a signal.
    timing_recent_sentiment_observations: int = 5
    timing_buy_watch_score: int = 35
    timing_buy_zone_score: int = 55
    timing_buy_first_tranche_score: int = 75
    timing_buy_small_start_confirmations: int = 3
    timing_buy_first_tranche_confirmations: int = 2
    timing_trim_watch_score: int = 35
    timing_trim_zone_score: int = 55
    timing_trim_strong_score: int = 70
    timing_trim_confirmations: int = 2

    @classmethod
    def load(cls, path: Path) -> "Settings":
        if not path.exists():
            print(f"No config found at {path}; using built-in defaults.")
            return cls()

        raw = json.loads(path.read_text(encoding="utf-8"))
        known_fields = set(cls.__dataclass_fields__)
        filtered = {key: value for key, value in raw.items() if key in known_fields}
        return cls(**filtered)


# =============================================================================
# CSV parsing helpers
# =============================================================================

def slugify(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def sniff_read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV/TSV whose delimiter and column casing may vary."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    last_error: Optional[Exception] = None

    for separator in (None, ",", "\t", ";"):
        try:
            frame = pd.read_csv(io.StringIO(text), sep=separator, engine="python")
        except Exception as error:  # noqa: BLE001
            last_error = error
            continue

        if frame.shape[1] >= 2:
            frame.columns = [slugify(column) for column in frame.columns]
            return frame

    raise ValueError(f"Could not parse {path.name}: {last_error}")


def pick_column(
    frame: pd.DataFrame,
    *,
    exact: set[str],
    contains: tuple[str, ...] = (),
    numeric_between: Optional[tuple[float, float]] = None,
    skip: set[str] = frozenset(),
) -> Optional[str]:
    """Resolve a column by exact name, fuzzy name, and then numeric shape."""
    for column in frame.columns:
        if column not in skip and column in exact:
            return column

    for column in frame.columns:
        if column not in skip and contains and any(token in column for token in contains):
            return column

    if numeric_between is not None:
        lower, upper = numeric_between
        best_column: Optional[str] = None
        best_hit_rate = 0.0

        for column in frame.columns:
            if column in skip:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty:
                continue
            hit_rate = float(values.between(lower, upper).mean())
            if hit_rate > best_hit_rate:
                best_column = column
                best_hit_rate = hit_rate

        if best_hit_rate >= 0.75:
            return best_column

    return None


def pick_date_column(frame: pd.DataFrame) -> str:
    likely = {"date", "day", "datetime", "timestamp", "market_date", "trade_date"}

    for column in frame.columns:
        if column in likely or "date" in column:
            if pd.to_datetime(frame[column], errors="coerce").notna().mean() >= 0.6:
                return column

    best_column: Optional[str] = None
    best_hit_rate = 0.0
    for column in frame.columns:
        hit_rate = float(pd.to_datetime(frame[column], errors="coerce").notna().mean())
        if hit_rate > best_hit_rate:
            best_column = column
            best_hit_rate = hit_rate

    if best_column is None or best_hit_rate < 0.6:
        raise ValueError("No column in this file looks like a date.")

    return best_column


# =============================================================================
# Public parsing API
# =============================================================================

def parse_fear_dataset(path: Path) -> pd.DataFrame:
    """Parse Fear & Greed history into an index named date."""
    frame = sniff_read_csv(path)
    date_column = pick_date_column(frame)
    value_column = pick_column(
        frame,
        exact={
            "fear_greed",
            "fear_and_greed",
            "fear_greed_index",
            "feargreed",
            "value",
            "score",
            "index_value",
            "rating_value",
        },
        contains=("fear", "greed"),
        numeric_between=(0, 100),
        skip={date_column},
    )

    if value_column is None:
        raise ValueError(f"Could not identify a Fear & Greed value column in {path}.")

    output = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_column], errors="coerce").dt.normalize(),
            "fear_greed": pd.to_numeric(frame[value_column], errors="coerce"),
        }
    ).dropna()

    output = output[output["fear_greed"].between(0, 100)]
    output = output.sort_values("date").drop_duplicates("date", keep="last")
    return output.set_index("date")


def parse_market_dataset(path: Path) -> pd.DataFrame:
    """Parse an S&P 500 OHLC CSV into open/high/low/close columns."""
    frame = sniff_read_csv(path)
    date_column = pick_date_column(frame)
    close_column = pick_column(
        frame,
        exact={
            "spx_close",
            "sp500_close",
            "close",
            "adj_close",
            "adjusted_close",
            "market_close",
        },
        contains=("close",),
        skip={date_column},
    )

    if close_column is None:
        raise ValueError(f"Could not identify a close-price column in {path}.")

    def numeric(name: str) -> pd.Series:
        source = name if name in frame.columns else close_column
        return pd.to_numeric(frame[source], errors="coerce")

    output = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_column], errors="coerce").dt.normalize(),
            "open": numeric("open"),
            "high": numeric("high"),
            "low": numeric("low"),
            "close": pd.to_numeric(frame[close_column], errors="coerce"),
        }
    ).dropna(subset=["date", "close"])

    output = output.sort_values("date").drop_duplicates("date", keep="last")
    return output.set_index("date")[["open", "high", "low", "close"]]


def parse_combined_dataset(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse the repository's joined Fear & Greed and SPX dataset."""
    frame = sniff_read_csv(path)
    required = {
        "fear_greed_date_utc",
        "fear_greed_value",
        "spx_date",
        "spx_open",
        "spx_high",
        "spx_low",
        "spx_close",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Combined dataset is missing columns: " + ", ".join(sorted(missing))
        )

    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(
                frame["fear_greed_date_utc"], errors="coerce"
            ).dt.normalize(),
            "fear_greed": pd.to_numeric(frame["fear_greed_value"], errors="coerce"),
        }
    ).dropna()
    daily = daily[daily["fear_greed"].between(0, 100)]
    daily = daily.sort_values("date").drop_duplicates("date", keep="last")
    daily = daily.set_index("date")

    market = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["spx_date"], errors="coerce").dt.normalize(),
            "open": pd.to_numeric(frame["spx_open"], errors="coerce"),
            "high": pd.to_numeric(frame["spx_high"], errors="coerce"),
            "low": pd.to_numeric(frame["spx_low"], errors="coerce"),
            "close": pd.to_numeric(frame["spx_close"], errors="coerce"),
        }
    ).dropna(subset=["date", "close"])
    market = market.sort_values("date").drop_duplicates("date", keep="last")
    market = market.set_index("date")[["open", "high", "low", "close"]]

    return daily, market


def download_market(settings: Settings, start: pd.Timestamp) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed and no market CSV is usable.")

    raw = yf.download(
        settings.fallback_ticker,
        start=(start - pd.Timedelta(days=450)).strftime("%Y-%m-%d"),
        end=(pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=False,
    )

    if raw.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {settings.fallback_ticker}.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [
            "_".join(str(part) for part in column if str(part))
            for column in raw.columns
        ]

    def locate(prefix: str) -> Optional[str]:
        for column in raw.columns:
            normalized = slugify(column)
            if normalized == prefix or normalized.startswith(prefix + "_"):
                return str(column)
        return None

    close_source = locate("close")
    if close_source is None:
        raise RuntimeError("Yahoo Finance response has no close column.")

    market = pd.DataFrame(index=pd.to_datetime(raw.index).tz_localize(None).normalize())
    for field_name in ("open", "high", "low", "close"):
        source = locate(field_name)
        market[field_name] = pd.to_numeric(
            raw[source] if source is not None else raw[close_source],
            errors="coerce",
        )

    market = market.dropna(subset=["close"]).sort_index()
    market.index.name = "date"
    return market


def load_data(
    settings: Settings,
    root: Path,
    allow_yahoo: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Resolve Fear & Greed and SPX history using the configured fallback order."""
    fear_path = root / settings.fear_greed_dataset
    market_path = root / settings.market_dataset
    combined_path = root / settings.combined_dataset

    if fear_path.exists() and market_path.exists():
        try:
            daily = parse_fear_dataset(fear_path)
            market = parse_market_dataset(market_path)
            return (
                daily,
                market,
                f"{fear_path.relative_to(root)} + {market_path.relative_to(root)}",
            )
        except Exception as error:  # noqa: BLE001
            print(f"[data] separate datasets unusable ({error}); trying combined file.")

    if combined_path.exists():
        try:
            daily, market = parse_combined_dataset(combined_path)
            return daily, market, str(combined_path.relative_to(root))
        except Exception as error:  # noqa: BLE001
            print(f"[data] combined dataset unusable ({error}); trying Yahoo fallback.")

    if not fear_path.exists():
        raise FileNotFoundError(
            f"No Fear & Greed history found at {fear_path.relative_to(root)}."
        )

    daily = parse_fear_dataset(fear_path)
    if not allow_yahoo:
        raise RuntimeError(
            "No usable market dataset and Yahoo fallback is disabled "
            "(--skip-yahoo-fallback)."
        )

    market = download_market(settings, daily.index.min())
    return daily, market, f"{fear_path.relative_to(root)} + Yahoo Finance"


# =============================================================================
# Feature engineering and signal/market merge
# =============================================================================

def classify_market_regime(row: pd.Series) -> Optional[str]:
    """Classify the market using only information available on that date."""
    distance_from_high = row.get("distance_from_252d_high")
    distance_from_sma_200 = row.get("distance_from_sma_200")

    if pd.isna(distance_from_high) or pd.isna(distance_from_sma_200):
        return None

    if float(distance_from_sma_200) < 0:
        return "downtrend"
    if float(distance_from_high) <= -0.10:
        return "correction"
    if float(distance_from_high) >= -0.02:
        return "near_high_uptrend"
    return "uptrend_off_high"


def add_features(
    daily: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add sentiment changes and backward-looking market-regime features."""
    daily = daily.sort_index().copy()
    market = market.sort_index().copy()

    for window in (1, 3, 5, 10):
        daily[f"fg_change_{window}"] = daily["fear_greed"].diff(window)

    close = market["close"].astype(float)
    daily_return = close.pct_change()

    market["return_1d"] = daily_return
    market["market_return_3d"] = close.pct_change(3)
    market["market_return_5d"] = close.pct_change(5)
    market["market_return_20d"] = close.pct_change(20)

    low_3 = close.rolling(3, min_periods=2).min()
    high_3 = close.rolling(3, min_periods=2).max()
    high_20 = close.rolling(20, min_periods=5).max()
    high_252 = close.rolling(252, min_periods=60).max()
    record_high = close.expanding(min_periods=1).max()
    sma_50 = close.rolling(50, min_periods=20).mean()
    sma_200 = close.rolling(200, min_periods=60).mean()

    market["distance_from_3d_low"] = close / low_3 - 1
    market["distance_from_3d_high"] = close / high_3 - 1
    market["drawdown_from_20d_high"] = close / high_20 - 1
    market["distance_from_20d_high"] = close / high_20 - 1
    market["distance_from_252d_high"] = close / high_252 - 1
    market["distance_from_record_high"] = close / record_high - 1
    market["distance_from_sma_50"] = close / sma_50 - 1
    market["distance_from_sma_200"] = close / sma_200 - 1
    market["volatility_20d"] = (
        daily_return.rolling(20, min_periods=10).std(ddof=1) * math.sqrt(252)
    )

    market["market_regime"] = market.apply(classify_market_regime, axis=1)
    return daily, market


def _plain_value(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def merge_signals(
    daily: pd.DataFrame,
    market: pd.DataFrame,
    settings: Optional[Settings] = None,
) -> pd.DataFrame:
    """
    Pair each sentiment reading with the most recent trading session at or
    before it. Simulate entry on the next trading session's open to avoid
    look-ahead bias, then calculate forward outcomes.
    """
    settings = settings or Settings()
    market = market[~market.index.duplicated(keep="last")].sort_index()

    if market.empty:
        raise RuntimeError("No market data available to merge against.")

    position_of = {timestamp: position for position, timestamp in enumerate(market.index)}
    closes = market["close"].to_numpy(dtype=float)
    lows = (
        market["low"] if "low" in market.columns else market["close"]
    ).to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []

    for signal_date, sentiment_row in daily.iterrows():
        available_dates = market.index[market.index <= signal_date]
        if available_dates.empty:
            continue

        market_date = available_dates[-1]
        signal_position = position_of[market_date]
        entry_position = signal_position + 1

        if entry_position >= len(market):
            continue

        signal_market_row = market.loc[market_date]
        entry_row = market.iloc[entry_position]
        entry_price = (
            float(entry_row["open"])
            if pd.notna(entry_row.get("open"))
            else float(entry_row["close"])
        )
        entry_date = market.index[entry_position]

        record: dict[str, Any] = {
            "signal_date": signal_date,
            "fear_greed": float(sentiment_row["fear_greed"]),
            "fg_change_1": _plain_value(sentiment_row.get("fg_change_1")),
            "fg_change_3": _plain_value(sentiment_row.get("fg_change_3")),
            "fg_change_5": _plain_value(sentiment_row.get("fg_change_5")),
            "fg_change_10": _plain_value(sentiment_row.get("fg_change_10")),
            "market_date": market_date,
            "entry_date": entry_date,
            "entry_price": entry_price,
        }

        for column in MARKET_CONTEXT_COLUMNS:
            record[column] = _plain_value(signal_market_row.get(column))

        for horizon in settings.horizons:
            target_position = entry_position + horizon - 1
            if target_position < len(market) and np.isfinite(entry_price):
                record[f"forward_{horizon}d"] = float(
                    closes[target_position] / entry_price - 1
                )
            else:
                record[f"forward_{horizon}d"] = np.nan

        window_end = min(entry_position + 19, len(market) - 1)
        window_lows = lows[entry_position : window_end + 1]
        window_lows = window_lows[~np.isnan(window_lows)]
        record["max_drawdown_20d"] = (
            float(window_lows.min() / entry_price - 1)
            if window_lows.size
            else np.nan
        )

        rows.append(record)

    merged = pd.DataFrame(rows)
    if merged.empty:
        raise RuntimeError(
            "No Fear & Greed reading had a following trading session to enter on. "
            "Check that market history extends past the sentiment history."
        )

    return merged.sort_values("entry_date").reset_index(drop=True)


def build_current_context(daily: pd.DataFrame, market: pd.DataFrame) -> pd.Series:
    """Build the live sentiment and market context without requiring a future entry."""
    latest_date = daily.index.max()
    sentiment_row = daily.loc[latest_date]
    available_dates = market.index[market.index <= latest_date]

    if available_dates.empty:
        raise RuntimeError("No market observation exists at or before the latest signal.")

    market_date = available_dates[-1]
    market_row = market.loc[market_date]

    recent_sentiment = daily["fear_greed"].tail(5)
    context: dict[str, Any] = {
        "signal_date": latest_date,
        "market_date": market_date,
        "fear_greed": float(sentiment_row["fear_greed"]),
        "fg_change_1": _plain_value(sentiment_row.get("fg_change_1")),
        "fg_change_3": _plain_value(sentiment_row.get("fg_change_3")),
        "fg_change_5": _plain_value(sentiment_row.get("fg_change_5")),
        "fg_change_10": _plain_value(sentiment_row.get("fg_change_10")),
        "market_close": _plain_value(market_row.get("close")),
        "recent_fg_low_5": (
            float(recent_sentiment.min()) if not recent_sentiment.empty else None
        ),
        "recent_fg_high_5": (
            float(recent_sentiment.max()) if not recent_sentiment.empty else None
        ),
    }

    for column in MARKET_CONTEXT_COLUMNS:
        context[column] = _plain_value(market_row.get(column))

    return pd.Series(context)


# =============================================================================
# Fast timing layer
# =============================================================================

@dataclass
class TimingCheck:
    label: str
    value: str
    passed: bool


@dataclass
class TimingSignal:
    action: str
    tone: str
    side: str
    score: int
    confirmation_count: int
    confirmation_total: int
    recommendation: str
    rationale: str
    recent_fg_low_5: Optional[float]
    recent_fg_high_5: Optional[float]
    checks: list[TimingCheck]


def _finite_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _score_at_least(value: Optional[float], threshold: float, points: int) -> int:
    return points if value is not None and value >= threshold else 0


def _score_at_most(value: Optional[float], threshold: float, points: int) -> int:
    return points if value is not None and value <= threshold else 0


def score_fast_timing(
    daily: pd.DataFrame,
    market: pd.DataFrame,
    current: pd.Series,
    settings: Settings,
) -> TimingSignal:
    """
    Produce an earlier tactical timing signal without waiting for a large
    analog sample.

    This layer is deliberately separate from score_analogs(). It only uses
    backward-looking/current-date sentiment and price information. Future
    local lows/highs are never used here; they are used only later by the
    evaluation helper.
    """
    lookback = max(2, int(settings.timing_recent_sentiment_observations))
    recent_sentiment = pd.to_numeric(
        daily.loc[daily.index <= pd.Timestamp(current["signal_date"]), "fear_greed"],
        errors="coerce",
    ).dropna().tail(lookback)

    current_fg = _finite_float(current.get("fear_greed"))
    recent_low = (
        float(recent_sentiment.min()) if not recent_sentiment.empty else current_fg
    )
    recent_high = (
        float(recent_sentiment.max()) if not recent_sentiment.empty else current_fg
    )

    fg_change_1 = _finite_float(current.get("fg_change_1"))
    fg_change_3 = _finite_float(current.get("fg_change_3"))
    return_1d = None
    market_date = pd.Timestamp(current["market_date"])
    if market_date in market.index:
        return_1d = _finite_float(market.loc[market_date].get("return_1d"))
    return_3d = _finite_float(current.get("market_return_3d"))
    return_5d = _finite_float(current.get("market_return_5d"))
    return_20d = _finite_float(current.get("market_return_20d"))
    distance_high = _finite_float(current.get("distance_from_252d_high"))
    distance_sma50 = _finite_float(current.get("distance_from_sma_50"))
    distance_sma200 = _finite_float(current.get("distance_from_sma_200"))
    distance_3d_low = _finite_float(current.get("distance_from_3d_low"))
    distance_3d_high = _finite_float(current.get("distance_from_3d_high"))
    volatility = _finite_float(current.get("volatility_20d"))

    # Stress score: high score means fear + price damage are already unusual.
    buy_score = 0
    buy_score += _score_at_most(recent_low, 30, 10)
    buy_score += _score_at_most(recent_low, 25, 15)
    buy_score += _score_at_most(recent_low, 20, 15)
    buy_score += _score_at_most(recent_low, 15, 10)
    buy_score += _score_at_most(distance_high, -0.05, 10)
    buy_score += _score_at_most(distance_high, -0.08, 10)
    buy_score += _score_at_most(distance_high, -0.12, 15)
    buy_score += _score_at_most(return_5d, -0.03, 10)
    buy_score += _score_at_most(return_5d, -0.05, 10)
    buy_score += _score_at_most(return_20d, -0.08, 10)
    buy_score += _score_at_least(volatility, 0.25, 5)

    buy_checks = [
        TimingCheck(
            "Fear & Greed rebounding from recent low",
            (
                "N/A"
                if current_fg is None or recent_low is None
                else f"{current_fg:.1f} vs {recent_low:.1f} low"
            ),
            current_fg is not None
            and recent_low is not None
            and current_fg >= recent_low + 2.0,
        ),
        TimingCheck(
            "1-observation sentiment change positive",
            "N/A" if fg_change_1 is None else f"{fg_change_1:+.1f}",
            fg_change_1 is not None and fg_change_1 > 0,
        ),
        TimingCheck(
            "3-observation sentiment change positive",
            "N/A" if fg_change_3 is None else f"{fg_change_3:+.1f}",
            fg_change_3 is not None and fg_change_3 > 0,
        ),
        TimingCheck(
            "Market positive today",
            "N/A" if return_1d is None else f"{return_1d * 100:+.2f}%",
            return_1d is not None and return_1d > 0,
        ),
        TimingCheck(
            "3-day market momentum stabilized",
            "N/A" if return_3d is None else f"{return_3d * 100:+.2f}%",
            return_3d is not None and return_3d > -0.005,
        ),
        TimingCheck(
            "Price bounced from 3-day low",
            "N/A" if distance_3d_low is None else f"{distance_3d_low * 100:.2f}%",
            distance_3d_low is not None and distance_3d_low >= 0.008,
        ),
    ]
    buy_confirmations = sum(check.passed for check in buy_checks)

    # Heat score: high score means greed/extension are elevated.
    trim_score = 0
    trim_score += _score_at_least(recent_high, 70, 10)
    trim_score += _score_at_least(recent_high, 75, 15)
    trim_score += _score_at_least(recent_high, 80, 15)
    trim_score += _score_at_least(recent_high, 85, 10)
    trim_score += _score_at_least(distance_high, -0.015, 10)
    trim_score += _score_at_least(return_20d, 0.04, 10)
    trim_score += _score_at_least(return_20d, 0.07, 10)
    trim_score += _score_at_least(distance_sma50, 0.03, 10)
    trim_score += _score_at_least(distance_sma200, 0.10, 10)
    trim_score += _score_at_least(distance_sma200, 0.15, 10)

    trim_checks = [
        TimingCheck(
            "Fear & Greed rolling over from recent high",
            (
                "N/A"
                if current_fg is None or recent_high is None
                else f"{current_fg:.1f} vs {recent_high:.1f} high"
            ),
            current_fg is not None
            and recent_high is not None
            and current_fg <= recent_high - 3.0,
        ),
        TimingCheck(
            "1-observation sentiment change negative",
            "N/A" if fg_change_1 is None else f"{fg_change_1:+.1f}",
            fg_change_1 is not None and fg_change_1 < 0,
        ),
        TimingCheck(
            "3-observation sentiment change negative",
            "N/A" if fg_change_3 is None else f"{fg_change_3:+.1f}",
            fg_change_3 is not None and fg_change_3 < 0,
        ),
        TimingCheck(
            "Market negative today",
            "N/A" if return_1d is None else f"{return_1d * 100:+.2f}%",
            return_1d is not None and return_1d < 0,
        ),
        TimingCheck(
            "3-day market momentum negative",
            "N/A" if return_3d is None else f"{return_3d * 100:+.2f}%",
            return_3d is not None and return_3d < 0,
        ),
        TimingCheck(
            "Price slipped from 3-day high",
            "N/A" if distance_3d_high is None else f"{distance_3d_high * 100:.2f}%",
            distance_3d_high is not None and distance_3d_high <= -0.008,
        ),
    ]
    trim_confirmations = sum(check.passed for check in trim_checks)

    # Prefer the side with the stronger setup when both scores are non-trivial.
    buy_active = buy_score >= settings.timing_buy_watch_score
    trim_active = trim_score >= settings.timing_trim_watch_score
    if buy_active and trim_active:
        if buy_score - settings.timing_buy_watch_score >= trim_score - settings.timing_trim_watch_score:
            trim_active = False
        else:
            buy_active = False

    if buy_active:
        if (
            buy_score >= settings.timing_buy_first_tranche_score
            and buy_confirmations >= settings.timing_buy_first_tranche_confirmations
        ):
            action = "EARLY BUY — FIRST TRANCHE"
            tone = "positive"
            recommendation = "Start ~25–35% of the tactical amount you planned to deploy."
            rationale = (
                "Fear/price stress is already extreme and at least two stabilization "
                "checks have turned. This is intentionally earlier than BUY GRADUALLY."
            )
        elif (
            buy_score >= settings.timing_buy_zone_score
            and buy_confirmations >= settings.timing_buy_small_start_confirmations
        ):
            action = "EARLY BUY — SMALL START"
            tone = "positive"
            recommendation = "Start ~10–20% of the tactical amount; keep most cash reserved."
            rationale = (
                "The market is meaningfully oversold and several stabilization checks "
                "have turned, but confirmation is still incomplete."
            )
        elif buy_score >= settings.timing_buy_zone_score:
            action = "EXTREME BUY ZONE — WAIT FOR STABILIZATION"
            tone = "mixed"
            recommendation = "Do not deploy the full amount yet; wait for 2–3 stabilization checks."
            rationale = (
                "Fear and/or drawdown are extreme, but selling pressure has not shown "
                "enough evidence of slowing."
            )
        else:
            action = "BUY WATCH — OVERSOLD"
            tone = "mixed"
            recommendation = "Prepare cash and a staged plan; no early tranche yet."
            rationale = (
                "Stress is elevated enough to watch closely, but the setup is not yet "
                "extreme or stable enough for an early tactical buy."
            )
        return TimingSignal(
            action=action,
            tone=tone,
            side="BUY",
            score=int(buy_score),
            confirmation_count=int(buy_confirmations),
            confirmation_total=len(buy_checks),
            recommendation=recommendation,
            rationale=rationale,
            recent_fg_low_5=recent_low,
            recent_fg_high_5=recent_high,
            checks=buy_checks,
        )

    if trim_active:
        if (
            trim_score >= settings.timing_trim_strong_score
            and trim_confirmations >= settings.timing_trim_confirmations + 1
        ):
            action = "EARLY TRIM — REDUCE TACTICAL RISK"
            tone = "negative"
            recommendation = "Consider trimming ~15–25% of tactical/overweight exposure."
            rationale = (
                "The market is highly extended and multiple rollover checks have "
                "turned negative. This is an early risk-reduction signal, not a call "
                "to liquidate a long-term core position."
            )
        elif (
            trim_score >= settings.timing_trim_zone_score
            and trim_confirmations >= settings.timing_trim_confirmations
        ):
            action = "EARLY TRIM / CAUTION"
            tone = "negative"
            recommendation = "Stop extra buying; optionally trim ~5–15% of tactical exposure."
            rationale = (
                "Greed/extension are elevated and rollover evidence has started to "
                "appear before the slower analog model turns negative."
            )
        elif trim_score >= settings.timing_trim_zone_score:
            action = "OVERHEATED — NO EXTRA BUYING"
            tone = "mixed"
            recommendation = "Pause discretionary adds; wait for rollover confirmation before trimming."
            rationale = (
                "The market is stretched, but price/sentiment have not rolled over "
                "enough to justify an early trim."
            )
        else:
            action = "OVERHEATED WATCH"
            tone = "mixed"
            recommendation = "Keep normal holdings; avoid chasing strength."
            rationale = (
                "Extension is elevated enough to monitor, but the setup is not yet "
                "strong enough for a trim signal."
            )
        return TimingSignal(
            action=action,
            tone=tone,
            side="TRIM",
            score=int(trim_score),
            confirmation_count=int(trim_confirmations),
            confirmation_total=len(trim_checks),
            recommendation=recommendation,
            rationale=rationale,
            recent_fg_low_5=recent_low,
            recent_fg_high_5=recent_high,
            checks=trim_checks,
        )

    neutral_checks = [
        TimingCheck("No extreme buy stress", f"buy score {buy_score}", True),
        TimingCheck("No extreme trim heat", f"trim score {trim_score}", True),
    ]
    return TimingSignal(
        action="NEUTRAL / NO TIMING EDGE",
        tone="neutral",
        side="NEUTRAL",
        score=int(max(buy_score, trim_score)),
        confirmation_count=0,
        confirmation_total=0,
        recommendation="Follow the baseline plan; no fast tactical action is indicated.",
        rationale=(
            "Neither the oversold-stress score nor the overheated-extension score "
            "is high enough to justify an early tactical action."
        ),
        recent_fg_low_5=recent_low,
        recent_fg_high_5=recent_high,
        checks=neutral_checks,
    )


# =============================================================================
# Analogs, regime baseline, and headline action
# =============================================================================

def apply_cooldown(
    events: pd.DataFrame,
    mask: pd.Series,
    cooldown_days: int,
) -> pd.DataFrame:
    """Keep events far enough apart that one episode is not counted repeatedly."""
    candidates = events.loc[mask.fillna(False)].sort_values("signal_date")
    if candidates.empty or cooldown_days <= 0:
        return candidates

    keep: list[int] = []
    last_kept: Optional[pd.Timestamp] = None

    for index, row in candidates.iterrows():
        current = pd.Timestamp(row["signal_date"])
        if last_kept is None or (current - last_kept).days >= cooldown_days:
            keep.append(index)
            last_kept = current

    return events.loc[keep]


def _safe_mean(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _safe_median(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if len(values) else None


def _win_rate(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float((values > 0).mean()) if len(values) else None


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return math.nan

    probability = successes / total
    denominator = 1 + z**2 / total
    center = (probability + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            (probability * (1 - probability) + z**2 / (4 * total)) / total
        )
        / denominator
    )
    return max(0.0, center - margin)


def bootstrap_excess_interval(
    analog_returns: pd.Series,
    baseline_returns: pd.Series,
    *,
    iterations: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[Optional[float], Optional[float]]:
    """Bootstrap the difference between analog and same-regime mean returns."""
    analog = pd.to_numeric(analog_returns, errors="coerce").dropna().to_numpy(float)
    baseline = pd.to_numeric(baseline_returns, errors="coerce").dropna().to_numpy(float)

    if len(analog) < 2 or len(baseline) < 2 or iterations <= 0:
        return None, None

    # Draw the full bootstrap matrices in NumPy rather than looping in Python.
    # This is the same bootstrap estimator (same seed, iterations and resampling
    # with replacement), but is fast enough to replay the complete history on
    # every dashboard build.
    generator = np.random.default_rng(seed)
    analog_samples = generator.choice(
        analog, size=(iterations, len(analog)), replace=True
    )
    baseline_samples = generator.choice(
        baseline, size=(iterations, len(baseline)), replace=True
    )
    differences = analog_samples.mean(axis=1) - baseline_samples.mean(axis=1)

    alpha = 1 - confidence
    lower = float(np.quantile(differences, alpha / 2))
    upper = float(np.quantile(differences, 1 - alpha / 2))
    return lower, upper


def determine_market_extension(current: pd.Series) -> tuple[str, int]:
    """Describe how extended the current uptrend is without treating an ATH as bearish."""
    points = 0

    distance_from_high = current.get("distance_from_252d_high")
    return_20d = current.get("market_return_20d")
    distance_from_sma_50 = current.get("distance_from_sma_50")
    distance_from_sma_200 = current.get("distance_from_sma_200")

    if pd.notna(distance_from_high) and float(distance_from_high) >= -0.01:
        points += 1
    if pd.notna(return_20d) and float(return_20d) >= 0.05:
        points += 1
    if pd.notna(distance_from_sma_50) and float(distance_from_sma_50) >= 0.03:
        points += 1
    if pd.notna(distance_from_sma_200) and float(distance_from_sma_200) >= 0.12:
        points += 1

    if points >= 2:
        return "High", points
    if points == 1:
        return "Moderate", points
    return "Normal", points


def _normalized_distance(
    frame: pd.DataFrame,
    current: pd.Series,
    settings: Settings,
) -> pd.Series:
    """Weighted distance within one market regime."""
    distance = pd.Series(0.0, index=frame.index, dtype=float)

    components = [
        ("fear_greed", settings.analog_level_band, 1.0),
        ("fg_change_5", settings.analog_change_band, 1.0),
        ("distance_from_252d_high", settings.analog_high_distance_band, 0.8),
        ("market_return_20d", settings.analog_return_20d_band, 0.7),
        ("distance_from_sma_200", settings.analog_sma_200_band, 0.6),
        ("volatility_20d", settings.analog_volatility_band, 0.4),
    ]

    for column, scale, weight in components:
        current_value = current.get(column)
        if column not in frame or pd.isna(current_value) or scale <= 0:
            continue
        difference = (
            pd.to_numeric(frame[column], errors="coerce") - float(current_value)
        ).abs()
        distance += weight * (difference / scale).fillna(1.5)

    return distance


def find_analogs(
    events: pd.DataFrame,
    current: pd.Series,
    settings: Settings,
) -> tuple[pd.DataFrame, str]:
    """Find independent historical observations in the same market regime."""
    complete = events[
        events["forward_5d"].notna() & events["market_regime"].notna()
    ].copy()

    current_regime = current.get("market_regime")
    if current_regime is None or pd.isna(current_regime):
        return complete.iloc[0:0], "market regime unavailable"

    same_regime = complete[complete["market_regime"] == current_regime].copy()
    if same_regime.empty:
        return same_regime, f"same regime only: {current_regime}"

    strict_mask = pd.Series(True, index=same_regime.index)
    strict_specs = [
        ("fear_greed", settings.analog_level_band),
        ("fg_change_5", settings.analog_change_band),
        ("distance_from_252d_high", settings.analog_high_distance_band),
        ("market_return_20d", settings.analog_return_20d_band),
        ("distance_from_sma_200", settings.analog_sma_200_band),
    ]

    for column, band in strict_specs:
        current_value = current.get(column)
        if column not in same_regime or pd.isna(current_value):
            continue
        strict_mask &= pd.to_numeric(same_regime[column], errors="coerce").between(
            float(current_value) - band,
            float(current_value) + band,
        )

    strict = apply_cooldown(
        same_regime,
        strict_mask,
        settings.cooldown_calendar_days,
    )
    if len(strict) >= settings.minimum_regime_sample:
        return strict.head(settings.maximum_analogs), (
            f"same {current_regime} regime; sentiment, high-distance, "
            "20D-return, and SMA200 bands"
        )

    relaxed_mask = pd.Series(True, index=same_regime.index)
    relaxed_specs = [
        ("fear_greed", settings.analog_level_band),
        ("fg_change_5", settings.analog_change_band),
        ("distance_from_252d_high", settings.analog_high_distance_band * 1.5),
        ("market_return_20d", settings.analog_return_20d_band * 1.5),
    ]

    for column, band in relaxed_specs:
        current_value = current.get(column)
        if column not in same_regime or pd.isna(current_value):
            continue
        relaxed_mask &= pd.to_numeric(same_regime[column], errors="coerce").between(
            float(current_value) - band,
            float(current_value) + band,
        )

    relaxed = apply_cooldown(
        same_regime,
        relaxed_mask,
        settings.cooldown_calendar_days,
    )
    if len(relaxed) >= settings.minimum_regime_sample:
        return relaxed.head(settings.maximum_analogs), (
            f"same {current_regime} regime; relaxed sentiment and price-context bands"
        )

    distance = _normalized_distance(same_regime, current, settings)
    ranked = same_regime.assign(_distance=distance).sort_values("_distance")
    ranked = ranked[ranked["_distance"] <= settings.max_analog_distance]

    chosen_indices: list[int] = []
    chosen_dates: list[pd.Timestamp] = []

    for index, row in ranked.iterrows():
        signal_date = pd.Timestamp(row["signal_date"])
        if all(
            abs((signal_date - prior_date).days) >= settings.cooldown_calendar_days
            for prior_date in chosen_dates
        ):
            chosen_indices.append(index)
            chosen_dates.append(signal_date)

        if len(chosen_indices) >= settings.maximum_analogs:
            break

    chosen = same_regime.loc[chosen_indices].copy()
    if not chosen.empty:
        chosen["analog_distance"] = distance.loc[chosen.index]

    return chosen, (
        f"nearest independent observations within the same {current_regime} regime"
    )


def find_regime_baseline(
    events: pd.DataFrame,
    current: pd.Series,
    settings: Settings,
) -> pd.DataFrame:
    """Return independent completed observations from the current market regime."""
    current_regime = current.get("market_regime")
    if current_regime is None or pd.isna(current_regime):
        return events.iloc[0:0]

    mask = (
        events["forward_5d"].notna()
        & events["market_regime"].eq(current_regime)
    )
    return apply_cooldown(events, mask, settings.cooldown_calendar_days)


@dataclass
class DecisionCheck:
    label: str
    value: str
    requirement: str
    passed: bool


@dataclass
class Verdict:
    action: str
    tone: str
    confidence: str
    sample_size: int
    regime_baseline_sample: int
    required_sample: int
    market_regime: str
    market_extension: str
    extension_points: int
    win_rate_5d: Optional[float]
    wilson_floor_5d: Optional[float]
    average_5d: Optional[float]
    median_5d: Optional[float]
    average_20d: Optional[float]
    regime_baseline_5d: Optional[float]
    excess_5d: Optional[float]
    excess_ci_low_5d: Optional[float]
    excess_ci_high_5d: Optional[float]
    required_excess_5d: float
    worst_5d: Optional[float]
    average_drawdown_20d: Optional[float]
    analog_method: str
    rationale: str
    positive_checks_passed: int
    positive_checks_total: int
    decision_checks: list[DecisionCheck]


def score_analogs(
    analogs: pd.DataFrame,
    regime_baseline: pd.DataFrame,
    current: pd.Series,
    method: str,
    settings: Settings,
) -> Verdict:
    five = pd.to_numeric(analogs.get("forward_5d"), errors="coerce").dropna()
    twenty = pd.to_numeric(analogs.get("forward_20d"), errors="coerce").dropna()
    baseline_five = pd.to_numeric(
        regime_baseline.get("forward_5d"), errors="coerce"
    ).dropna()

    sample = len(five)
    baseline_sample = len(baseline_five)
    win_rate = _win_rate(five)
    average_5d = _safe_mean(five)
    median_5d = _safe_median(five)
    average_20d = _safe_mean(twenty)
    baseline_5d = _safe_mean(baseline_five)
    excess = (
        None
        if average_5d is None or baseline_5d is None
        else average_5d - baseline_5d
    )
    wilson_floor = (
        wilson_lower_bound(int((five > 0).sum()), sample)
        if sample
        else math.nan
    )
    wilson_floor_value = None if not np.isfinite(wilson_floor) else float(wilson_floor)

    ci_low, ci_high = bootstrap_excess_interval(
        five,
        baseline_five,
        iterations=settings.bootstrap_iterations,
        seed=settings.bootstrap_seed,
    )

    average_drawdown = _safe_mean(
        pd.to_numeric(analogs.get("max_drawdown_20d"), errors="coerce")
    )
    worst_5d = None if five.empty else float(five.min())

    extension, extension_points = determine_market_extension(current)
    if extension == "High":
        required_sample = max(settings.minimum_regime_sample, 25)
        required_excess = settings.high_extension_minimum_excess_5d
        maximum_drawdown = settings.high_extension_maximum_average_drawdown_20d
    elif extension == "Moderate":
        required_sample = settings.minimum_regime_sample
        required_excess = settings.moderate_extension_minimum_excess_5d
        maximum_drawdown = settings.normal_maximum_average_drawdown_20d
    else:
        required_sample = settings.minimum_regime_sample
        required_excess = settings.normal_minimum_excess_5d
        maximum_drawdown = settings.normal_maximum_average_drawdown_20d

    sample_ok = sample >= required_sample
    baseline_ok = baseline_sample >= settings.minimum_regime_baseline_sample
    excess_ok = excess is not None and excess >= required_excess
    confidence_ok = ci_low is not None and ci_low > 0

    support_checks = {
        "Win rate": win_rate is not None and win_rate >= 0.60,
        "Median 5D": median_5d is not None and median_5d > 0,
        "Average 20D": average_20d is not None and average_20d > 0,
        "Average drawdown": (
            average_drawdown is not None and average_drawdown >= maximum_drawdown
        ),
    }
    support_passes = sum(support_checks.values())

    positive_checks = {
        "Analog sample": sample_ok,
        "Regime baseline": baseline_ok,
        "Average excess": excess_ok,
        "Excess confidence": confidence_ok,
        **support_checks,
    }
    positive_passes = sum(positive_checks.values())

    negative_support = [
        win_rate is not None and win_rate <= 0.45,
        average_5d is not None and average_5d < 0,
        median_5d is not None and median_5d < 0,
        average_20d is not None and average_20d < 0,
    ]
    negative_excess_ok = (
        excess is not None and excess <= settings.negative_excess_5d
    )
    negative_confidence_ok = ci_high is not None and ci_high < 0

    if not sample_ok or not baseline_ok:
        action = "INSUFFICIENT EVIDENCE"
        tone = "neutral"
        confidence = "Very low"
        rationale = (
            f"Only {sample} independent same-regime analogs and "
            f"{baseline_sample} same-regime baseline observations are available. "
            "The model will not replace missing evidence with observations from a "
            "different market regime."
        )
    elif excess_ok and confidence_ok and support_passes >= 3:
        action = "BUY GRADUALLY"
        tone = "positive"
        confidence = "Moderate" if sample >= 30 else "Low"
        rationale = (
            "Same-regime analogs produced a meaningful five-day excess return, "
            "the bootstrap confidence interval remained above zero, and most "
            "supporting return and drawdown tests passed. This supports only a "
            "staged tactical addition, not an all-in purchase or a bottom call."
        )
    elif (
        negative_excess_ok
        and negative_confidence_ok
        and sum(negative_support) >= 3
    ):
        action = "WAIT ON BUYING"
        tone = "negative"
        confidence = "Moderate" if sample >= 30 else "Low"
        rationale = (
            "Same-regime analogs showed statistically negative excess returns and "
            "most short-term outcome tests were unfavorable. This means wait on "
            "buying for now; it is not an automatic sell signal."
        )
    else:
        action = "HOLD / NO EXTRA BUYING"
        tone = "mixed"
        confidence = "Moderate" if sample >= 30 else "Low"
        rationale = (
            "The same-regime evidence does not establish a positive tactical edge. "
            "Avoid an above-normal purchase. This signal does not decide whether "
            "to continue a separate long-term contribution plan and is not an "
            "automatic sell signal."
        )

    checks = [
        DecisionCheck(
            "Independent analogs",
            str(sample),
            f"at least {required_sample}",
            sample_ok,
        ),
        DecisionCheck(
            "Same-regime baseline",
            str(baseline_sample),
            f"at least {settings.minimum_regime_baseline_sample}",
            baseline_ok,
        ),
        DecisionCheck(
            "Average 5D excess",
            fmt_pct(excess, 2),
            f"at least {fmt_pct(required_excess, 2)}",
            excess_ok,
        ),
        DecisionCheck(
            "95% excess CI lower bound",
            fmt_pct(ci_low, 2),
            "above 0.00%",
            confidence_ok,
        ),
        DecisionCheck(
            "5D win rate",
            fmt_pct(win_rate, 1),
            "at least 60.0%",
            support_checks["Win rate"],
        ),
        DecisionCheck(
            "Median 5D return",
            fmt_pct(median_5d, 2),
            "above 0.00%",
            support_checks["Median 5D"],
        ),
        DecisionCheck(
            "Average 20D return",
            fmt_pct(average_20d, 2),
            "above 0.00%",
            support_checks["Average 20D"],
        ),
        DecisionCheck(
            "Average 20D drawdown",
            fmt_pct(average_drawdown, 2),
            f"not worse than {fmt_pct(maximum_drawdown, 2)}",
            support_checks["Average drawdown"],
        ),
    ]

    return Verdict(
        action=action,
        tone=tone,
        confidence=confidence,
        sample_size=sample,
        regime_baseline_sample=baseline_sample,
        required_sample=required_sample,
        market_regime=str(current.get("market_regime") or "unavailable"),
        market_extension=extension,
        extension_points=extension_points,
        win_rate_5d=win_rate,
        wilson_floor_5d=wilson_floor_value,
        average_5d=average_5d,
        median_5d=median_5d,
        average_20d=average_20d,
        regime_baseline_5d=baseline_5d,
        excess_5d=excess,
        excess_ci_low_5d=ci_low,
        excess_ci_high_5d=ci_high,
        required_excess_5d=required_excess,
        worst_5d=worst_5d,
        average_drawdown_20d=average_drawdown,
        analog_method=method,
        rationale=rationale,
        positive_checks_passed=positive_passes,
        positive_checks_total=len(positive_checks),
        decision_checks=checks,
    )


# =============================================================================
# Position-sizing guidance
# =============================================================================

@dataclass
class PositionGuidance:
    tier: str
    sizing_label: str
    sizing_detail: str
    guardrail: str


def build_position_guidance(verdict: Verdict, settings: Settings) -> PositionGuidance:
    """Translate the verdict into the dashboard's rule-based sizing suggestion."""
    checks_ratio = (
        verdict.positive_checks_passed / verdict.positive_checks_total
        if verdict.positive_checks_total
        else 0.0
    )
    worst_note = (
        f"Worst same-regime analog was {fmt_pct(verdict.worst_5d)} over the "
        "next 5 sessions — size any addition so that outcome would still be "
        "tolerable."
        if verdict.worst_5d is not None
        else "No worst-case analog is available yet to size against."
    )

    if verdict.action == "BUY GRADUALLY":
        if checks_ratio >= settings.sizing_strong_buy_min_checks_ratio:
            tier = "Elevated tactical buy"
            sizing_label = f"~{settings.sizing_strong_buy_pct}% of your normal buy"
            sizing_detail = (
                "Same-regime analogs cleared nearly every test "
                f"({verdict.positive_checks_passed}/{verdict.positive_checks_total}). "
                "Consider stepping up your usual contribution, still spread across "
                "more than one purchase rather than deployed all at once."
            )
        else:
            tier = "Modest tactical buy"
            sizing_label = (
                f"~{settings.sizing_modest_buy_low_pct}-"
                f"{settings.sizing_modest_buy_high_pct}% of your normal buy"
            )
            sizing_detail = (
                "The edge cleared the bar but not by a wide margin "
                f"({verdict.positive_checks_passed}/{verdict.positive_checks_total} "
                "checks passed). A small step-up over your normal size, staged "
                "over 2-3 purchases, keeps risk contained if the edge doesn't hold."
            )
        guardrail = worst_note
    elif verdict.action == "WAIT ON BUYING":
        tier = "Pause discretionary buying"
        sizing_label = "Skip this week's extra buy"
        sizing_detail = (
            "Same-regime analogs skewed negative with reasonable statistical "
            "confidence. Consider skipping or delaying any discretionary buy "
            "while keeping scheduled/automatic contributions running as normal."
        )
        guardrail = "This is a pause on new buying, not a signal to sell existing positions."
    elif verdict.action == "HOLD / NO EXTRA BUYING":
        tier = "Stay at your baseline"
        sizing_label = "Normal plan only, no discretionary add"
        sizing_detail = (
            "The evidence doesn't clear the bar for an above-normal purchase, "
            "but it isn't negative enough to justify pulling back either. Keep "
            "scheduled contributions as-is and skip extra discretionary buying "
            "until conditions clarify."
        )
        guardrail = "Not a sell signal — this only withholds an above-normal buy."
    else:
        tier = "No reliable tactical read"
        sizing_label = "Fall back to your baseline plan"
        sizing_detail = (
            "Too few independent same-regime analogs exist right now to trust "
            "a tactical signal either way. Treat this like a normal week and "
            "keep whatever plan you'd otherwise follow."
        )
        guardrail = "Insufficient evidence — not a bullish or bearish signal."

    return PositionGuidance(
        tier=tier,
        sizing_label=sizing_label,
        sizing_detail=sizing_detail,
        guardrail=guardrail,
    )


def build_event_study(events: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    baseline = _safe_mean(events["forward_5d"])
    rows: list[dict[str, Any]] = []

    def summarize(label: str, subset: pd.DataFrame) -> None:
        five = subset["forward_5d"].dropna()
        average = _safe_mean(five)
        rows.append(
            {
                "Signal": label,
                "Events": len(five),
                "Win rate 5D": _win_rate(five),
                "Average 5D": average,
                "Median 5D": _safe_median(five),
                "Average 20D": _safe_mean(subset["forward_20d"].dropna()),
                "Worst 5D": None if five.empty else float(five.min()),
                "Avg worst drawdown 20D": _safe_mean(subset["max_drawdown_20d"]),
                "Excess vs baseline 5D": (
                    None
                    if average is None or baseline is None
                    else average - baseline
                ),
            }
        )

    for threshold in settings.level_thresholds:
        summarize(
            f"Fear & Greed ≤ {threshold}",
            apply_cooldown(
                events,
                events["fear_greed"] <= threshold,
                settings.cooldown_calendar_days,
            ),
        )

    for window in settings.drop_windows:
        column = f"fg_change_{window}"
        if column not in events.columns:
            continue
        for threshold in settings.drop_thresholds:
            summarize(
                f"{window}-obs drop ≥ {threshold}",
                apply_cooldown(
                    events,
                    events[column] <= -threshold,
                    settings.cooldown_calendar_days,
                ),
            )

    return pd.DataFrame(rows)


def build_signal_scorecard(study: pd.DataFrame, current: pd.Series) -> pd.DataFrame:
    if study.empty:
        return study.iloc[0:0]

    fg = current.get("fear_greed")
    matches: list[pd.DataFrame] = []

    level_rows = study[study["Signal"].str.startswith("Fear & Greed")].copy()
    if pd.notna(fg) and not level_rows.empty:
        level_rows["_threshold"] = (
            level_rows["Signal"].str.extract(r"≤\s*(\d+)").astype(float)[0]
        )
        active_levels = level_rows[level_rows["_threshold"] >= float(fg)]
        if not active_levels.empty:
            tightest = active_levels.sort_values("_threshold").iloc[[0]]
            matches.append(tightest.drop(columns="_threshold"))

    drop_rows = study[study["Signal"].str.contains("obs drop", regex=False)].copy()
    if not drop_rows.empty:

        def drop_is_active(row: pd.Series) -> bool:
            try:
                window = int(row["Signal"].split("-obs")[0])
                threshold = int(row["Signal"].split("≥")[-1].strip())
            except (ValueError, IndexError):
                return False
            change_value = current.get(f"fg_change_{window}")
            return pd.notna(change_value) and float(change_value) <= -threshold

        active_drops = drop_rows[drop_rows.apply(drop_is_active, axis=1)]
        if not active_drops.empty:
            matches.append(active_drops)

    if not matches:
        return study.iloc[0:0]

    combined = pd.concat(matches, ignore_index=True)
    combined = combined.drop_duplicates(subset="Signal")
    return combined.sort_values(
        "Excess vs baseline 5D", ascending=False, na_position="last"
    ).head(5)


def format_scorecard(rows: pd.DataFrame, edge_threshold: float = 0.003) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        excess = row.get("Excess vs baseline 5D")
        tone = "neutral"
        if pd.notna(excess):
            if excess > edge_threshold:
                tone = "positive"
            elif excess < -edge_threshold:
                tone = "negative"

        items.append(
            {
                "label": row["Signal"],
                "events": int(row["Events"]) if pd.notna(row["Events"]) else 0,
                "win_rate": fmt_pct(row.get("Win rate 5D")),
                "average": fmt_pct(row.get("Average 5D"), 2),
                "excess": fmt_pct(excess, 2),
                "tone": tone,
            }
        )
    return items


# =============================================================================
# Point-in-time historical replay
# =============================================================================

def _as_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _plain_history_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if pd.isna(value):
        return None
    return value


def attach_outcome_known_dates(
    events: pd.DataFrame,
    market: pd.DataFrame,
    horizons: Iterable[int] = (5, 20),
) -> pd.DataFrame:
    """Attach the market close date when each forward return becomes knowable."""
    output = events.copy()
    market_index = pd.DatetimeIndex(market.index)
    if market_index.tz is not None:
        market_index = market_index.tz_localize(None)
    market_index = market_index.normalize()
    position_of = {timestamp: position for position, timestamp in enumerate(market_index)}

    entry_positions: list[float] = []
    for value in output["entry_date"]:
        position = position_of.get(_as_timestamp(value))
        entry_positions.append(float(position) if position is not None else np.nan)
    output["_entry_position"] = entry_positions

    for horizon in horizons:
        known_dates: list[pd.Timestamp | pd.NaT] = []
        for raw_position in output["_entry_position"]:
            if pd.isna(raw_position):
                known_dates.append(pd.NaT)
                continue
            target_position = int(raw_position) + int(horizon) - 1
            if 0 <= target_position < len(market_index):
                known_dates.append(market_index[target_position])
            else:
                known_dates.append(pd.NaT)
        output[f"_forward_{horizon}d_known_date"] = known_dates

    return output


def eligible_events_asof(
    events: pd.DataFrame,
    cutoff_market_date: pd.Timestamp,
) -> pd.DataFrame:
    """Expose only events whose 5-day outcome was known by the replay cutoff."""
    cutoff = _as_timestamp(cutoff_market_date)
    known = pd.to_datetime(events["_forward_5d_known_date"], errors="coerce")
    return events.loc[known.notna() & known.le(cutoff)].copy()


def mask_analog_outcomes_asof(
    analogs: pd.DataFrame,
    market: pd.DataFrame,
    cutoff_market_date: pd.Timestamp,
) -> pd.DataFrame:
    """Hide future 20D return data and recompute drawdown only through cutoff."""
    if analogs.empty:
        return analogs.copy()

    output = analogs.copy()
    cutoff = _as_timestamp(cutoff_market_date)
    market_index = pd.DatetimeIndex(market.index)
    if market_index.tz is not None:
        market_index = market_index.tz_localize(None)
    market_index = market_index.normalize()
    cutoff_position = int(market_index.searchsorted(cutoff, side="right") - 1)

    if "forward_20d" in output.columns and "_forward_20d_known_date" in output.columns:
        known_20 = pd.to_datetime(output["_forward_20d_known_date"], errors="coerce")
        output.loc[known_20.isna() | known_20.gt(cutoff), "forward_20d"] = np.nan

    lows_source = market["low"] if "low" in market.columns else market["close"]
    lows = pd.to_numeric(lows_source, errors="coerce").to_numpy(dtype=float)

    drawdowns: list[float] = []
    for _, row in output.iterrows():
        raw_entry_position = row.get("_entry_position")
        entry_price = pd.to_numeric(
            pd.Series([row.get("entry_price")]), errors="coerce"
        ).iloc[0]

        if pd.isna(raw_entry_position) or pd.isna(entry_price) or float(entry_price) <= 0:
            drawdowns.append(np.nan)
            continue

        entry_position = int(raw_entry_position)
        end_position = min(entry_position + 19, cutoff_position, len(lows) - 1)
        if end_position < entry_position:
            drawdowns.append(np.nan)
            continue

        window = lows[entry_position : end_position + 1]
        window = window[np.isfinite(window)]
        if window.size == 0:
            drawdowns.append(np.nan)
            continue

        drawdowns.append(float(window.min() / float(entry_price) - 1.0))

    output["max_drawdown_20d"] = drawdowns
    return output


def replay_historical_decisions(
    settings: Settings,
    daily: pd.DataFrame,
    market: pd.DataFrame,
    events: pd.DataFrame,
    *,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Replay this exact dashboard engine one historical observation at a time."""
    prepared_events = attach_outcome_known_dates(events, market, horizons=(5, 20))
    replay_dates = pd.DatetimeIndex(daily.index)
    if replay_dates.tz is not None:
        replay_dates = replay_dates.tz_localize(None)
    replay_dates = replay_dates.normalize()

    rows: list[dict[str, Any]] = []
    total = len(replay_dates)

    for ordinal, decision_date in enumerate(replay_dates, start=1):
        daily_asof = daily.loc[daily.index <= decision_date]
        market_asof = market.loc[market.index <= decision_date]
        if daily_asof.empty or market_asof.empty:
            continue

        current = build_current_context(daily_asof, market_asof)
        cutoff_market_date = _as_timestamp(current["market_date"])

        eligible = eligible_events_asof(prepared_events, cutoff_market_date)
        historical_analogs, method = find_analogs(eligible, current, settings)
        historical_baseline = find_regime_baseline(eligible, current, settings)
        historical_analogs = mask_analog_outcomes_asof(
            historical_analogs,
            market,
            cutoff_market_date,
        )

        historical_verdict = score_analogs(
            historical_analogs,
            historical_baseline,
            current,
            method,
            settings,
        )
        historical_guidance = build_position_guidance(historical_verdict, settings)
        historical_timing = score_fast_timing(
            daily_asof,
            market_asof,
            current,
            settings,
        )

        rows.append(
            {
                "decision_date": decision_date.date().isoformat(),
                "market_date": cutoff_market_date.date().isoformat(),
                "fear_greed": _plain_history_value(current.get("fear_greed")),
                "fg_change_5": _plain_history_value(current.get("fg_change_5")),
                "market_regime": historical_verdict.market_regime,
                "market_extension": historical_verdict.market_extension,
                "timing_action": historical_timing.action,
                "timing_tone": historical_timing.tone,
                "timing_side": historical_timing.side,
                "timing_score": historical_timing.score,
                "timing_confirmation_count": historical_timing.confirmation_count,
                "timing_confirmation_total": historical_timing.confirmation_total,
                "timing_recommendation": historical_timing.recommendation,
                "timing_rationale": historical_timing.rationale,
                "recent_fg_low_5": _plain_history_value(historical_timing.recent_fg_low_5),
                "recent_fg_high_5": _plain_history_value(historical_timing.recent_fg_high_5),
                "distance_from_252d_high": _plain_history_value(
                    current.get("distance_from_252d_high")
                ),
                "market_return_3d": _plain_history_value(current.get("market_return_3d")),
                "market_return_5d": _plain_history_value(current.get("market_return_5d")),
                "market_return_20d": _plain_history_value(current.get("market_return_20d")),
                "action": historical_verdict.action,
                "confidence": historical_verdict.confidence,
                "sizing_tier": historical_guidance.tier,
                "sizing_label": historical_guidance.sizing_label,
                "analog_sample": historical_verdict.sample_size,
                "regime_baseline_sample": historical_verdict.regime_baseline_sample,
                "required_sample": historical_verdict.required_sample,
                "win_rate_5d": _plain_history_value(historical_verdict.win_rate_5d),
                "average_5d": _plain_history_value(historical_verdict.average_5d),
                "median_5d": _plain_history_value(historical_verdict.median_5d),
                "average_20d": _plain_history_value(historical_verdict.average_20d),
                "regime_baseline_5d": _plain_history_value(
                    historical_verdict.regime_baseline_5d
                ),
                "excess_5d": _plain_history_value(historical_verdict.excess_5d),
                "excess_ci_low_5d": _plain_history_value(
                    historical_verdict.excess_ci_low_5d
                ),
                "excess_ci_high_5d": _plain_history_value(
                    historical_verdict.excess_ci_high_5d
                ),
                "average_drawdown_20d": _plain_history_value(
                    historical_verdict.average_drawdown_20d
                ),
                "positive_checks_passed": historical_verdict.positive_checks_passed,
                "positive_checks_total": historical_verdict.positive_checks_total,
                "analog_method": historical_verdict.analog_method,
                "rationale": historical_verdict.rationale,
            }
        )

        if progress_every > 0 and (ordinal % progress_every == 0 or ordinal == total):
            print(f"[history] replayed {ordinal}/{total} decision dates")

    return pd.DataFrame(rows, columns=HISTORY_OUTPUT_COLUMNS)


def decision_change_rows(history: pd.DataFrame) -> pd.DataFrame:
    """Return the first decision and every later date when headline action changed."""
    if history.empty:
        return history.copy()
    changed = history["action"].ne(history["action"].shift(1))
    return history.loc[changed].reset_index(drop=True)


def timing_change_rows(history: pd.DataFrame) -> pd.DataFrame:
    """Return dates when the fast timing action changes."""
    if history.empty or "timing_action" not in history.columns:
        return history.iloc[0:0].copy()
    changed = history["timing_action"].ne(history["timing_action"].shift(1))
    return history.loc[changed].reset_index(drop=True)


def evaluate_timing_signals(
    timing_changes: pd.DataFrame,
    market: pd.DataFrame,
    *,
    window_sessions: int = 10,
) -> pd.DataFrame:
    """
    Evaluate early signals against the local low/high in a +/- window.

    IMPORTANT: this function is diagnostics only. It runs after signals are
    generated and its output is never fed back into score_fast_timing().
    Positive trading_days_to_extreme means the signal came before the local
    turning point; negative means it came after.
    """
    columns = [
        "decision_date",
        "market_date",
        "timing_action",
        "side",
        "signal_price",
        "local_extreme_date",
        "local_extreme_price",
        "trading_days_to_extreme",
        "price_gap_to_extreme",
    ]
    if timing_changes.empty:
        return pd.DataFrame(columns=columns)

    market_sorted = market.sort_index()
    market_index = pd.DatetimeIndex(market_sorted.index)
    closes = pd.to_numeric(market_sorted["close"], errors="coerce")

    records: list[dict[str, Any]] = []
    for _, row in timing_changes.iterrows():
        action = str(row.get("timing_action", ""))
        if not (action.startswith("EARLY BUY") or action.startswith("EARLY TRIM")):
            continue

        signal_market_date = _as_timestamp(row["market_date"])
        if signal_market_date not in market_index:
            continue

        pos = int(market_index.get_loc(signal_market_date))
        start = max(0, pos - window_sessions)
        end = min(len(market_index), pos + window_sessions + 1)
        window = closes.iloc[start:end].dropna()
        if window.empty:
            continue

        if action.startswith("EARLY BUY"):
            extreme_date = window.idxmin()
            side = "BUY"
        else:
            extreme_date = window.idxmax()
            side = "TRIM"

        extreme_pos = int(market_index.get_loc(extreme_date))
        signal_price = float(closes.iloc[pos])
        extreme_price = float(closes.loc[extreme_date])
        price_gap = (
            signal_price / extreme_price - 1.0
            if side == "BUY"
            else extreme_price / signal_price - 1.0
        )

        records.append(
            {
                "decision_date": row["decision_date"],
                "market_date": row["market_date"],
                "timing_action": action,
                "side": side,
                "signal_price": signal_price,
                "local_extreme_date": pd.Timestamp(extreme_date).date().isoformat(),
                "local_extreme_price": extreme_price,
                "trading_days_to_extreme": extreme_pos - pos,
                "price_gap_to_extreme": price_gap,
            }
        )

    return pd.DataFrame(records, columns=columns)


def _history_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, (np.integer, int)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                numeric = float(value)
                clean[key] = numeric if math.isfinite(numeric) else None
            elif value is None or pd.isna(value):
                clean[key] = None
            else:
                clean[key] = value
        records.append(clean)
    return records


# =============================================================================
# Rendering helpers
# =============================================================================

def fmt_pct(value: Optional[float], digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def fmt_num(value: Optional[float], digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def pretty_regime(value: str) -> str:
    mapping = {
        "near_high_uptrend": "Near high / uptrend",
        "uptrend_off_high": "Uptrend below high",
        "correction": "Correction",
        "downtrend": "Downtrend",
        "unavailable": "Unavailable",
    }
    return mapping.get(value, value.replace("_", " ").title())


CHART_BG = "rgba(0,0,0,0)"
CHART_GRID = "rgba(148,163,184,.14)"
CHART_TEXT = "#a6b2c4"
CHART_LINE = "#5aa9ff"
SENTIMENT_SCALE = [
    [0.0, "#c0392b"],
    [0.25, "#e07a3f"],
    [0.5, "#e8c547"],
    [0.75, "#7fbf6b"],
    [1.0, "#2fa860"],
]


def render_chart(
    daily: pd.DataFrame,
    market: pd.DataFrame,
    timing_changes: Optional[pd.DataFrame] = None,
) -> str:
    visible_market = market.loc[market.index >= daily.index.min()]
    price = visible_market["close"]
    padding = (price.max() - price.min()) * 0.08 or price.max() * 0.02

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.035,
    )

    figure.add_trace(
        go.Scatter(
            x=price.index,
            y=price.values,
            name="S&P 500 close",
            mode="lines",
            line={"width": 2.2, "color": CHART_LINE},
            fill="tozeroy",
            fillcolor="rgba(90,169,255,.14)",
            hovertemplate="%{y:,.0f}<extra>S&P 500</extra>",
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Heatmap(
            x=daily.index,
            y=[""],
            z=[daily["fear_greed"].values],
            zmin=0,
            zmax=100,
            colorscale=SENTIMENT_SCALE,
            showscale=False,
            hovertemplate=(
                "%{x|%Y-%m-%d}: %{z:.0f}<extra>Fear &amp; Greed</extra>"
            ),
        ),
        row=2,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["fear_greed"],
            name="Fear & Greed",
            mode="lines",
            line={"width": 1.4, "color": "rgba(255,255,255,.85)"},
            yaxis="y3",
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )

    if timing_changes is not None and not timing_changes.empty:
        marker_specs = [
            ("EARLY BUY", "triangle-up", "#67d391", "Early buy"),
            ("EARLY TRIM", "triangle-down", "#ee796b", "Early trim"),
        ]
        for prefix, symbol, color, label in marker_specs:
            subset = timing_changes[
                timing_changes["timing_action"].astype(str).str.startswith(prefix)
            ].copy()
            if subset.empty:
                continue
            x_values: list[pd.Timestamp] = []
            y_values: list[float] = []
            hover_text: list[str] = []
            for _, row in subset.iterrows():
                market_date = pd.Timestamp(row["market_date"])
                if market_date not in market.index:
                    continue
                x_values.append(market_date)
                y_values.append(float(market.loc[market_date, "close"]))
                hover_text.append(
                    f'{row["timing_action"]}<br>F&G {float(row["fear_greed"]):.1f}'
                    f'<br>Score {int(row["timing_score"])}'
                )
            if x_values:
                figure.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode="markers",
                        name=label,
                        marker={
                            "symbol": symbol,
                            "size": 11,
                            "color": color,
                            "line": {"width": 1, "color": "rgba(255,255,255,.7)"},
                        },
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

    figure.update_layout(
        template="plotly_dark",
        height=430,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font={"color": CHART_TEXT, "size": 12},
        margin={"l": 50, "r": 20, "t": 10, "b": 34},
        hovermode="x unified",
        showlegend=timing_changes is not None and not timing_changes.empty,
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        yaxis={
            "title": None,
            "range": [price.min() - padding, price.max() + padding],
            "gridcolor": CHART_GRID,
            "zeroline": False,
        },
        yaxis2={"showticklabels": False, "ticks": ""},
        yaxis3={"overlaying": "y2", "range": [0, 100], "visible": False},
        xaxis2={"gridcolor": CHART_GRID, "title": None},
    )
    figure.update_xaxes(showgrid=False, row=1, col=1)

    return figure.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displaylogo": False},
    )


def render_gauge(value: float) -> str:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(value, 1),
            number={"suffix": "", "font": {"size": 34, "color": "#e7ecf3"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": CHART_TEXT,
                    "tickfont": {"size": 9, "color": CHART_TEXT},
                },
                "bar": {"color": "rgba(255,255,255,.9)", "thickness": 0.22},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "#c0392b"},
                    {"range": [25, 45], "color": "#e07a3f"},
                    {"range": [45, 55], "color": "#e8c547"},
                    {"range": [55, 75], "color": "#7fbf6b"},
                    {"range": [75, 100], "color": "#2fa860"},
                ],
            },
        )
    )
    figure.update_layout(
        height=150,
        paper_bgcolor=CHART_BG,
        font={"color": CHART_TEXT},
        margin={"l": 18, "r": 18, "t": 8, "b": 0},
    )
    return figure.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displaylogo": False, "staticPlot": True},
    )


def render_analog_outcomes_chart(
    analogs: pd.DataFrame,
    regime_baseline: pd.DataFrame,
) -> str:
    analog_returns = (
        pd.to_numeric(analogs.get("forward_5d"), errors="coerce").dropna() * 100
    )
    baseline_returns = (
        pd.to_numeric(regime_baseline.get("forward_5d"), errors="coerce").dropna() * 100
    )

    figure = go.Figure()

    if not baseline_returns.empty:
        figure.add_trace(
            go.Box(
                x=baseline_returns,
                name="Same-regime baseline",
                boxpoints=False,
                marker_color="rgba(148,163,184,.55)",
                line_color="rgba(148,163,184,.65)",
                fillcolor="rgba(148,163,184,.12)",
                orientation="h",
            )
        )

    if not analog_returns.empty:
        figure.add_trace(
            go.Box(
                x=analog_returns,
                name="Current analogs",
                boxpoints="all",
                jitter=0.45,
                pointpos=0,
                marker={"color": CHART_LINE, "size": 5, "opacity": 0.75},
                line_color=CHART_LINE,
                fillcolor="rgba(90,169,255,.10)",
                orientation="h",
            )
        )

    figure.add_vline(
        x=0,
        line_width=1,
        line_dash="dot",
        line_color="rgba(255,255,255,.35)",
    )

    figure.update_layout(
        template="plotly_dark",
        height=190,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font={"color": CHART_TEXT, "size": 11},
        margin={"l": 110, "r": 24, "t": 10, "b": 32},
        showlegend=False,
        xaxis={
            "title": "5-day forward return (%)",
            "gridcolor": CHART_GRID,
            "zeroline": False,
        },
        yaxis={"gridcolor": CHART_GRID},
    )

    return figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
    )


def render_table(
    frame: pd.DataFrame,
    percent_columns: set[str] = frozenset(),
) -> str:
    display = frame.copy()

    for column in display.columns:
        if column in percent_columns:
            display[column] = display[column].map(
                lambda value: fmt_pct(value, 2) if pd.notna(value) else "—"
            )
        elif pd.api.types.is_datetime64_any_dtype(display[column]):
            display[column] = display[column].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: f"{value:.2f}" if pd.notna(value) else "—"
            )

    return display.to_html(
        index=False,
        classes="data-table",
        border=0,
        escape=True,
    )


def _history_pct(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _history_num(value: Any, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _history_action_class(action: str) -> str:
    if action.startswith("EARLY BUY"):
        return "buy"
    if action.startswith("EARLY TRIM"):
        return "wait"
    if action in {"EXTREME BUY ZONE — WAIT FOR STABILIZATION", "BUY WATCH — OVERSOLD"}:
        return "hold"
    if action in {"OVERHEATED — NO EXTRA BUYING", "OVERHEATED WATCH"}:
        return "hold"
    if action == "BUY GRADUALLY":
        return "buy"
    if action == "WAIT ON BUYING":
        return "wait"
    if action == "HOLD / NO EXTRA BUYING":
        return "hold"
    return "insufficient"


def _render_timing_rows(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<tr><td colspan="11" class="empty">No timing decisions in this view.</td></tr>'

    rendered: list[str] = []
    for _, row in frame.iterrows():
        action = str(row["timing_action"])
        rendered.append(
            "<tr "
            f'data-timing-action="{html_lib.escape(action, quote=True)}" '
            f'data-side="{html_lib.escape(str(row.get("timing_side", "")), quote=True)}">'
            f'<td>{html_lib.escape(str(row["decision_date"]))}</td>'
            f'<td><span class="action action-{_history_action_class(action)}">{html_lib.escape(action)}</span></td>'
            f'<td>{_history_num(row.get("fear_greed"), 1)}</td>'
            f'<td>{_history_num(row.get("recent_fg_low_5"), 1)}</td>'
            f'<td>{_history_num(row.get("recent_fg_high_5"), 1)}</td>'
            f'<td>{int(row.get("timing_score", 0))}</td>'
            f'<td>{int(row.get("timing_confirmation_count", 0))}/{int(row.get("timing_confirmation_total", 0))}</td>'
            f'<td>{html_lib.escape(str(row.get("market_regime", "")))}</td>'
            f'<td>{_history_pct(row.get("distance_from_252d_high")) if "distance_from_252d_high" in row else "—"}</td>'
            f'<td>{html_lib.escape(str(row.get("timing_recommendation", "")))}</td>'
            f'<td>{html_lib.escape(str(row.get("action", "")))}</td>'
            "</tr>"
        )
    return "\n".join(rendered)


def _render_history_rows(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<tr><td colspan="13" class="empty">No decisions in this view.</td></tr>'

    rendered: list[str] = []
    for _, row in frame.iterrows():
        action = str(row["action"])
        rendered.append(
            "<tr "
            f'data-action="{html_lib.escape(action, quote=True)}" '
            f'data-date="{html_lib.escape(str(row["decision_date"]), quote=True)}">'
            f'<td>{html_lib.escape(str(row["decision_date"]))}</td>'
            f'<td><span class="action action-{_history_action_class(action)}">{html_lib.escape(action)}</span></td>'
            f'<td>{html_lib.escape(str(row["confidence"]))}</td>'
            f'<td>{_history_num(row["fear_greed"], 1)}</td>'
            f'<td>{html_lib.escape(str(row["market_regime"]))}</td>'
            f'<td>{html_lib.escape(str(row["market_extension"]))}</td>'
            f'<td>{int(row["analog_sample"])}</td>'
            f'<td>{int(row["regime_baseline_sample"])}</td>'
            f'<td>{_history_pct(row["excess_5d"])}</td>'
            f'<td>{_history_pct(row["excess_ci_low_5d"])}</td>'
            f'<td>{_history_pct(row["excess_ci_high_5d"])}</td>'
            f'<td>{_history_pct(row["average_drawdown_20d"])}</td>'
            f'<td>{html_lib.escape(str(row["sizing_label"]))}</td>'
            "</tr>"
        )
    return "\n".join(rendered)


def render_history_page(
    history: pd.DataFrame,
    changes: pd.DataFrame,
    timing_changes: pd.DataFrame,
    timing_evaluation: pd.DataFrame,
    *,
    generated_at: datetime,
    data_source: str,
) -> str:
    counts = history["action"].value_counts().to_dict() if not history.empty else {}
    timing_counts = (
        history["timing_action"].value_counts().to_dict()
        if not history.empty and "timing_action" in history.columns
        else {}
    )
    buy_count = int(counts.get("BUY GRADUALLY", 0))
    wait_count = int(counts.get("WAIT ON BUYING", 0))
    early_buy_count = int(
        sum(count for action, count in timing_counts.items() if str(action).startswith("EARLY BUY"))
    )
    early_trim_count = int(
        sum(count for action, count in timing_counts.items() if str(action).startswith("EARLY TRIM"))
    )
    first_date = str(history.iloc[0]["decision_date"]) if not history.empty else "—"
    last_date = str(history.iloc[-1]["decision_date"]) if not history.empty else "—"

    eval_buy = timing_evaluation[timing_evaluation["side"] == "BUY"] if not timing_evaluation.empty else timing_evaluation
    eval_trim = timing_evaluation[timing_evaluation["side"] == "TRIM"] if not timing_evaluation.empty else timing_evaluation

    def timing_eval_summary(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No evaluated signals yet"
        median_days = float(frame["trading_days_to_extreme"].median())
        median_gap = float(frame["price_gap_to_extreme"].median()) * 100
        relation = "before" if median_days >= 0 else "after"
        return f"Median {abs(median_days):.1f} sessions {relation} local turn · {median_gap:.2f}% price gap"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="robots" content="noindex,nofollow">
  <title>Fear &amp; Greed Decision History</title>
  <link rel="stylesheet" href="styles.css">
  <style>
    body {{ min-height:100vh; }}
    .history-shell {{ max-width:1600px; margin:0 auto; padding:24px; }}
    .history-top {{ display:flex; justify-content:space-between; gap:18px; align-items:flex-start; flex-wrap:wrap; margin-bottom:16px; }}
    .history-top h1 {{ font-size:1.5rem; margin:3px 0 6px; }}
    .history-top p {{ color:var(--muted); max-width:1050px; line-height:1.5; font-size:.86rem; }}
    .back-link {{ text-decoration:none; border:1px solid var(--border); border-radius:9px; padding:8px 12px; font-weight:700; }}
    .summary-grid {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:10px; margin-bottom:16px; }}
    .summary-card {{ background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:12px; }}
    .summary-card small {{ display:block; color:var(--muted); font-size:.68rem; margin-bottom:4px; }}
    .summary-card strong {{ font-size:1.05rem; }}
    .history-table-wrap {{ overflow:auto; max-height:68vh; border:1px solid var(--border-soft); border-radius:12px; }}
    .history-table {{ width:100%; border-collapse:collapse; white-space:nowrap; font-size:.75rem; }}
    .history-table th,.history-table td {{ padding:8px 9px; border-bottom:1px solid var(--border-soft); text-align:right; }}
    .history-table th:first-child,.history-table td:first-child,.history-table th:nth-child(2),.history-table td:nth-child(2) {{ text-align:left; }}
    .history-table thead th {{ position:sticky; top:0; z-index:2; background:var(--panel-2); color:var(--muted); }}
    .history-table tbody tr:hover {{ background:rgba(90,169,255,.06); }}
    .action {{ display:inline-block; padding:3px 7px; border-radius:999px; font-size:.68rem; font-weight:800; }}
    .action-buy {{ color:#67d391; background:rgba(47,168,96,.16); }}
    .action-wait {{ color:#ee796b; background:rgba(209,80,63,.16); }}
    .action-hold {{ color:#e3c55a; background:rgba(210,167,45,.16); }}
    .action-insufficient {{ color:#a7b0bd; background:rgba(148,163,184,.10); }}
    .method-note {{ color:var(--faint); font-size:.75rem; line-height:1.5; margin-top:12px; }}
    .downloads {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }}
    .downloads a {{ font-size:.76rem; font-weight:700; }}
    .empty {{ text-align:center !important; color:var(--faint); padding:24px !important; }}
    .section-gap {{ margin-top:16px; }}
    @media (max-width:1100px) {{ .summary-grid {{ grid-template-columns:repeat(3,1fr); }} }}
    @media (max-width:700px) {{ .history-shell {{ padding:14px; }} .summary-grid {{ grid-template-columns:repeat(2,1fr); }} }}
  </style>
</head>
<body>
  <main class="history-shell">
    <div class="history-top">
      <div>
        <div class="eyebrow">POINT-IN-TIME REPLAY + FAST TIMING</div>
        <h1>Historical Dashboard Decisions</h1>
        <p>
          The fast timing layer is intentionally earlier than the conservative analog model.
          It scores fear/greed extremes, drawdown/extension, and same-day stabilization or
          rollover evidence. The conservative decision still uses the point-in-time analog
          replay. Future local lows/highs are used only in the diagnostic evaluation CSV and
          never to generate a timing signal.
        </p>
      </div>
      <a class="back-link" href="index.html">← Back to dashboard</a>
    </div>

    <section class="summary-grid">
      <div class="summary-card"><small>Coverage</small><strong>{html_lib.escape(first_date)}</strong><small>through {html_lib.escape(last_date)}</small></div>
      <div class="summary-card"><small>Early BUY days</small><strong>{early_buy_count}</strong></div>
      <div class="summary-card"><small>Early TRIM days</small><strong>{early_trim_count}</strong></div>
      <div class="summary-card"><small>Confirmed BUY days</small><strong>{buy_count}</strong></div>
      <div class="summary-card"><small>WAIT days</small><strong>{wait_count}</strong></div>
      <div class="summary-card"><small>Timing changes</small><strong>{len(timing_changes)}</strong></div>
    </section>

    <section class="panel">
      <div class="panel-heading">
        <div><div class="eyebrow">FAST TIMING CHANGES</div><h2>Earlier buy / trim dates</h2></div>
        <div class="hint">This is the table to use when judging whether signals are still too late.</div>
      </div>
      <div class="history-table-wrap">
        <table class="history-table">
          <thead><tr>
            <th>Date</th><th>Fast timing action</th><th>F&amp;G</th><th>Recent low</th>
            <th>Recent high</th><th>Score</th><th>Confirm</th><th>Regime</th>
            <th>252D DD</th><th>Suggested action</th><th>Slow model</th>
          </tr></thead>
          <tbody>{_render_timing_rows(timing_changes)}</tbody>
        </table>
      </div>
      <div class="downloads">
        <a href="timing_decision_changes.csv">Download fast timing changes CSV</a>
        <a href="timing_evaluation.csv">Download turning-point evaluation CSV</a>
        <a href="historical_decisions.csv">Download every date CSV</a>
      </div>
      <p class="method-note">
        BUY evaluation: {html_lib.escape(timing_eval_summary(eval_buy))}.<br>
        TRIM evaluation: {html_lib.escape(timing_eval_summary(eval_trim))}.
      </p>
    </section>

    <section class="panel section-gap">
      <div class="panel-heading">
        <div><div class="eyebrow">CONSERVATIVE CONFIRMATION</div><h2>Original analog-model action changes</h2></div>
        <div class="hint">Kept unchanged so you can compare how much earlier the fast layer reacts.</div>
      </div>
      <div class="history-table-wrap">
        <table class="history-table">
          <thead><tr>
            <th>Date</th><th>Action</th><th>Confidence</th><th>F&amp;G</th>
            <th>Regime</th><th>Extension</th><th>Analogs</th><th>Baseline N</th>
            <th>5D Excess</th><th>CI Low</th><th>CI High</th><th>20D DD</th><th>Sizing</th>
          </tr></thead>
          <tbody>{_render_history_rows(changes)}</tbody>
        </table>
      </div>
      <div class="downloads">
        <a href="decision_changes.csv">Download slow-model changes CSV</a>
        <a href="historical_decisions.json">Download full JSON</a>
      </div>
      <p class="method-note">
        Generated {generated_at.strftime('%Y-%m-%d %H:%M UTC')} from {html_lib.escape(data_source)}.
        Return columns in downloadable data are decimals: 0.012 means 1.2%.
      </p>
    </section>
  </main>
</body>
</html>
"""


PAGE_TEMPLATE = Template(
    r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="robots" content="noindex,nofollow">
  <title>Fear &amp; Greed Market Dashboard</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body data-build-id="{{ build_id }}" data-refresh-seconds="{{ refresh_seconds }}">
  <div class="shell">
    <aside class="sidebar">
      <div class="brand">
        <span class="dot dot-{{ verdict.tone }}"></span>
        <div>
          <p class="eyebrow">RESEARCH DASHBOARD</p>
          <h1>Fear &amp; Greed<br>vs. S&amp;P 500</h1>
        </div>
      </div>

      <div class="gauge-card">
        {{ gauge | safe }}
        <div class="gauge-caption">
          <strong>{{ metrics[0].value }}</strong> fear &amp; greed
          <span>as of {{ metrics[0].note }}</span>
        </div>
      </div>

      <div class="verdict-card verdict-{{ verdict.tone }}">
        <p class="eyebrow">TACTICAL RESEARCH ACTION</p>
        <h2>{{ verdict.action }}</h2>
        <p class="verdict-rationale">{{ verdict.rationale }}</p>
        <div class="verdict-stats">
          <div><dt>Confidence</dt><dd>{{ verdict.confidence }}</dd></div>
          <div><dt>Analogs</dt><dd>{{ verdict.sample_size }}</dd></div>
          <div><dt>Regime</dt><dd>{{ regime_label }}</dd></div>
          <div><dt>Extension</dt><dd>{{ verdict.market_extension }}</dd></div>
        </div>
        <p class="verdict-method">Method: {{ verdict.analog_method }}</p>

        <div class="check-list">
          {% for check in verdict.decision_checks %}
          <div class="check-row check-{{ 'pass' if check.passed else 'fail' }}">
            <span class="check-icon">{{ '✓' if check.passed else '×' }}</span>
            <div><strong>{{ check.label }}</strong><small>{{ check.value }} · {{ check.requirement }}</small></div>
          </div>
          {% endfor %}
        </div>
      </div>

      <dl class="metric-list">
        {% for metric in metrics[1:] %}
        <div class="metric-row"><dt>{{ metric.label }}</dt><dd>{{ metric.value }}<small>{{ metric.note }}</small></dd></div>
        {% endfor %}
      </dl>

      <div class="sidebar-foot">
        <span>Last built</span><strong>{{ generated }}</strong>
        <span id="refresh-status">Checking for updates…</span>
        <span class="source">{{ source }}</span>
      </div>
    </aside>

    <main class="workspace">
      {% if warnings %}
      <section class="warnings">{% for warning in warnings %}<div class="warning">{{ warning }}</div>{% endfor %}</section>
      {% endif %}

      <section class="panel chart-panel">
        <div class="panel-heading">
          <div><p class="eyebrow">MARKET CONTEXT</p><h2>Price vs. sentiment</h2></div>
          <span class="hint">The action compares sentiment only with historical observations in the same price regime.</span>
        </div>
        {{ chart | safe }}
      </section>

      <section class="panel timing-panel timing-{{ timing.tone }}">
        <div class="panel-heading">
          <div><p class="eyebrow">FAST TIMING LAYER</p><h2>{{ timing.action }}</h2></div>
          <span class="hint">Designed to react earlier than the analog model. Uses only current/backward-looking data.</span>
        </div>
        <div class="timing-topline">
          <span class="action-sizing-label">{{ timing.recommendation }}</span>
        </div>
        <p class="action-detail">{{ timing.rationale }}</p>
        <div class="timing-stats">
          <div><dt>Side</dt><dd>{{ timing.side }}</dd></div>
          <div><dt>Stress / heat score</dt><dd>{{ timing.score }}</dd></div>
          <div><dt>Confirmations</dt><dd>{{ timing.confirmation_count }}/{{ timing.confirmation_total }}</dd></div>
          <div><dt>Recent F&amp;G range</dt><dd>{{ '%.1f'|format(timing.recent_fg_low_5) if timing.recent_fg_low_5 is not none else 'N/A' }} – {{ '%.1f'|format(timing.recent_fg_high_5) if timing.recent_fg_high_5 is not none else 'N/A' }}</dd></div>
        </div>
        <div class="timing-checks">
          {% for check in timing.checks %}
          <div class="timing-check timing-check-{{ 'pass' if check.passed else 'fail' }}">
            <span>{{ '✓' if check.passed else '·' }}</span>
            <div><strong>{{ check.label }}</strong><small>{{ check.value }}</small></div>
          </div>
          {% endfor %}
        </div>
      </section>

      <section class="panel action-panel action-{{ verdict.tone }}">
        <div class="panel-heading">
          <div><p class="eyebrow">SLOW CONFIRMATION LAYER</p><h2>{{ guidance.tier }}</h2></div>
          <span class="hint">The original analog/bootstrap model remains unchanged and acts as confirmation.</span>
        </div>
        <div class="action-sizing"><span class="action-sizing-label">{{ guidance.sizing_label }}</span></div>
        <p class="action-detail">{{ guidance.sizing_detail }}</p>
        <p class="action-guardrail">{{ guidance.guardrail }}</p>
      </section>

      <section class="panel history-panel">
        <div class="panel-heading">
          <div><p class="eyebrow">POINT-IN-TIME REPLAY</p><h2>Historical Decisions</h2></div>
          <a href="decision_history.html">Open full history →</a>
        </div>
        <p class="history-description">
          Compare the new fast timing layer against the original confirmation model using only
          information knowable on each replay date. {{ history_summary.total }} dates replayed;
          {{ history_summary.timing_changes }} fast timing changes.
        </p>
        <div class="history-mini-stats">
          <div><dt>Early BUY days</dt><dd>{{ history_summary.early_buy }}</dd></div>
          <div><dt>Early TRIM days</dt><dd>{{ history_summary.early_trim }}</dd></div>
          <div><dt>Confirmed BUY days</dt><dd>{{ history_summary.buy }}</dd></div>
          <div><dt>Fast changes</dt><dd>{{ history_summary.timing_changes }}</dd></div>
        </div>
        <div class="history-downloads">
          <a href="timing_decision_changes.csv" download>Fast timing changes CSV</a>
          <a href="timing_evaluation.csv" download>Timing evaluation CSV</a>
          <a href="historical_decisions.csv" download>Every date CSV</a>
        </div>
      </section>

      <div class="table-grid">
        <section class="panel table-panel outcomes-panel">
          <div class="panel-heading">
            <div><p class="eyebrow">ANALOG OUTCOMES</p><h2>How similar setups actually played out</h2></div>
            <a href="analogs.csv" download>CSV</a>
          </div>
          {{ outcomes_chart | safe }}
          <div class="outcomes-stats">
            <div><dt>Win rate</dt><dd>{{ outcomes_stats.win_rate }}</dd></div>
            <div><dt>Median 5D</dt><dd>{{ outcomes_stats.median_5d }}</dd></div>
            <div><dt>Worst 5D</dt><dd>{{ outcomes_stats.worst_5d }}</dd></div>
            <div><dt>Analogs</dt><dd>{{ outcomes_stats.sample_size }}</dd></div>
          </div>
          <details><summary>Show full analog table</summary><div class="table-wrap">{{ analog_table | safe }}</div></details>
        </section>

        <section class="panel table-panel scorecard-panel">
          <div class="panel-heading">
            <div><p class="eyebrow">SIGNAL SCORECARD</p><h2>Backtests active right now</h2></div>
            <a href="event_study.csv" download>CSV</a>
          </div>
          {% if scorecard %}
          <div class="scorecard-list">
            {% for item in scorecard %}
            <div class="scorecard-row scorecard-{{ item.tone }}">
              <strong>{{ item.label }}</strong>
              <div class="scorecard-stats"><span>{{ item.events }} events</span><span>Win {{ item.win_rate }}</span><span>Avg {{ item.average }}</span><span>Edge {{ item.excess }}</span></div>
            </div>
            {% endfor %}
          </div>
          {% else %}
          <p class="empty-note">No backtested threshold or drop signal currently matches today's setup closely enough to score. See the full table below.</p>
          {% endif %}
          <details><summary>Show full backtest table</summary><div class="table-wrap">{{ event_study_table | safe }}</div></details>
        </section>
      </div>

      <section class="panel note-panel">
        <p class="eyebrow">HOW TO READ THIS</p>
        <p>
          The FAST TIMING layer can issue an EARLY BUY or EARLY TRIM before the slower analog
          model has enough completed historical evidence. BUY GRADUALLY still requires positive
          same-regime excess return and bootstrap confirmation. The two layers are intentionally
          separate: the fast layer improves timing; the slow layer improves confidence. Future local
          lows/highs are used only to evaluate timing after the fact, never to create a signal.
        </p>
      </section>
    </main>
  </div>

  <script src="app.js"></script>
</body>
</html>
"""
)


STYLES_CSS = r"""
:root {
  color-scheme: dark;
  --bg: #0a0e14; --sidebar: #0d1420; --panel: #111a26; --panel-2: #0d1520;
  --text: #e7ecf3; --muted: #8a97ab; --faint: #59667a;
  --border: #1f2c3d; --border-soft: #182131; --accent: #5aa9ff;
  --positive: #2fa860; --positive-bg: rgba(47,168,96,.12);
  --negative: #d1503f; --negative-bg: rgba(209,80,63,.12);
  --mixed: #d2a72d; --mixed-bg: rgba(210,167,45,.12);
  --neutral-bg: rgba(148,163,184,.08);
  font-family: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body { margin: 0; background: var(--bg); color: var(--text); -webkit-font-smoothing: antialiased; }
h1, h2, h3, p, dl, dd, dt { margin: 0; }
a { color: var(--accent); }
.shell { display: grid; grid-template-columns: 330px 1fr; height: 100vh; height: 100dvh; }
.sidebar { background: var(--sidebar); border-right: 1px solid var(--border); padding: 22px 18px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; }
.workspace { overflow-y: auto; padding: 22px 26px 36px; display: grid; align-content: start; gap: 16px; }
.eyebrow { color: var(--accent); font-size: .68rem; font-weight: 800; letter-spacing: .12em; margin-bottom: 4px; }
.brand { display: flex; align-items: center; gap: 10px; padding-bottom: 4px; }
.brand h1 { font-size: 1.05rem; line-height: 1.25; font-weight: 800; }
.dot { width: 10px; height: 10px; border-radius: 50%; flex: none; box-shadow: 0 0 10px currentColor; }
.dot-positive { background: var(--positive); color: var(--positive); }
.dot-negative { background: var(--negative); color: var(--negative); }
.dot-mixed { background: var(--mixed); color: var(--mixed); }
.dot-neutral { background: var(--faint); color: var(--faint); }
.gauge-card { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 4px 4px 10px; text-align: center; }
.gauge-card .plotly-graph-div { margin: 0 auto; }
.gauge-caption { font-size: .78rem; color: var(--muted); margin-top: -6px; }
.gauge-caption strong { color: var(--text); font-size: .95rem; }
.gauge-caption span { display: block; }
.verdict-card { border: 1px solid var(--border); border-radius: 14px; padding: 16px; background: var(--neutral-bg); }
.verdict-positive { border-color: rgba(47,168,96,.4); background: var(--positive-bg); }
.verdict-negative { border-color: rgba(209,80,63,.4); background: var(--negative-bg); }
.verdict-mixed { border-color: rgba(210,167,45,.4); background: var(--mixed-bg); }
.verdict-card h2 { font-size: 1.18rem; letter-spacing: .01em; margin: 2px 0 8px; }
.verdict-rationale { font-size: .82rem; color: var(--muted); line-height: 1.45; }
.verdict-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 12px 0 8px; }
.verdict-stats div { background: rgba(0,0,0,.18); border-radius: 9px; padding: 8px 10px; }
.verdict-stats dt { color: var(--muted); font-size: .68rem; }
.verdict-stats dd { font-weight: 800; margin-top: 2px; font-size: .8rem; }
.verdict-method { font-size: .72rem; color: var(--faint); line-height: 1.35; }
.check-list { display: grid; gap: 5px; margin-top: 12px; }
.check-row { display: grid; grid-template-columns: 18px 1fr; gap: 7px; align-items: start; padding: 6px 8px; border-radius: 8px; background: rgba(0,0,0,.14); }
.check-icon { font-weight: 900; line-height: 1.1; }
.check-pass .check-icon { color: var(--positive); }
.check-fail .check-icon { color: var(--negative); }
.check-row strong { display: block; font-size: .7rem; }
.check-row small { display: block; color: var(--faint); font-size: .64rem; margin-top: 1px; }
.metric-list { display: grid; gap: 1px; background: var(--border-soft); border: 1px solid var(--border-soft); border-radius: 12px; overflow: hidden; }
.metric-row { background: var(--panel); padding: 9px 12px; display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.metric-row dt { color: var(--muted); font-size: .74rem; }
.metric-row dd { text-align: right; font-weight: 700; font-size: .84rem; }
.metric-row dd small { display: block; font-weight: 400; color: var(--faint); font-size: .66rem; }
.sidebar-foot { margin-top: auto; padding-top: 12px; border-top: 1px solid var(--border-soft); display: grid; gap: 3px; font-size: .72rem; color: var(--muted); }
.sidebar-foot strong { color: var(--text); font-size: .78rem; }
.sidebar-foot .source { color: var(--faint); }
.warnings { display: grid; gap: 8px; }
.warning { padding: 10px 14px; border: 1px solid rgba(210,167,45,.4); border-radius: 10px; background: var(--mixed-bg); color: #e8cf7a; font-size: .85rem; }
.panel { border: 1px solid var(--border); border-radius: 16px; background: var(--panel); padding: 18px 20px; }
.panel-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; margin-bottom: 12px; flex-wrap: wrap; }
.panel-heading h2 { font-size: 1.05rem; margin-top: 2px; }
.panel-heading a { font-weight: 700; font-size: .82rem; text-decoration: none; border: 1px solid var(--border); border-radius: 8px; padding: 5px 10px; }
.panel-heading a:hover { border-color: var(--accent); }
.hint { color: var(--faint); font-size: .76rem; max-width: 340px; text-align: right; }
.chart-panel .plotly-graph-div { width: 100% !important; }
.action-panel { border-width: 1px; }
.action-positive { border-color: rgba(47,168,96,.4); background: linear-gradient(180deg, var(--positive-bg), var(--panel) 70%); }
.action-negative { border-color: rgba(209,80,63,.4); background: linear-gradient(180deg, var(--negative-bg), var(--panel) 70%); }
.action-mixed { border-color: rgba(210,167,45,.4); background: linear-gradient(180deg, var(--mixed-bg), var(--panel) 70%); }
.action-sizing { margin: 4px 0 10px; }
.action-sizing-label { display: inline-block; font-size: 1.1rem; font-weight: 800; padding: 7px 16px; border-radius: 999px; background: rgba(90,169,255,.14); color: var(--accent); }
.action-detail { font-size: .85rem; color: var(--muted); line-height: 1.5; margin-bottom: 8px; }
.action-guardrail { font-size: .76rem; color: var(--faint); font-style: italic; }
.timing-panel { border-width:1px; }
.timing-positive { border-color:rgba(47,168,96,.5); background:linear-gradient(180deg,rgba(47,168,96,.12),var(--panel) 72%); }
.timing-negative { border-color:rgba(209,80,63,.5); background:linear-gradient(180deg,rgba(209,80,63,.12),var(--panel) 72%); }
.timing-mixed { border-color:rgba(210,167,45,.48); background:linear-gradient(180deg,rgba(210,167,45,.10),var(--panel) 72%); }
.timing-neutral { border-color:var(--border); }
.timing-topline { margin:4px 0 10px; }
.timing-stats { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin:12px 0; }
.timing-stats div { background:rgba(0,0,0,.18); border-radius:9px; padding:9px 10px; }
.timing-stats dt { color:var(--muted); font-size:.68rem; }
.timing-stats dd { font-weight:800; margin-top:2px; font-size:.82rem; }
.timing-checks { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:6px; margin-top:10px; }
.timing-check { display:grid; grid-template-columns:18px 1fr; gap:7px; align-items:start; background:rgba(0,0,0,.14); border-radius:8px; padding:7px 9px; }
.timing-check > span { font-weight:900; }
.timing-check-pass > span { color:var(--positive); }
.timing-check-fail > span { color:var(--faint); }
.timing-check strong { display:block; font-size:.72rem; }
.timing-check small { display:block; color:var(--faint); font-size:.65rem; margin-top:2px; }
.history-description { color: var(--muted); font-size:.84rem; line-height:1.5; }
.history-mini-stats { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin:12px 0; }
.history-mini-stats div { background:rgba(0,0,0,.18); border-radius:9px; padding:9px 10px; }
.history-mini-stats dt { color:var(--muted); font-size:.68rem; }
.history-mini-stats dd { font-weight:800; margin-top:2px; }
.history-downloads { display:flex; gap:10px; flex-wrap:wrap; }
.history-downloads a { font-size:.76rem; font-weight:700; }
.table-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }
.table-panel { display: flex; flex-direction: column; min-width: 0; }
.table-wrap { overflow: auto; border: 1px solid var(--border-soft); border-radius: 10px; max-height: 320px; }
.data-table { width: 100%; border-collapse: collapse; font-size: .78rem; white-space: nowrap; }
.data-table th, .data-table td { padding: 7px 10px; border-bottom: 1px solid var(--border-soft); text-align: right; }
.data-table th:first-child, .data-table td:first-child { text-align: left; }
.data-table thead th { position: sticky; top: 0; background: var(--panel-2); color: var(--muted); font-weight: 700; z-index: 1; }
.data-table tbody tr:hover { background: rgba(90,169,255,.06); }
.outcomes-panel .plotly-graph-div { width: 100% !important; }
.outcomes-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 10px 0 4px; }
.outcomes-stats div { background: rgba(0,0,0,.18); border-radius: 9px; padding: 8px 6px; text-align: center; }
.outcomes-stats dt { color: var(--muted); font-size: .64rem; display: block; }
.outcomes-stats dd { font-weight: 800; font-size: .84rem; margin-top: 2px; }
.scorecard-list { display: grid; gap: 8px; }
.scorecard-row { border-radius: 10px; padding: 10px 12px; background: rgba(0,0,0,.16); border-left: 3px solid var(--faint); }
.scorecard-positive { border-left-color: var(--positive); background: var(--positive-bg); }
.scorecard-negative { border-left-color: var(--negative); background: var(--negative-bg); }
.scorecard-row strong { display: block; font-size: .82rem; margin-bottom: 5px; }
.scorecard-stats { display: flex; gap: 12px; flex-wrap: wrap; font-size: .72rem; color: var(--muted); }
.empty-note { color: var(--faint); font-size: .82rem; line-height: 1.5; }
details { margin-top: 12px; }
details summary { cursor: pointer; font-size: .76rem; color: var(--accent); font-weight: 700; padding: 4px 0; list-style: none; }
details summary::-webkit-details-marker { display: none; }
details summary::before { content: "▸ "; }
details[open] summary::before { content: "▾ "; }
details .table-wrap { margin-top: 8px; }
.note-panel p { font-size: .8rem; color: var(--muted); line-height: 1.5; }
@media (max-width: 1050px) { .table-grid { grid-template-columns: 1fr; } }
@media (max-width: 860px) {
  .shell { grid-template-columns: 1fr; height: auto; }
  .sidebar { border-right: none; border-bottom: 1px solid var(--border); overflow-y: visible; }
  .workspace { overflow-y: visible; padding: 18px 16px 30px; }
  .metric-list { grid-template-columns: 1fr 1fr; display: grid; }
  .hint { text-align: left; max-width: none; }
  .outcomes-stats { grid-template-columns: 1fr 1fr; }
  .history-mini-stats { grid-template-columns:1fr 1fr; }
  .timing-stats { grid-template-columns:1fr 1fr; }
  .timing-checks { grid-template-columns:1fr; }
}
"""


APP_JS = r"""
const currentBuild = document.body.dataset.buildId;
const intervalSeconds = Number(document.body.dataset.refreshSeconds || 300);
const statusNode = document.getElementById("refresh-status");

async function checkForUpdate() {
  try {
    const response = await fetch(`version.json?t=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const version = await response.json();
    if (version.build_id && version.build_id !== currentBuild) {
      statusNode.textContent = "New dashboard available — refreshing";
      window.location.reload();
      return;
    }
    statusNode.textContent = `Current as of ${new Date(version.generated_at).toLocaleString()}`;
  } catch {
    statusNode.textContent = "Could not check for a newer build";
  }
}

setInterval(checkForUpdate, intervalSeconds * 1000);
checkForUpdate();
"""


# =============================================================================
# Orchestration
# =============================================================================

def build(config_path: Path, root: Path, site_dir: Path, allow_yahoo: bool) -> None:
    site_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings.load(config_path)

    daily, market, source_label = load_data(
        settings,
        root,
        allow_yahoo=allow_yahoo,
    )
    daily, market = add_features(daily, market)
    events = merge_signals(daily, market, settings)
    current = build_current_context(daily, market)

    analogs, analog_method = find_analogs(events, current, settings)
    regime_baseline = find_regime_baseline(events, current, settings)
    verdict = score_analogs(
        analogs,
        regime_baseline,
        current,
        analog_method,
        settings,
    )
    guidance = build_position_guidance(verdict, settings)
    timing = score_fast_timing(daily, market, current, settings)
    study = build_event_study(events, settings)
    scorecard_rows = build_signal_scorecard(study, current)
    scorecard = format_scorecard(scorecard_rows)

    generated = datetime.now(timezone.utc)
    build_id = generated.strftime("%Y%m%dT%H%M%SZ")

    print("[history] building point-in-time historical decision replay...")
    history = replay_historical_decisions(
        settings,
        daily,
        market,
        events,
        progress_every=100,
    )
    changes = decision_change_rows(history)
    timing_changes = timing_change_rows(history)
    timing_evaluation = evaluate_timing_signals(timing_changes, market)

    history.to_csv(site_dir / "historical_decisions.csv", index=False)
    changes.to_csv(site_dir / "decision_changes.csv", index=False)
    timing_changes.to_csv(site_dir / "timing_decision_changes.csv", index=False)
    timing_evaluation.to_csv(site_dir / "timing_evaluation.csv", index=False)
    history_payload = {
        "generated_at": generated.isoformat(),
        "data_source": source_label,
        "method": "point_in_time_replay_of_current_dashboard_engine",
        "notes": [
            "5-day returns are exposed only after the fifth trading session is known.",
            "20-day returns are exposed only after the twentieth trading session is known.",
            "20-day drawdown is recomputed through each replay cutoff.",
            "WAIT ON BUYING remains the slow model's non-sell action.",
            "The fast timing layer can issue EARLY TRIM / CAUTION for tactical risk reduction.",
            "Future local lows/highs are used only by timing_evaluation.csv and never by signal generation.",
        ],
        "decisions": _history_records(history),
        "action_changes": _history_records(changes),
        "timing_changes": _history_records(timing_changes),
        "timing_evaluation": _history_records(timing_evaluation),
    }
    (site_dir / "historical_decisions.json").write_text(
        json.dumps(history_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (site_dir / "decision_history.html").write_text(
        render_history_page(
            history,
            changes,
            timing_changes,
            timing_evaluation,
            generated_at=generated,
            data_source=source_label,
        ),
        encoding="utf-8",
    )

    history_counts = history["action"].value_counts().to_dict() if not history.empty else {}
    timing_counts = (
        history["timing_action"].value_counts().to_dict()
        if not history.empty
        else {}
    )
    history_summary = {
        "total": int(len(history)),
        "changes": int(len(changes)),
        "timing_changes": int(len(timing_changes)),
        "buy": int(history_counts.get("BUY GRADUALLY", 0)),
        "wait": int(history_counts.get("WAIT ON BUYING", 0)),
        "hold": int(history_counts.get("HOLD / NO EXTRA BUYING", 0)),
        "insufficient": int(history_counts.get("INSUFFICIENT EVIDENCE", 0)),
        "early_buy": int(
            sum(
                count
                for action, count in timing_counts.items()
                if str(action).startswith("EARLY BUY")
            )
        ),
        "early_trim": int(
            sum(
                count
                for action, count in timing_counts.items()
                if str(action).startswith("EARLY TRIM")
            )
        ),
    }

    gaps = daily.index.to_series().diff().dt.days.dropna()
    large_gap_count = int((gaps > 7).sum())
    completed_5d = int(events["forward_5d"].notna().sum())

    warnings: list[str] = []
    if large_gap_count:
        warnings.append(
            f"Sentiment history has {large_gap_count} gap(s) longer than seven days."
        )
    if completed_5d < 100:
        warnings.append(
            f"Only {completed_5d} observations have completed five-day outcomes yet."
        )
    if verdict.sample_size < verdict.required_sample:
        warnings.append(
            "The tactical action is disabled because there are too few independent "
            "same-regime analogs."
        )
    if verdict.market_extension == "High":
        warnings.append(
            "The market is highly extended, so the model requires a larger sample, "
            "a stronger excess return, and a better drawdown profile before issuing "
            "BUY GRADUALLY."
        )

    latest_change_5 = current.get("fg_change_5")
    excess_interval = (
        f"{fmt_pct(verdict.excess_ci_low_5d, 2)} to "
        f"{fmt_pct(verdict.excess_ci_high_5d, 2)}"
    )

    metrics = [
        {
            "label": "Latest Fear & Greed",
            "value": fmt_num(float(current["fear_greed"])),
            "note": pd.Timestamp(current["signal_date"]).strftime("%Y-%m-%d"),
        },
        {
            "label": "Market regime",
            "value": pretty_regime(verdict.market_regime),
            "note": f"Extension: {verdict.market_extension}",
        },
        {
            "label": "Distance from 252D high",
            "value": fmt_pct(current.get("distance_from_252d_high"), 2),
            "note": "0% means at the rolling one-year high",
        },
        {
            "label": "20-day market return",
            "value": fmt_pct(current.get("market_return_20d"), 2),
            "note": "Recent price momentum",
        },
        {
            "label": "Distance from SMA 200",
            "value": fmt_pct(current.get("distance_from_sma_200"), 2),
            "note": "Long-term trend extension",
        },
        {
            "label": "5-observation FG change",
            "value": fmt_num(
                float(latest_change_5) if pd.notna(latest_change_5) else None
            ),
            "note": "Positive means improving sentiment",
        },
        {
            "label": "Win rate, 5 days out",
            "value": fmt_pct(verdict.win_rate_5d),
            "note": f"{verdict.sample_size} same-regime analogs",
        },
        {
            "label": "Wilson win-rate floor",
            "value": fmt_pct(verdict.wilson_floor_5d),
            "note": "Diagnostic only; not the main decision test",
        },
        {
            "label": "Average 5-day outcome",
            "value": fmt_pct(verdict.average_5d),
            "note": f"Same-regime baseline: {fmt_pct(verdict.regime_baseline_5d)}",
        },
        {
            "label": "5-day excess return",
            "value": fmt_pct(verdict.excess_5d, 2),
            "note": f"95% bootstrap interval: {excess_interval}",
        },
        {
            "label": "Median 5-day outcome",
            "value": fmt_pct(verdict.median_5d),
            "note": "Less sensitive to outliers",
        },
        {
            "label": "Average 20-day outcome",
            "value": fmt_pct(verdict.average_20d),
            "note": "Where enough future data exists",
        },
        {
            "label": "Worst 5-day analog",
            "value": fmt_pct(verdict.worst_5d),
            "note": "Historical downside example",
        },
        {
            "label": "Avg. worst drawdown, next 20D",
            "value": fmt_pct(verdict.average_drawdown_20d),
            "note": "Adverse move after the signal",
        },
    ]

    outcomes_stats = {
        "win_rate": fmt_pct(verdict.win_rate_5d),
        "median_5d": fmt_pct(verdict.median_5d),
        "worst_5d": fmt_pct(verdict.worst_5d),
        "sample_size": verdict.sample_size,
    }

    analog_columns = [
        "signal_date",
        "fear_greed",
        "fg_change_1",
        "fg_change_3",
        "fg_change_5",
        "market_regime",
        "distance_from_252d_high",
        "market_return_20d",
        "distance_from_sma_200",
        "volatility_20d",
        "market_date",
        "entry_date",
        "entry_price",
        *[f"forward_{horizon}d" for horizon in settings.horizons],
        "max_drawdown_20d",
        "analog_distance",
    ]
    analog_columns = [column for column in analog_columns if column in analogs.columns]
    analog_export = analogs[analog_columns].copy()

    analog_percent_columns = {
        "distance_from_252d_high",
        "market_return_20d",
        "distance_from_sma_200",
        "volatility_20d",
        "max_drawdown_20d",
        *{
            column
            for column in analog_export.columns
            if column.startswith("forward_")
        },
    }

    dashboard_html = PAGE_TEMPLATE.render(
        build_id=build_id,
        refresh_seconds=settings.refresh_seconds,
        source=source_label,
        generated=generated.strftime("%Y-%m-%d %H:%M UTC"),
        verdict=verdict,
        guidance=guidance,
        timing=timing,
        regime_label=pretty_regime(verdict.market_regime),
        warnings=warnings,
        metrics=metrics,
        chart=render_chart(daily, market, timing_changes),
        gauge=render_gauge(float(current["fear_greed"])),
        outcomes_chart=render_analog_outcomes_chart(analogs, regime_baseline),
        outcomes_stats=outcomes_stats,
        scorecard=scorecard,
        history_summary=history_summary,
        event_study_table=render_table(
            study,
            {
                "Win rate 5D",
                "Average 5D",
                "Median 5D",
                "Average 20D",
                "Worst 5D",
                "Avg worst drawdown 20D",
                "Excess vs baseline 5D",
            },
        ),
        analog_table=render_table(analog_export, analog_percent_columns),
    )

    (site_dir / "index.html").write_text(dashboard_html, encoding="utf-8")
    (site_dir / "styles.css").write_text(STYLES_CSS, encoding="utf-8")
    (site_dir / "app.js").write_text(APP_JS, encoding="utf-8")
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")
    study.to_csv(site_dir / "event_study.csv", index=False)
    analog_export.to_csv(site_dir / "analogs.csv", index=False)
    events.to_csv(site_dir / "full_analysis.csv", index=False)

    report = {
        "generated_at": generated.isoformat(),
        "build_id": build_id,
        "data_source": source_label,
        "coverage": {
            "first_date": daily.index.min().date().isoformat(),
            "last_date": daily.index.max().date().isoformat(),
            "daily_observations": int(len(daily)),
            "completed_5d_outcomes": completed_5d,
            "gaps_over_7_days": large_gap_count,
        },
        "latest": {
            "signal_date": pd.Timestamp(current["signal_date"]).date().isoformat(),
            "market_date": pd.Timestamp(current["market_date"]).date().isoformat(),
            "fear_greed": float(current["fear_greed"]),
            "fg_change_5": (
                None if pd.isna(latest_change_5) else float(latest_change_5)
            ),
            "market_regime": verdict.market_regime,
            "market_extension": verdict.market_extension,
            "distance_from_252d_high": current.get("distance_from_252d_high"),
            "market_return_20d": current.get("market_return_20d"),
            "distance_from_sma_200": current.get("distance_from_sma_200"),
            "volatility_20d": current.get("volatility_20d"),
        },
        "verdict": asdict(verdict),
        "position_guidance": asdict(guidance),
        "fast_timing": asdict(timing),
        "historical_decisions": history_summary,
        "warnings": warnings,
    }

    (site_dir / "analysis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (site_dir / "version.json").write_text(
        json.dumps(
            {
                "generated_at": generated.isoformat(),
                "build_id": build_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Data source : {source_label}")
    print(f"Coverage    : {daily.index.min().date()} through {daily.index.max().date()}")
    print(f"Observations: {len(daily)}")
    print(f"Regime      : {pretty_regime(verdict.market_regime)}")
    print(
        f"Action      : {verdict.action} "
        f"({verdict.confidence} confidence, n={verdict.sample_size})"
    )
    print(f"Timing      : {timing.action} — {timing.recommendation}")
    print(f"Sizing      : {guidance.tier} — {guidance.sizing_label}")
    print(
        f"History     : {len(history)} decisions, {len(timing_changes)} fast timing changes, "
        f"{len(changes)} slow-model changes"
    )
    print(f"Site        : {site_dir / 'index.html'}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.json (default: <root>/config.json)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (default: repo containing scripts/)",
    )
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/site)",
    )
    parser.add_argument(
        "--skip-yahoo-fallback",
        action="store_true",
        help=(
            "Fail instead of downloading from Yahoo Finance when repository "
            "market data are unusable"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    site_dir = args.site_dir or (args.root / "site")
    config_path = args.config or (args.root / "config.json")

    try:
        build(
            config_path,
            args.root,
            site_dir,
            allow_yahoo=not args.skip_yahoo_fallback,
        )
    except Exception as error:  # noqa: BLE001
        print(f"Dashboard build failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())