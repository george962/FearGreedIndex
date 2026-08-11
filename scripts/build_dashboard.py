#!/usr/bin/env python3
"""
scripts/build_dashboard.py

Build a static "Fear & Greed vs. S&P 500" research dashboard from the data
already stored in this repository.

The headline action is regime-aware. Historical analogs must resemble the
current Fear & Greed setup AND the current S&P 500 price/risk regime. The
signal is compared with a same-regime baseline rather than the unconditional
market average.

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
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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
    "market_return_5d",
    "market_return_20d",
    "distance_from_20d_high",
    "distance_from_252d_high",
    "distance_from_record_high",
    "distance_from_sma_50",
    "distance_from_sma_200",
    "volatility_20d",
    "market_regime",
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

    # Position-sizing tiers shown in the "What to do with this" panel.
    # These are illustrative defaults, not personalized financial advice —
    # tune them to whatever your own contribution plan looks like.
    sizing_strong_buy_pct: int = 150
    sizing_modest_buy_low_pct: int = 110
    sizing_modest_buy_high_pct: int = 125
    sizing_strong_buy_min_checks_ratio: float = 0.8

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
    market["market_return_5d"] = close.pct_change(5)
    market["market_return_20d"] = close.pct_change(20)

    high_20 = close.rolling(20, min_periods=5).max()
    high_252 = close.rolling(252, min_periods=60).max()
    record_high = close.expanding(min_periods=1).max()
    sma_50 = close.rolling(50, min_periods=20).mean()
    sma_200 = close.rolling(200, min_periods=60).mean()

    market["drawdown_from_20d_high"] = close / high_20 - 1
    market["distance_from_20d_high"] = close / high_20 - 1
    market["distance_from_252d_high"] = close / high_252 - 1
    market["distance_from_record_high"] = close / record_high - 1
    market["distance_from_sma_50"] = close / sma_50 - 1
    market["distance_from_sma_200"] = close / sma_200 - 1
    market["volatility_20d"] = daily_return.rolling(20, min_periods=10).std(ddof=1) * math.sqrt(252)

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

        # Historical outcomes require a next-session entry. The latest live
        # observation is handled separately by build_current_context().
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

    context: dict[str, Any] = {
        "signal_date": latest_date,
        "market_date": market_date,
        "fear_greed": float(sentiment_row["fear_greed"]),
        "fg_change_1": _plain_value(sentiment_row.get("fg_change_1")),
        "fg_change_3": _plain_value(sentiment_row.get("fg_change_3")),
        "fg_change_5": _plain_value(sentiment_row.get("fg_change_5")),
        "fg_change_10": _plain_value(sentiment_row.get("fg_change_10")),
        "market_close": _plain_value(market_row.get("close")),
    }

    for column in MARKET_CONTEXT_COLUMNS:
        context[column] = _plain_value(market_row.get(column))

    return pd.Series(context)


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

    generator = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=float)

    for index in range(iterations):
        analog_sample = generator.choice(analog, size=len(analog), replace=True)
        baseline_sample = generator.choice(baseline, size=len(baseline), replace=True)
        differences[index] = analog_sample.mean() - baseline_sample.mean()

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
# Position-sizing guidance ("what do I actually do with this")
# =============================================================================

@dataclass
class PositionGuidance:
    tier: str
    sizing_label: str
    sizing_detail: str
    guardrail: str


def build_position_guidance(verdict: Verdict, settings: Settings) -> PositionGuidance:
    """
    Translate the verdict into a concrete, rule-based sizing suggestion.

    This does not invent new evidence — it only restates the same checks
    already computed in score_analogs() as a tiered action. It is a research
    heuristic you can tune (see the sizing_* fields in Settings), not
    personalized financial advice.
    """
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
    """
    Narrow the full event study down to the handful of backtested signals
    that are actually "live" given today's Fear & Greed level and recent
    move, so the right-hand panel shows what's relevant today instead of
    every threshold ever tested.
    """
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
    """Turn scorecard rows into small, template-friendly cards with a tone."""
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


def render_chart(daily: pd.DataFrame, market: pd.DataFrame) -> str:
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

    figure.update_layout(
        template="plotly_dark",
        height=430,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font={"color": CHART_TEXT, "size": 12},
        margin={"l": 50, "r": 20, "t": 10, "b": 34},
        hovermode="x unified",
        showlegend=False,
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
    """
    Show how the current analogs actually played out over the next 5
    sessions, next to the same-regime baseline, as a strip/box plot instead
    of a raw table of numbers.
    """
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
            <div>
              <strong>{{ check.label }}</strong>
              <small>{{ check.value }} · {{ check.requirement }}</small>
            </div>
          </div>
          {% endfor %}
        </div>
      </div>

      <dl class="metric-list">
        {% for metric in metrics[1:] %}
        <div class="metric-row">
          <dt>{{ metric.label }}</dt>
          <dd>{{ metric.value }}<small>{{ metric.note }}</small></dd>
        </div>
        {% endfor %}
      </dl>

      <div class="sidebar-foot">
        <span>Last built</span>
        <strong>{{ generated }}</strong>
        <span id="refresh-status">Checking for updates…</span>
        <span class="source">{{ source }}</span>
      </div>
    </aside>

    <main class="workspace">
      {% if warnings %}
      <section class="warnings">
        {% for warning in warnings %}
        <div class="warning">{{ warning }}</div>
        {% endfor %}
      </section>
      {% endif %}

      <section class="panel chart-panel">
        <div class="panel-heading">
          <div>
            <p class="eyebrow">MARKET CONTEXT</p>
            <h2>Price vs. sentiment</h2>
          </div>
          <span class="hint">The action compares sentiment only with historical observations in the same price regime.</span>
        </div>
        {{ chart | safe }}
      </section>

      <section class="panel action-panel action-{{ verdict.tone }}">
        <div class="panel-heading">
          <div>
            <p class="eyebrow">WHAT TO DO WITH THIS</p>
            <h2>{{ guidance.tier }}</h2>
          </div>
          <span class="hint">Rule-based sizing derived from the checks on the left. Not personalized financial advice.</span>
        </div>
        <div class="action-sizing">
          <span class="action-sizing-label">{{ guidance.sizing_label }}</span>
        </div>
        <p class="action-detail">{{ guidance.sizing_detail }}</p>
        <p class="action-guardrail">{{ guidance.guardrail }}</p>
      </section>

      <div class="table-grid">
        <section class="panel table-panel outcomes-panel">
          <div class="panel-heading">
            <div>
              <p class="eyebrow">ANALOG OUTCOMES</p>
              <h2>How similar setups actually played out</h2>
            </div>
            <a href="analogs.csv" download>CSV</a>
          </div>
          {{ outcomes_chart | safe }}
          <div class="outcomes-stats">
            <div><dt>Win rate</dt><dd>{{ outcomes_stats.win_rate }}</dd></div>
            <div><dt>Median 5D</dt><dd>{{ outcomes_stats.median_5d }}</dd></div>
            <div><dt>Worst 5D</dt><dd>{{ outcomes_stats.worst_5d }}</dd></div>
            <div><dt>Analogs</dt><dd>{{ outcomes_stats.sample_size }}</dd></div>
          </div>
          <details>
            <summary>Show full analog table</summary>
            <div class="table-wrap">{{ analog_table | safe }}</div>
          </details>
        </section>

        <section class="panel table-panel scorecard-panel">
          <div class="panel-heading">
            <div>
              <p class="eyebrow">SIGNAL SCORECARD</p>
              <h2>Backtests active right now</h2>
            </div>
            <a href="event_study.csv" download>CSV</a>
          </div>
          {% if scorecard %}
          <div class="scorecard-list">
            {% for item in scorecard %}
            <div class="scorecard-row scorecard-{{ item.tone }}">
              <strong>{{ item.label }}</strong>
              <div class="scorecard-stats">
                <span>{{ item.events }} events</span>
                <span>Win {{ item.win_rate }}</span>
                <span>Avg {{ item.average }}</span>
                <span>Edge {{ item.excess }}</span>
              </div>
            </div>
            {% endfor %}
          </div>
          {% else %}
          <p class="empty-note">No backtested threshold or drop signal currently matches today's setup closely enough to score. See the full table below.</p>
          {% endif %}
          <details>
            <summary>Show full backtest table</summary>
            <div class="table-wrap">{{ event_study_table | safe }}</div>
          </details>
        </section>
      </div>

      <section class="panel note-panel">
        <p class="eyebrow">HOW TO READ THIS</p>
        <p>
          BUY GRADUALLY requires a positive excess return over the same-regime
          baseline, a bootstrap confidence interval above zero, and supporting
          return/drawdown evidence. HOLD / NO EXTRA BUYING means the model does
          not support an above-normal tactical purchase. It does not tell you to
          stop a separate long-term contribution plan. WAIT ON BUYING is not
          a sell signal. The sizing suggestion above restates these same checks
          as a tiered rule of thumb — adjust the sizing_* values in config.json
          to match your own plan. Historical performance does not guarantee
          future results.
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
.sidebar {
  background: var(--sidebar); border-right: 1px solid var(--border); padding: 22px 18px;
  overflow-y: auto; display: flex; flex-direction: column; gap: 16px;
}
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
.action-sizing-label {
  display: inline-block; font-size: 1.1rem; font-weight: 800; padding: 7px 16px;
  border-radius: 999px; background: rgba(90,169,255,.14); color: var(--accent);
}
.action-detail { font-size: .85rem; color: var(--muted); line-height: 1.5; margin-bottom: 8px; }
.action-guardrail { font-size: .76rem; color: var(--faint); font-style: italic; }

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
    study = build_event_study(events, settings)
    scorecard_rows = build_signal_scorecard(study, current)
    scorecard = format_scorecard(scorecard_rows)

    generated = datetime.now(timezone.utc)
    build_id = generated.strftime("%Y%m%dT%H%M%SZ")

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

    html = PAGE_TEMPLATE.render(
        build_id=build_id,
        refresh_seconds=settings.refresh_seconds,
        source=source_label,
        generated=generated.strftime("%Y-%m-%d %H:%M UTC"),
        verdict=verdict,
        guidance=guidance,
        regime_label=pretty_regime(verdict.market_regime),
        warnings=warnings,
        metrics=metrics,
        chart=render_chart(daily, market),
        gauge=render_gauge(float(current["fear_greed"])),
        outcomes_chart=render_analog_outcomes_chart(analogs, regime_baseline),
        outcomes_stats=outcomes_stats,
        scorecard=scorecard,
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

    (site_dir / "index.html").write_text(html, encoding="utf-8")
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
    print(
        f"Coverage    : {daily.index.min().date()} through {daily.index.max().date()}"
    )
    print(f"Observations: {len(daily)}")
    print(f"Regime      : {pretty_regime(verdict.market_regime)}")
    print(
        f"Action      : {verdict.action} "
        f"({verdict.confidence} confidence, n={verdict.sample_size})"
    )
    print(f"Sizing      : {guidance.tier} — {guidance.sizing_label}")
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