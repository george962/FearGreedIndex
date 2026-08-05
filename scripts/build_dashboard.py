#!/usr/bin/env python3
"""
build_dashboard.py

Builds a static "Fear & Greed vs. S&P 500" research dashboard from the
data already living in this repository.

Expected inputs (paths come from config.json, all relative to the repo root):
    combined_dataset  data/fear_greed_spx_daily.csv   (fear+market already joined)
    fear_greed_dataset data/fear_greed_daily.csv       (Fear & Greed only)
    market_dataset     data/spx_daily.csv              (S&P 500 OHLC only)

Resolution order:
    1. Separate fear_greed_dataset + market_dataset, if both exist and parse.
    2. combined_dataset, if it exists and parses.
    3. fear_greed_dataset alone, with market data pulled from Yahoo Finance
       via yfinance as a last resort (only triggers if 1 and 2 fail).

Output (written to --site-dir, default "site/"):
    index.html          the dashboard page
    styles.css, app.js  static assets (auto-refresh polling included)
    event_study.csv      threshold / drawdown backtest table
    analogs.csv           the historical analogs used for today's call
    full_analysis.csv     every matched observation, for your own digging
    analysis.json          machine-readable summary of the whole run
    version.json            build id + timestamp, polled by app.js

Usage:
    python build_dashboard.py
    python build_dashboard.py --config config.json --site-dir site
    python build_dashboard.py --skip-yahoo-fallback   # fail loudly instead
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

try:
    import yfinance as yf
except ImportError:
    yf = None


ROOT = Path(__file__).resolve().parent


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
    minimum_action_sample: int = 8
    refresh_seconds: int = 300
    horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 20, 60])
    level_thresholds: list[int] = field(
        default_factory=lambda: [15, 20, 25, 30, 35, 40, 50]
    )
    drop_windows: list[int] = field(default_factory=lambda: [1, 3, 5])
    drop_thresholds: list[int] = field(default_factory=lambda: [5, 10, 15, 20])
    analog_level_band: float = 7
    analog_change_band: float = 5

    @classmethod
    def load(cls, path: Path) -> "Settings":
        if not path.exists():
            print(f"No config found at {path}; using built-in defaults.")
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        known_fields = {f for f in cls.__dataclass_fields__}
        filtered = {key: value for key, value in raw.items() if key in known_fields}
        return cls(**filtered)


# =============================================================================
# CSV parsing helpers
# =============================================================================

def slugify(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def sniff_read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV/TSV whose delimiter and column casing we can't fully trust."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    last_error: Optional[Exception] = None
    for sep in (None, ",", "\t", ";"):
        try:
            frame = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
        if frame.shape[1] >= 2:
            frame.columns = [slugify(c) for c in frame.columns]
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
    """Best-effort column resolution: exact name, then fuzzy, then by shape."""
    for column in frame.columns:
        if column not in skip and column in exact:
            return column
    for column in frame.columns:
        if column not in skip and contains and any(token in column for token in contains):
            return column
    if numeric_between is not None:
        lo, hi = numeric_between
        best, best_hit_rate = None, 0.0
        for column in frame.columns:
            if column in skip:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty:
                continue
            hit_rate = float(values.between(lo, hi).mean())
            if hit_rate > best_hit_rate:
                best, best_hit_rate = column, hit_rate
        if best_hit_rate >= 0.75:
            return best
    return None


def pick_date_column(frame: pd.DataFrame) -> str:
    likely = {"date", "day", "datetime", "timestamp", "market_date", "trade_date"}
    for column in frame.columns:
        if column in likely or "date" in column:
            if pd.to_datetime(frame[column], errors="coerce").notna().mean() >= 0.6:
                return column
    best, best_hit_rate = None, 0.0
    for column in frame.columns:
        hit_rate = float(pd.to_datetime(frame[column], errors="coerce").notna().mean())
        if hit_rate > best_hit_rate:
            best, best_hit_rate = column, hit_rate
    if best is None or best_hit_rate < 0.6:
        raise ValueError("No column in this file looks like a date.")
    return best


# =============================================================================
# Loading the two series
# =============================================================================

class DataSource:
    """Resolves Fear & Greed + S&P 500 history using the configured fallback chain."""

    def __init__(self, settings: Settings, root: Path, allow_yahoo: bool = True):
        self.settings = settings
        self.root = root
        self.allow_yahoo = allow_yahoo

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        fear_path = self.root / self.settings.fear_greed_dataset
        market_path = self.root / self.settings.market_dataset
        combined_path = self.root / self.settings.combined_dataset

        if fear_path.exists() and market_path.exists():
            try:
                sentiment = self._parse_sentiment(fear_path)
                market = self._parse_market(market_path)
                return (
                    sentiment,
                    market,
                    f"{fear_path.relative_to(self.root)} + {market_path.relative_to(self.root)}",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[data] separate datasets unusable ({exc}); trying combined file.")

        if combined_path.exists():
            try:
                sentiment, market = self._parse_combined(combined_path)
                return sentiment, market, str(combined_path.relative_to(self.root))
            except Exception as exc:  # noqa: BLE001
                print(f"[data] combined dataset unusable ({exc}); trying Yahoo fallback.")

        if not fear_path.exists():
            raise FileNotFoundError(
                f"No Fear & Greed history found at {fear_path.relative_to(self.root)}."
            )

        sentiment = self._parse_sentiment(fear_path)
        if not self.allow_yahoo:
            raise RuntimeError(
                "No usable market dataset and Yahoo fallback is disabled "
                "(--skip-yahoo-fallback)."
            )
        market = self._download_market(sentiment.index.min())
        return sentiment, market, f"{fear_path.relative_to(self.root)} + Yahoo Finance"

    def _parse_sentiment(self, path: Path) -> pd.DataFrame:
        frame = sniff_read_csv(path)
        date_col = pick_date_column(frame)
        value_col = pick_column(
            frame,
            exact={
                "fear_greed", "fear_and_greed", "fear_greed_index",
                "feargreed", "value", "score", "index_value", "rating_value",
            },
            contains=("fear", "greed"),
            numeric_between=(0, 100),
            skip={date_col},
        )
        if value_col is None:
            raise ValueError("no Fear & Greed value column found")

        out = pd.DataFrame({
            "date": pd.to_datetime(frame[date_col], errors="coerce").dt.normalize(),
            "fear_greed": pd.to_numeric(frame[value_col], errors="coerce"),
        }).dropna()
        out = out[out["fear_greed"].between(0, 100)]
        out = out.sort_values("date").drop_duplicates("date", keep="last")
        return out.set_index("date")

    def _parse_market(self, path: Path) -> pd.DataFrame:
        frame = sniff_read_csv(path)
        date_col = pick_date_column(frame)
        close_col = pick_column(
            frame,
            exact={
                "spx_close", "sp500_close", "close", "adj_close",
                "adjusted_close", "market_close",
            },
            contains=("close",),
            skip={date_col},
        )
        if close_col is None:
            raise ValueError("no close-price column found")

        def numeric(name: str) -> pd.Series:
            source = name if name in frame.columns else close_col
            return pd.to_numeric(frame[source], errors="coerce")

        out = pd.DataFrame({
            "date": pd.to_datetime(frame[date_col], errors="coerce").dt.normalize(),
            "open": numeric("open"),
            "high": numeric("high"),
            "low": numeric("low"),
            "close": pd.to_numeric(frame[close_col], errors="coerce"),
        }).dropna(subset=["date", "close"])
        out = out.sort_values("date").drop_duplicates("date", keep="last")
        return out.set_index("date")

    def _parse_combined(self, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        frame = sniff_read_csv(path)
        required = {
            "fear_greed_date_utc", "fear_greed_value", "spx_date",
            "spx_open", "spx_high", "spx_low", "spx_close",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")

        sentiment = pd.DataFrame({
            "date": pd.to_datetime(frame["fear_greed_date_utc"], errors="coerce").dt.normalize(),
            "fear_greed": pd.to_numeric(frame["fear_greed_value"], errors="coerce"),
        }).dropna()
        sentiment = sentiment[sentiment["fear_greed"].between(0, 100)]
        sentiment = sentiment.sort_values("date").drop_duplicates("date", keep="last")
        sentiment = sentiment.set_index("date")

        market = pd.DataFrame({
            "date": pd.to_datetime(frame["spx_date"], errors="coerce").dt.normalize(),
            "open": pd.to_numeric(frame["spx_open"], errors="coerce"),
            "high": pd.to_numeric(frame["spx_high"], errors="coerce"),
            "low": pd.to_numeric(frame["spx_low"], errors="coerce"),
            "close": pd.to_numeric(frame["spx_close"], errors="coerce"),
        }).dropna(subset=["date", "close"])
        market = market.sort_values("date").drop_duplicates("date", keep="last")
        market = market.set_index("date")

        return sentiment, market

    def _download_market(self, start: pd.Timestamp) -> pd.DataFrame:
        if yf is None:
            raise RuntimeError("yfinance is not installed and no market CSV is usable.")
        ticker = self.settings.fallback_ticker
        raw = yf.download(
            ticker,
            start=(start - pd.Timedelta(days=90)).strftime("%Y-%m-%d"),
            end=(pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            actions=False,
            threads=False,
        )
        if raw.empty:
            raise RuntimeError(f"Yahoo Finance returned no data for {ticker}.")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = ["_".join(str(p) for p in col if str(p)) for col in raw.columns]

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
                raw[source] if source is not None else raw[close_source], errors="coerce"
            )
        market = market.dropna(subset=["close"]).sort_index()
        market.index.name = "date"
        return market


# =============================================================================
# Feature engineering + signal/market merge
# =============================================================================

def add_sentiment_features(sentiment: pd.DataFrame) -> pd.DataFrame:
    sentiment = sentiment.sort_index().copy()
    for window in (1, 3, 5, 10):
        sentiment[f"change_{window}"] = sentiment["fear_greed"].diff(window)
    return sentiment


def add_market_features(market: pd.DataFrame) -> pd.DataFrame:
    market = market.sort_index().copy()
    market["return_1d"] = market["close"].pct_change()
    rolling_high = market["close"].rolling(20, min_periods=1).max()
    market["drawdown_from_20d_high"] = market["close"] / rolling_high - 1
    return market


def build_event_table(
    sentiment: pd.DataFrame, market: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    """
    Pair every sentiment reading with the *next* trading session's open (to
    avoid look-ahead bias), then compute forward returns for each configured
    horizon plus a rolling 20-session worst drawdown from entry.
    """
    market = market[~market.index.duplicated(keep="last")].sort_index()
    position_of = {ts: i for i, ts in enumerate(market.index)}
    closes = market["close"].to_numpy()
    lows = (market["low"] if "low" in market.columns else market["close"]).to_numpy()

    rows: list[dict[str, Any]] = []
    for signal_date, sentiment_row in sentiment.iterrows():
        # Find the most recent trading session at/before the signal date.
        as_of = market.index[market.index <= signal_date]
        if as_of.empty:
            continue
        signal_position = position_of[as_of[-1]]
        entry_position = signal_position + 1
        if entry_position >= len(market):
            continue

        entry_row = market.iloc[entry_position]
        entry_price = float(entry_row["open"]) if pd.notna(entry_row.get("open")) else float(entry_row["close"])
        entry_date = market.index[entry_position]

        record: dict[str, Any] = {
            "signal_date": signal_date,
            "fear_greed": float(sentiment_row["fear_greed"]),
            "change_1": sentiment_row.get("change_1"),
            "change_3": sentiment_row.get("change_3"),
            "change_5": sentiment_row.get("change_5"),
            "entry_date": entry_date,
            "entry_price": entry_price,
        }

        for horizon in settings.horizons:
            target_position = entry_position + horizon - 1
            if target_position < len(market) and np.isfinite(entry_price):
                record[f"forward_{horizon}d"] = float(closes[target_position] / entry_price - 1)
            else:
                record[f"forward_{horizon}d"] = np.nan

        window_end = min(entry_position + 19, len(market) - 1)
        window_lows = lows[entry_position:window_end + 1]
        window_lows = window_lows[~np.isnan(window_lows)]
        record["max_drawdown_20d"] = (
            float(window_lows.min() / entry_price - 1) if window_lows.size else np.nan
        )

        rows.append(record)

    events = pd.DataFrame(rows)
    if events.empty:
        raise RuntimeError(
            "No sentiment reading had a following trading session to enter on. "
            "Check that the market dataset extends at least one session past "
            "the Fear & Greed history."
        )
    return events.sort_values("entry_date").reset_index(drop=True)


# =============================================================================
# Analogs, event study, and the headline call
# =============================================================================

def apply_cooldown(events: pd.DataFrame, mask: pd.Series, cooldown_days: int) -> pd.DataFrame:
    """Greedily keep events at least `cooldown_days` apart so one sentiment
    episode doesn't dominate a statistic through repeated near-daily hits."""
    candidates = events.loc[mask.fillna(False)].sort_values("signal_date")
    if candidates.empty or cooldown_days <= 0:
        return candidates
    keep: list[int] = []
    last_kept: Optional[pd.Timestamp] = None
    for idx, row in candidates.iterrows():
        current = pd.Timestamp(row["signal_date"])
        if last_kept is None or (current - last_kept).days >= cooldown_days:
            keep.append(idx)
            last_kept = current
    return events.loc[keep]


def _safe_mean(series: pd.Series) -> Optional[float]:
    values = series.dropna()
    return float(values.mean()) if len(values) else None


def _safe_median(series: pd.Series) -> Optional[float]:
    values = series.dropna()
    return float(values.median()) if len(values) else None


def _win_rate(series: pd.Series) -> Optional[float]:
    values = series.dropna()
    return float((values > 0).mean()) if len(values) else None


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return math.nan
    p = successes / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * total)) / total) / denom
    return max(0.0, center - margin)


def find_analogs(events: pd.DataFrame, latest: pd.Series, settings: Settings) -> tuple[pd.DataFrame, str]:
    complete = events[events["forward_5d"].notna()].copy()
    current_level = float(latest["fear_greed"])
    current_change = latest.get("change_5", np.nan)

    level_mask = complete["fear_greed"].between(
        current_level - settings.analog_level_band, current_level + settings.analog_level_band
    )

    if pd.notna(current_change):
        change_mask = complete["change_5"].between(
            float(current_change) - settings.analog_change_band,
            float(current_change) + settings.analog_change_band,
        )
        tight = apply_cooldown(complete, level_mask & change_mask, settings.cooldown_calendar_days)
        if len(tight) >= settings.minimum_action_sample:
            return tight, f"level ±{settings.analog_level_band:g} and 5-obs change ±{settings.analog_change_band:g}"

    loose = apply_cooldown(complete, level_mask, settings.cooldown_calendar_days)
    if len(loose) >= settings.minimum_action_sample:
        return loose, f"level ±{settings.analog_level_band:g} only"

    # Nothing in-band: fall back to a nearest-neighbor search by z-scored distance.
    level_scale = max(float(complete["fear_greed"].std(ddof=0)), 1.0)
    distance = ((complete["fear_greed"] - current_level) / level_scale).abs()
    if pd.notna(current_change) and complete["change_5"].notna().any():
        change_scale = max(float(complete["change_5"].std(ddof=0)), 1.0)
        distance = distance + ((complete["change_5"] - float(current_change)) / change_scale).abs().fillna(1.0)

    ranked = complete.assign(_distance=distance).sort_values("_distance")
    chosen_idx: list[int] = []
    chosen_dates: list[pd.Timestamp] = []
    for idx, row in ranked.iterrows():
        date = pd.Timestamp(row["signal_date"])
        if all(abs((date - d).days) >= settings.cooldown_calendar_days for d in chosen_dates):
            chosen_idx.append(idx)
            chosen_dates.append(date)
        if len(chosen_idx) >= max(settings.minimum_action_sample, 20):
            break

    return complete.loc[chosen_idx], "nearest independent historical observations"


@dataclass
class Verdict:
    action: str
    tone: str
    confidence: str
    sample_size: int
    win_rate_5d: Optional[float]
    average_5d: Optional[float]
    median_5d: Optional[float]
    average_20d: Optional[float]
    baseline_5d: Optional[float]
    excess_5d: Optional[float]
    worst_5d: Optional[float]
    average_drawdown_20d: Optional[float]
    analog_method: str
    rationale: str


def score_analogs(analogs: pd.DataFrame, all_events: pd.DataFrame, method: str, settings: Settings) -> Verdict:
    five = analogs["forward_5d"].dropna()
    twenty = analogs["forward_20d"].dropna()
    sample = len(five)

    win_rate = _win_rate(five)
    average_5d = _safe_mean(five)
    baseline_5d = _safe_mean(all_events["forward_5d"])
    excess = None if average_5d is None or baseline_5d is None else average_5d - baseline_5d
    confidence_floor = wilson_lower_bound(int((five > 0).sum()), sample) if sample else math.nan

    if sample < settings.minimum_action_sample:
        action, tone, confidence = "INSUFFICIENT EVIDENCE", "neutral", "Very low"
        rationale = f"Only {sample} completed analogs — not enough to act on."
    elif (
        win_rate is not None and win_rate >= 0.65
        and confidence_floor >= 0.45
        and average_5d is not None and average_5d > 0
        and excess is not None and excess > 0
    ):
        action, tone = "BUY GRADUALLY", "positive"
        confidence = "Moderate" if sample >= 30 else "Low"
        rationale = (
            "Similar historical setups skewed positive over the next five "
            "sessions and beat the unconditional baseline. Treat as a case "
            "for staged buying, not a precise bottom call."
        )
    elif win_rate is not None and win_rate <= 0.45 and average_5d is not None and average_5d < 0:
        action, tone = "WAIT ON EXTRA BUYING", "negative"
        confidence = "Moderate" if sample >= 30 else "Low"
        rationale = "Similar setups were more often followed by further short-term weakness."
    else:
        action, tone, confidence = "NEUTRAL", "mixed", "Low"
        rationale = "Historical outcomes were mixed — not enough edge to justify a directional call."

    return Verdict(
        action=action, tone=tone, confidence=confidence, sample_size=sample,
        win_rate_5d=win_rate, average_5d=average_5d, median_5d=_safe_median(five),
        average_20d=_safe_mean(twenty), baseline_5d=baseline_5d, excess_5d=excess,
        worst_5d=None if five.empty else float(five.min()),
        average_drawdown_20d=_safe_mean(analogs["max_drawdown_20d"]),
        analog_method=method, rationale=rationale,
    )


def build_event_study(events: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    baseline = _safe_mean(events["forward_5d"])
    rows: list[dict[str, Any]] = []

    def summarize(label: str, subset: pd.DataFrame) -> None:
        five = subset["forward_5d"].dropna()
        average = _safe_mean(five)
        rows.append({
            "Signal": label,
            "Events": len(five),
            "Win rate 5D": _win_rate(five),
            "Average 5D": average,
            "Median 5D": _safe_median(five),
            "Average 20D": _safe_mean(subset["forward_20d"].dropna()),
            "Worst 5D": None if five.empty else float(five.min()),
            "Avg worst drawdown 20D": _safe_mean(subset["max_drawdown_20d"]),
            "Excess vs baseline 5D": None if average is None or baseline is None else average - baseline,
        })

    for threshold in settings.level_thresholds:
        summarize(f"Fear & Greed ≤ {threshold}", apply_cooldown(
            events, events["fear_greed"] <= threshold, settings.cooldown_calendar_days))

    for window in settings.drop_windows:
        column = f"change_{window}"
        if column not in events.columns:
            continue
        for threshold in settings.drop_thresholds:
            summarize(f"{window}-obs drop ≥ {threshold}", apply_cooldown(
                events, events[column] <= -threshold, settings.cooldown_calendar_days))

    return pd.DataFrame(rows)


# =============================================================================
# Rendering
# =============================================================================

def fmt_pct(value: Optional[float], digits: int = 1) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{value * 100:.{digits}f}%"


def fmt_num(value: Optional[float], digits: int = 1) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def render_chart(sentiment: pd.DataFrame, market: pd.DataFrame) -> str:
    visible_market = market.loc[market.index >= sentiment.index.min()]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=visible_market.index, y=visible_market["close"],
        name="S&P 500 close", line={"width": 2}, yaxis="y",
    ))
    figure.add_trace(go.Scatter(
        x=sentiment.index, y=sentiment["fear_greed"],
        name="Fear & Greed", mode="lines+markers", marker={"size": 4},
        line={"width": 2}, yaxis="y2",
    ))
    figure.add_hrect(y0=0, y1=25, yref="y2", opacity=0.08, line_width=0)
    figure.add_hrect(y0=75, y1=100, yref="y2", opacity=0.08, line_width=0)
    figure.update_layout(
        template="plotly_white", height=520,
        margin={"l": 45, "r": 55, "t": 25, "b": 40},
        hovermode="x unified", legend={"orientation": "h", "y": 1.08},
        yaxis={"title": "S&P 500"},
        yaxis2={"title": "Fear & Greed", "overlaying": "y", "side": "right",
                "range": [0, 100], "showgrid": False},
        xaxis={"title": "Date"},
    )
    return figure.to_html(full_html=False, include_plotlyjs="cdn",
                           config={"responsive": True, "displaylogo": False})


def render_table(frame: pd.DataFrame, percent_columns: set[str] = frozenset()) -> str:
    display = frame.copy()
    for column in display.columns:
        if column in percent_columns:
            display[column] = display[column].map(lambda v: fmt_pct(v, 2) if pd.notna(v) else "—")
        elif pd.api.types.is_datetime64_any_dtype(display[column]):
            display[column] = display[column].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    return display.to_html(index=False, classes="data-table", border=0, escape=True)


PAGE_TEMPLATE = Template(r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="robots" content="noindex,nofollow">
  <title>Fear & Greed Market Dashboard</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body data-build-id="{{ build_id }}" data-refresh-seconds="{{ refresh_seconds }}">
  <header class="site-header">
    <div>
      <p class="eyebrow">RESEARCH DASHBOARD</p>
      <h1>Fear &amp; Greed vs. S&amp;P 500</h1>
      <p class="subtitle">Built from {{ source }}.</p>
    </div>
    <div class="updated">
      <span>Last built</span>
      <strong>{{ generated }}</strong>
      <span id="refresh-status">Checking for updates…</span>
    </div>
  </header>

  {% if warnings %}
  <section class="warnings">
    {% for warning in warnings %}
    <div class="warning">{{ warning }}</div>
    {% endfor %}
  </section>
  {% endif %}

  <main>
    <section class="action-panel action-{{ verdict.tone }}">
      <div>
        <p class="eyebrow">CURRENT RESEARCH ACTION</p>
        <h2>{{ verdict.action }}</h2>
        <p>{{ verdict.rationale }}</p>
      </div>
      <dl>
        <div><dt>Confidence</dt><dd>{{ verdict.confidence }}</dd></div>
        <div><dt>Historical analogs</dt><dd>{{ verdict.sample_size }}</dd></div>
        <div><dt>Analog method</dt><dd>{{ verdict.analog_method }}</dd></div>
      </dl>
    </section>

    <section class="metrics">
      {% for metric in metrics %}
      <article class="metric">
        <span>{{ metric.label }}</span>
        <strong>{{ metric.value }}</strong>
        <small>{{ metric.note }}</small>
      </article>
      {% endfor %}
    </section>

    <section class="panel">
      <p class="eyebrow">MARKET CONTEXT</p>
      <h2>Price and sentiment history</h2>
      {{ chart | safe }}
    </section>

    <section class="panel">
      <div class="panel-heading">
        <div>
          <p class="eyebrow">EVENT STUDY</p>
          <h2>Threshold and sudden-drop backtests</h2>
        </div>
        <a href="event_study.csv">Download CSV</a>
      </div>
      <div class="table-wrap">{{ event_study_table | safe }}</div>
    </section>

    <section class="panel">
      <div class="panel-heading">
        <div>
          <p class="eyebrow">CURRENT ANALOGS</p>
          <h2>Historically similar observations</h2>
        </div>
        <a href="analogs.csv">Download CSV</a>
      </div>
      <div class="table-wrap">{{ analog_table | safe }}</div>
    </section>

    <section class="panel">
      <p class="eyebrow">LIMITATIONS</p>
      <h2>How to read this</h2>
      <p>
        The action label is a mechanical summary of past analogs, not a
        forecast guarantee. Fear &amp; Greed can react to the same price move
        it's being used to predict, and a small analog sample, long data
        gaps, or a regime change can all bias the numbers above.
      </p>
    </section>
  </main>

  <footer>
    <span>Historical performance does not guarantee future results.</span>
    <span>This page is public.</span>
  </footer>

  <script src="app.js"></script>
</body>
</html>
""")


STYLES_CSS = r"""
:root {
  color-scheme: light dark;
  --bg: #f4f6f8; --panel: #ffffff; --text: #111827; --muted: #5f6b7a;
  --border: #d8dee8; --accent: #275dad;
  --positive-bg: #eaf7f0; --positive: #137a46;
  --negative-bg: #fbecec; --negative: #a62929;
  --mixed-bg: #fff6dc;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); }
.site-header, main, .warnings, footer { width: min(1180px, calc(100% - 32px)); margin-inline: auto; }
.site-header { display: flex; justify-content: space-between; gap: 28px; align-items: end; padding: 38px 0 22px; }
h1 { margin: 0 0 8px; font-size: clamp(2rem, 4vw, 3.4rem); }
h2, p { margin-top: 0; }
.subtitle, small, .updated { color: var(--muted); }
.eyebrow { margin-bottom: 7px; color: var(--accent); font-size: .75rem; font-weight: 800; letter-spacing: .12em; }
.updated { display: grid; gap: 4px; text-align: right; font-size: .84rem; }
.updated strong { color: var(--text); }
.warnings { display: grid; gap: 8px; margin-bottom: 18px; }
.warning { padding: 12px 14px; border: 1px solid #e7c66d; border-radius: 10px; background: #fff8df; color: #644b00; }
main { display: grid; gap: 20px; padding-bottom: 36px; }
.panel, .action-panel, .metric { border: 1px solid var(--border); border-radius: 16px; background: var(--panel); }
.panel { padding: 22px; }
.action-panel { display: grid; grid-template-columns: 1.5fr 1fr; gap: 24px; padding: 26px; border-width: 2px; }
.action-positive { border-color: var(--positive); background: var(--positive-bg); }
.action-negative { border-color: var(--negative); background: var(--negative-bg); }
.action-mixed { border-color: #d2a72d; background: var(--mixed-bg); }
.action-neutral { background: #edf1f5; }
.action-panel dl { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 0; }
.action-panel dl div { padding: 12px; border: 1px solid rgba(100,116,139,.25); border-radius: 12px; background: rgba(255,255,255,.45); }
dt { color: var(--muted); font-size: .78rem; }
dd { margin: 6px 0 0; font-weight: 800; }
.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.metric { padding: 18px; }
.metric span, .metric small { display: block; color: var(--muted); }
.metric strong { display: block; margin: 8px 0 4px; font-size: 1.7rem; }
.panel-heading { display: flex; align-items: center; justify-content: space-between; gap: 18px; }
.panel-heading a { color: var(--accent); font-weight: 700; text-decoration: none; }
.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 12px; }
.data-table { width: 100%; border-collapse: collapse; font-size: .9rem; white-space: nowrap; }
.data-table th, .data-table td { padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: right; }
.data-table th:first-child, .data-table td:first-child { text-align: left; }
.data-table th { background: rgba(100,116,139,.08); }
footer { display: flex; justify-content: space-between; gap: 20px; padding: 22px 0 34px; color: var(--muted); font-size: .82rem; }
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0c1118; --panel: #121a24; --text: #edf2f7; --muted: #a1acba;
    --border: #2a3645; --accent: #83b4ff;
    --positive-bg: #102a20; --negative-bg: #301616; --mixed-bg: #2e260f;
  }
  .warning { background: #2e260f; color: #f5d77d; border-color: #65551f; }
  .action-neutral { background: #17212d; }
  .action-panel dl div { background: rgba(0,0,0,.12); }
}
@media (max-width: 850px) {
  .site-header { display: grid; align-items: start; }
  .updated { text-align: left; }
  .action-panel { grid-template-columns: 1fr; }
  .action-panel dl { grid-template-columns: 1fr; }
  .metrics { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 520px) { .metrics { grid-template-columns: 1fr; } }
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

    source = DataSource(settings, root, allow_yahoo=allow_yahoo)
    sentiment, market, source_label = source.load()

    sentiment = add_sentiment_features(sentiment)
    market = add_market_features(market)
    events = build_event_table(sentiment, market, settings)

    latest = sentiment.iloc[-1]
    analogs, analog_method = find_analogs(events, latest, settings)
    verdict = score_analogs(analogs, events, analog_method, settings)
    study = build_event_study(events, settings)

    generated = datetime.now(timezone.utc)
    build_id = generated.strftime("%Y%m%dT%H%M%SZ")

    gaps = sentiment.index.to_series().diff().dt.days.dropna()
    large_gap_count = int((gaps > 7).sum())
    completed_5d = int(events["forward_5d"].notna().sum())

    warnings: list[str] = []
    if large_gap_count:
        warnings.append(f"Sentiment history has {large_gap_count} gap(s) longer than seven days.")
    if completed_5d < 100:
        warnings.append(f"Only {completed_5d} observations have completed five-day outcomes yet.")
    if verdict.sample_size < settings.minimum_action_sample:
        warnings.append("The action call is disabled — too few analogs in this dataset.")

    latest_change_5 = latest.get("change_5")
    metrics = [
        {"label": "Latest Fear & Greed", "value": fmt_num(float(latest["fear_greed"])),
         "note": sentiment.index.max().strftime("%Y-%m-%d")},
        {"label": "5-observation change",
         "value": fmt_num(float(latest_change_5) if pd.notna(latest_change_5) else None),
         "note": "Negative means worsening sentiment"},
        {"label": "Win rate, 5 days out", "value": fmt_pct(verdict.win_rate_5d),
         "note": f"{verdict.sample_size} historical analogs"},
        {"label": "Average 5-day outcome", "value": fmt_pct(verdict.average_5d),
         "note": f"Excess vs. baseline: {fmt_pct(verdict.excess_5d)}"},
        {"label": "Median 5-day outcome", "value": fmt_pct(verdict.median_5d),
         "note": "Less sensitive to outliers"},
        {"label": "Average 20-day outcome", "value": fmt_pct(verdict.average_20d),
         "note": "Where enough future data exists"},
        {"label": "Worst 5-day analog", "value": fmt_pct(verdict.worst_5d),
         "note": "Historical downside example"},
        {"label": "Avg. worst drawdown, next 20D", "value": fmt_pct(verdict.average_drawdown_20d),
         "note": "Adverse move after the signal"},
    ]

    analog_columns = [
        "signal_date", "fear_greed", "change_1", "change_3", "change_5",
        "entry_date", "entry_price",
        *[f"forward_{h}d" for h in settings.horizons],
        "max_drawdown_20d",
    ]
    analog_columns = [c for c in analog_columns if c in analogs.columns]
    analog_export = analogs[analog_columns].copy()

    html = PAGE_TEMPLATE.render(
        build_id=build_id,
        refresh_seconds=settings.refresh_seconds,
        source=source_label,
        generated=generated.strftime("%Y-%m-%d %H:%M UTC"),
        verdict=verdict,
        warnings=warnings,
        metrics=metrics,
        chart=render_chart(sentiment, market),
        event_study_table=render_table(study, {
            "Win rate 5D", "Average 5D", "Median 5D", "Average 20D",
            "Worst 5D", "Avg worst drawdown 20D", "Excess vs baseline 5D",
        }),
        analog_table=render_table(analog_export, {
            c for c in analog_export.columns if c.startswith("forward_") or c == "max_drawdown_20d"
        }),
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
            "first_date": sentiment.index.min().date().isoformat(),
            "last_date": sentiment.index.max().date().isoformat(),
            "daily_observations": int(len(sentiment)),
            "completed_5d_outcomes": completed_5d,
            "gaps_over_7_days": large_gap_count,
        },
        "latest": {
            "date": sentiment.index.max().date().isoformat(),
            "fear_greed": float(latest["fear_greed"]),
            "change_5": None if pd.isna(latest_change_5) else float(latest_change_5),
        },
        "verdict": asdict(verdict),
        "warnings": warnings,
    }
    (site_dir / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (site_dir / "version.json").write_text(
        json.dumps({"generated_at": generated.isoformat(), "build_id": build_id}, indent=2),
        encoding="utf-8",
    )

    print(f"Data source : {source_label}")
    print(f"Coverage    : {sentiment.index.min().date()} through {sentiment.index.max().date()}")
    print(f"Observations: {len(sentiment)}")
    print(f"Action      : {verdict.action}  ({verdict.confidence} confidence, n={verdict.sample_size})")
    print(f"Site        : {site_dir / 'index.html'}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None,
                         help="Path to config.json (default: <root>/config.json)")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root (default: script location)")
    parser.add_argument("--site-dir", type=Path, default=None, help="Output directory (default: <root>/site)")
    parser.add_argument("--skip-yahoo-fallback", action="store_true",
                         help="Fail instead of downloading from Yahoo Finance when repo data is unusable")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    site_dir = args.site_dir or (args.root / "site")
    config_path = args.config or (args.root / "config.json")
    try:
        build(config_path, args.root, site_dir, allow_yahoo=not args.skip_yahoo_fallback)
    except Exception as exc:  # noqa: BLE001
        print(f"Dashboard build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())