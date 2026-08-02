"""Build the static Fear & Greed dashboard from repository history.

Preferred input:
    data/fear_greed_spx_daily.csv

Fallback inputs:
    data/fear_greed_daily.csv
    data/spx_daily.csv

The parser intentionally accepts many common column-name variants.
"""

from __future__ import annotations

import io
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from jinja2 import Template


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


@dataclass
class ActionSummary:
    action: str
    tone: str
    confidence: str
    sample_size: int
    probability_positive_5d: float | None
    average_return_5d: float | None
    median_return_5d: float | None
    average_return_20d: float | None
    baseline_average_5d: float | None
    excess_average_5d: float | None
    worst_return_5d: float | None
    average_max_drawdown_20d: float | None
    analog_method: str
    rationale: str


def normalize_column(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def read_csv_flexible(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    errors: list[str] = []

    for separator in (None, ",", "\t", ";"):
        try:
            frame = pd.read_csv(
                io.StringIO(raw),
                sep=separator,
                engine="python",
            )
            if frame.shape[1] >= 2:
                frame.columns = [normalize_column(column) for column in frame.columns]
                return frame
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    raise ValueError(f"Could not parse {path}: {' | '.join(errors[-2:])}")


def find_column(
    frame: pd.DataFrame,
    exact_names: set[str],
    contains_any: tuple[str, ...] = (),
    numeric_range: tuple[float, float] | None = None,
    excluded: set[str] | None = None,
) -> str | None:
    excluded = excluded or set()

    for column in frame.columns:
        if column in excluded:
            continue
        if column in exact_names:
            return column

    for column in frame.columns:
        if column in excluded:
            continue
        if contains_any and any(token in column for token in contains_any):
            return column

    if numeric_range is not None:
        low, high = numeric_range
        best_column: str | None = None
        best_score = 0.0
        for column in frame.columns:
            if column in excluded:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty:
                continue
            score = float(values.between(low, high).mean())
            if score > best_score:
                best_column = column
                best_score = score
        if best_score >= 0.75:
            return best_column

    return None


def detect_date_column(frame: pd.DataFrame) -> str:
    date_names = {
        "date",
        "day",
        "datetime",
        "timestamp",
        "market_date",
        "trade_date",
    }
    for column in frame.columns:
        if column in date_names or "date" in column:
            converted = pd.to_datetime(frame[column], errors="coerce")
            if converted.notna().mean() >= 0.6:
                return column

    best_column: str | None = None
    best_score = 0.0
    for column in frame.columns:
        converted = pd.to_datetime(frame[column], errors="coerce")
        score = float(converted.notna().mean())
        if score > best_score:
            best_column = column
            best_score = score

    if best_column is None or best_score < 0.6:
        raise ValueError("Could not identify a date column.")
    return best_column


def parse_combined_dataset(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = read_csv_flexible(path)
    date_column = detect_date_column(frame)

    fear_column = find_column(
        frame,
        {
            "fear_greed",
            "fear_and_greed",
            "fear_greed_index",
            "feargreed",
            "feargreedindex",
            "value",
            "score",
            "index_value",
        },
        contains_any=("fear", "greed"),
        numeric_range=(0, 100),
        excluded={date_column},
    )

    market_column = find_column(
        frame,
        {
            "spx_close",
            "sp500_close",
            "s_p_500_close",
            "sp_500_close",
            "close",
            "adj_close",
            "adjusted_close",
            "market_close",
        },
        contains_any=("spx", "sp500", "s_p_500", "close"),
        excluded={date_column, fear_column} if fear_column else {date_column},
    )

    if fear_column is None:
        raise ValueError(f"Could not identify Fear & Greed column in {path}.")
    if market_column is None:
        raise ValueError(f"Could not identify S&P 500 close column in {path}.")

    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    fear = pd.to_numeric(frame[fear_column], errors="coerce")
    market_close = pd.to_numeric(frame[market_column], errors="coerce")

    combined = pd.DataFrame(
        {
            "date": dates,
            "fear_greed": fear,
            "close": market_close,
        }
    ).dropna(subset=["date", "fear_greed", "close"])

    combined = combined[combined["fear_greed"].between(0, 100)]
    combined = combined.sort_values("date").drop_duplicates("date", keep="last")

    daily = combined.set_index("date")[["fear_greed"]]
    market = combined.set_index("date")[["close"]]
    market.index.name = "market_date"
    market["open"] = market["close"]
    market["high"] = market["close"]
    market["low"] = market["close"]
    return daily, market


def parse_fear_dataset(path: Path) -> pd.DataFrame:
    frame = read_csv_flexible(path)
    date_column = detect_date_column(frame)

    fear_column = find_column(
        frame,
        {
            "fear_greed",
            "fear_and_greed",
            "fear_greed_index",
            "feargreed",
            "feargreedindex",
            "value",
            "score",
            "index_value",
        },
        contains_any=("fear", "greed"),
        numeric_range=(0, 100),
        excluded={date_column},
    )
    if fear_column is None:
        raise ValueError(f"Could not identify Fear & Greed column in {path}.")

    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_column], errors="coerce").dt.normalize(),
            "fear_greed": pd.to_numeric(frame[fear_column], errors="coerce"),
        }
    ).dropna(subset=["date", "fear_greed"])

    daily = daily[daily["fear_greed"].between(0, 100)]
    daily = daily.sort_values("date").drop_duplicates("date", keep="last")
    return daily.set_index("date")[["fear_greed"]]


def parse_market_dataset(path: Path) -> pd.DataFrame:
    frame = read_csv_flexible(path)
    date_column = detect_date_column(frame)

    close_column = find_column(
        frame,
        {
            "spx_close",
            "sp500_close",
            "s_p_500_close",
            "sp_500_close",
            "close",
            "adj_close",
            "adjusted_close",
            "market_close",
        },
        contains_any=("close", "spx", "sp500"),
        excluded={date_column},
    )
    if close_column is None:
        raise ValueError(f"Could not identify close column in {path}.")

    market = pd.DataFrame(
        {
            "market_date": pd.to_datetime(
                frame[date_column],
                errors="coerce",
            ).dt.normalize(),
            "close": pd.to_numeric(frame[close_column], errors="coerce"),
        }
    ).dropna(subset=["market_date", "close"])

    market = market.sort_values("market_date").drop_duplicates(
        "market_date",
        keep="last",
    )
    market = market.set_index("market_date")
    market["open"] = market["close"]
    market["high"] = market["close"]
    market["low"] = market["close"]
    return market[["open", "high", "low", "close"]]


def download_market(start: pd.Timestamp) -> pd.DataFrame:
    ticker = CONFIG["fallback_ticker"]
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
        raise RuntimeError(f"No fallback market data returned for {ticker}.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [
            "_".join(str(part) for part in column if str(part))
            for column in raw.columns
        ]

    def locate(prefix: str) -> str | None:
        for column in raw.columns:
            normalized = normalize_column(column)
            if normalized == prefix or normalized.startswith(prefix + "_"):
                return str(column)
        return None

    close = locate("close")
    if close is None:
        raise RuntimeError("Fallback market data has no close column.")

    market = pd.DataFrame(
        index=pd.to_datetime(raw.index).tz_localize(None).normalize()
    )
    for field in ("open", "high", "low", "close"):
        source = locate(field)
        market[field] = (
            pd.to_numeric(raw[source], errors="coerce")
            if source is not None
            else pd.to_numeric(raw[close], errors="coerce")
        )

    market = market.dropna(subset=["close"]).sort_index()
    market.index.name = "market_date"
    return market


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    combined_path = ROOT / CONFIG["combined_dataset"]
    fear_path = ROOT / CONFIG["fear_greed_dataset"]
    market_path = ROOT / CONFIG["market_dataset"]

    combined_error: Exception | None = None

    if combined_path.exists():
        try:
            daily, market = parse_combined_dataset(combined_path)
            return daily, market, str(combined_path.relative_to(ROOT))
        except Exception as exc:  # noqa: BLE001
            combined_error = exc
            print(f"Combined dataset could not be used: {exc}")

    if not fear_path.exists():
        raise FileNotFoundError(
            f"Missing Fear & Greed history: {fear_path.relative_to(ROOT)}"
        )

    daily = parse_fear_dataset(fear_path)

    if market_path.exists():
        try:
            market = parse_market_dataset(market_path)
            return (
                daily,
                market,
                f"{fear_path.relative_to(ROOT)} + {market_path.relative_to(ROOT)}",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Repository market dataset could not be used: {exc}")

    market = download_market(daily.index.min())
    source = f"{fear_path.relative_to(ROOT)} + Yahoo Finance fallback"

    if combined_error:
        source += " (combined parser fallback used)"
    return daily, market, source


def add_features(
    daily: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = daily.sort_index().copy()
    market = market.sort_index().copy()

    for window in (1, 3, 5, 10):
        daily[f"fg_change_{window}d"] = daily["fear_greed"].diff(window)

    market["return_1d"] = market["close"].pct_change()
    market["return_5d"] = market["close"].pct_change(5)
    market["return_20d"] = market["close"].pct_change(20)
    market["high_20d"] = market["close"].rolling(20, min_periods=1).max()
    market["drawdown_from_20d_high"] = market["close"] / market["high_20d"] - 1
    return daily, market


def merge_signals(daily: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    signal = daily.reset_index().rename(columns={"date": "signal_date"})
    sessions = market.reset_index()

    merged = pd.merge_asof(
        signal.sort_values("signal_date"),
        sessions.sort_values("market_date"),
        left_on="signal_date",
        right_on="market_date",
        direction="forward",
        allow_exact_matches=False,
    ).dropna(subset=["market_date", "close"])

    merged = merged.sort_values("market_date").reset_index(drop=True)
    merged["entry_price"] = merged["open"].where(
        merged["open"].notna(),
        merged["close"],
    )

    positions = {
        pd.Timestamp(index): position
        for position, index in enumerate(market.index)
    }

    for horizon in CONFIG["horizons"]:
        outcomes: list[float] = []
        for row in merged.itertuples():
            position = positions.get(pd.Timestamp(row.market_date))
            target = None if position is None else position + horizon - 1
            if position is None or target is None or target >= len(market):
                outcomes.append(np.nan)
            else:
                outcomes.append(
                    float(market["close"].iloc[target] / row.entry_price - 1)
                )
        merged[f"forward_{horizon}d"] = outcomes

    drawdowns: list[float] = []
    for row in merged.itertuples():
        position = positions.get(pd.Timestamp(row.market_date))
        if position is None:
            drawdowns.append(np.nan)
            continue
        end = min(position + 19, len(market) - 1)
        lows = market["low"].iloc[position : end + 1].dropna()
        drawdowns.append(
            np.nan if lows.empty else float(lows.min() / row.entry_price - 1)
        )
    merged["max_drawdown_20d"] = drawdowns
    return merged


def cooldown(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    candidates = frame.loc[mask.fillna(False)].sort_values("signal_date")
    days = int(CONFIG["cooldown_calendar_days"])
    if candidates.empty or days <= 0:
        return candidates

    selected: list[int] = []
    previous: pd.Timestamp | None = None
    for index, row in candidates.iterrows():
        current = pd.Timestamp(row["signal_date"])
        if previous is None or (current - previous).days >= days:
            selected.append(index)
            previous = current
    return frame.loc[selected].sort_values("signal_date")


def mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def median(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else float(values.median())


def positive_rate(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else float((values > 0).mean())


def wilson_low(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return math.nan
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
        / denominator
    )
    return max(0.0, center - margin)


def find_analogs(
    merged: pd.DataFrame,
    latest: pd.Series,
) -> tuple[pd.DataFrame, str]:
    complete = merged[merged["forward_5d"].notna()].copy()
    minimum = int(CONFIG["minimum_action_sample"])
    level_band = float(CONFIG["analog_level_band"])
    change_band = float(CONFIG["analog_change_band"])
    current_level = float(latest["fear_greed"])
    current_change = latest.get("fg_change_5d", np.nan)

    level_mask = complete["fear_greed"].between(
        current_level - level_band,
        current_level + level_band,
    )

    if pd.notna(current_change):
        change_mask = complete["fg_change_5d"].between(
            float(current_change) - change_band,
            float(current_change) + change_band,
        )
        strict = complete[level_mask & change_mask]
        if len(strict) >= minimum:
            return strict, (
                f"Fear & Greed ±{level_band:g} and "
                f"five-observation change ±{change_band:g}"
            )

    level_only = complete[level_mask]
    if len(level_only) >= minimum:
        return level_only, f"Fear & Greed level ±{level_band:g}"

    level_scale = max(float(complete["fear_greed"].std(ddof=0)), 1.0)
    distance = ((complete["fear_greed"] - current_level) / level_scale).abs()

    if pd.notna(current_change) and complete["fg_change_5d"].notna().any():
        change_scale = max(float(complete["fg_change_5d"].std(ddof=0)), 1.0)
        distance += (
            (complete["fg_change_5d"] - float(current_change)) / change_scale
        ).abs().fillna(1.0)

    nearest = complete.assign(_distance=distance).nsmallest(
        min(max(minimum, 20), len(complete)),
        "_distance",
    )
    return nearest.drop(columns="_distance"), "nearest historical observations"


def action_summary(
    analogs: pd.DataFrame,
    baseline: pd.DataFrame,
    method: str,
) -> ActionSummary:
    five = analogs["forward_5d"].dropna()
    twenty = analogs["forward_20d"].dropna()
    sample = len(five)
    probability = positive_rate(five)
    average_5d = mean(five)
    baseline_5d = mean(baseline["forward_5d"])
    excess = (
        None
        if average_5d is None or baseline_5d is None
        else average_5d - baseline_5d
    )

    confidence_low = (
        wilson_low(int((five > 0).sum()), sample)
        if sample
        else math.nan
    )
    minimum = int(CONFIG["minimum_action_sample"])

    if sample < minimum:
        action = "INSUFFICIENT EVIDENCE"
        tone = "neutral"
        confidence = "Very low"
        rationale = (
            f"Only {sample} completed analogs were available. "
            "No directional timing action is justified."
        )
    elif (
        probability is not None
        and probability >= 0.65
        and confidence_low >= 0.45
        and average_5d is not None
        and average_5d > 0
        and excess is not None
        and excess > 0
    ):
        action = "BUY GRADUALLY"
        tone = "positive"
        confidence = "Moderate" if sample >= 30 else "Low"
        rationale = (
            "Similar historical observations had favorable five-day odds and "
            "beat the unconditional five-day average. Use staged buying rather "
            "than treating this as an exact bottom."
        )
    elif (
        probability is not None
        and probability <= 0.45
        and average_5d is not None
        and average_5d < 0
    ):
        action = "WAIT ON EXTRA BUYING"
        tone = "negative"
        confidence = "Moderate" if sample >= 30 else "Low"
        rationale = (
            "Similar historical observations were more often followed by "
            "additional short-term weakness than gains."
        )
    else:
        action = "NEUTRAL"
        tone = "mixed"
        confidence = "Low"
        rationale = (
            "Historical outcomes were mixed or not strong enough to support a "
            "directional timing decision."
        )

    return ActionSummary(
        action=action,
        tone=tone,
        confidence=confidence,
        sample_size=sample,
        probability_positive_5d=probability,
        average_return_5d=average_5d,
        median_return_5d=median(five),
        average_return_20d=mean(twenty),
        baseline_average_5d=baseline_5d,
        excess_average_5d=excess,
        worst_return_5d=None if five.empty else float(five.min()),
        average_max_drawdown_20d=mean(analogs["max_drawdown_20d"]),
        analog_method=method,
        rationale=rationale,
    )


def event_study(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline = mean(merged["forward_5d"])

    def summarize(label: str, events: pd.DataFrame) -> None:
        five = events["forward_5d"].dropna()
        twenty = events["forward_20d"].dropna()
        event_average = mean(five)
        rows.append(
            {
                "Signal": label,
                "Events": len(five),
                "Positive after 5D": positive_rate(five),
                "Average 5D": event_average,
                "Median 5D": median(five),
                "Average 20D": mean(twenty),
                "Worst 5D": None if five.empty else float(five.min()),
                "Average max drawdown 20D": mean(events["max_drawdown_20d"]),
                "Excess vs baseline 5D": (
                    None
                    if event_average is None or baseline is None
                    else event_average - baseline
                ),
            }
        )

    for threshold in CONFIG["level_thresholds"]:
        summarize(
            f"Fear & Greed ≤ {threshold}",
            cooldown(merged, merged["fear_greed"] <= threshold),
        )

    for window in CONFIG["drop_windows"]:
        column = f"fg_change_{window}d"
        for threshold in CONFIG["drop_thresholds"]:
            summarize(
                f"{window}-observation drop ≥ {threshold}",
                cooldown(merged, merged[column] <= -threshold),
            )

    return pd.DataFrame(rows)


def pct(value: float | None, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def num(value: float | None, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def chart_html(daily: pd.DataFrame, market: pd.DataFrame) -> str:
    chart_market = market.loc[market.index >= daily.index.min()]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=chart_market.index,
            y=chart_market["close"],
            name="S&P 500 close",
            line={"width": 2},
            yaxis="y",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["fear_greed"],
            name="Fear & Greed",
            mode="lines+markers",
            marker={"size": 4},
            line={"width": 2},
            yaxis="y2",
        )
    )
    figure.add_hrect(
        y0=0,
        y1=25,
        yref="y2",
        opacity=0.08,
        line_width=0,
    )
    figure.update_layout(
        template="plotly_white",
        height=520,
        margin={"l": 45, "r": 55, "t": 25, "b": 40},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08},
        yaxis={"title": "S&P 500"},
        yaxis2={
            "title": "Fear & Greed",
            "overlaying": "y",
            "side": "right",
            "range": [0, 100],
            "showgrid": False,
        },
        xaxis={"title": "Date"},
    )
    return figure.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displaylogo": False},
    )


def dataframe_html(
    frame: pd.DataFrame,
    percentage_columns: set[str] | None = None,
) -> str:
    percentage_columns = percentage_columns or set()
    output = frame.copy()

    for column in output.columns:
        if column in percentage_columns:
            output[column] = output[column].map(
                lambda item: pct(item, 2) if pd.notna(item) else "—"
            )
        elif pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(output[column]):
            output[column] = output[column].map(
                lambda item: f"{item:.2f}" if pd.notna(item) else "—"
            )

    return output.to_html(
        index=False,
        classes="data-table",
        border=0,
        escape=True,
    )


PAGE = Template(r"""<!doctype html>
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
      <p class="subtitle">Using repository history from {{ source }}.</p>
    </div>
    <div class="updated">
      <span>Last built</span>
      <strong>{{ generated }}</strong>
      <span id="refresh-status">Checking for updates</span>
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
    <section class="action-panel action-{{ summary.tone }}">
      <div>
        <p class="eyebrow">CURRENT RESEARCH ACTION</p>
        <h2>{{ summary.action }}</h2>
        <p>{{ summary.rationale }}</p>
      </div>
      <dl>
        <div><dt>Confidence</dt><dd>{{ summary.confidence }}</dd></div>
        <div><dt>Historical analogs</dt><dd>{{ summary.sample_size }}</dd></div>
        <div><dt>Analog method</dt><dd>{{ summary.analog_method }}</dd></div>
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
      <div class="table-wrap">{{ event_table | safe }}</div>
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
      <h2>How to interpret the result</h2>
      <p>
        The label is a mechanical summary of historical analogs, not a forecast
        guarantee. Fear &amp; Greed may react to the same price movement it is
        being used to predict. Long gaps, repeated observations from one market
        episode, and changes in market regime can bias the results.
      </p>
    </section>
  </main>

  <footer>
    <span>Historical performance does not guarantee future results.</span>
    <span>The published GitHub Pages site is public.</span>
  </footer>

  <script src="app.js"></script>
</body>
</html>
""")


CSS = r"""
:root {
  color-scheme: light dark;
  --bg: #f4f6f8;
  --panel: #ffffff;
  --text: #111827;
  --muted: #5f6b7a;
  --border: #d8dee8;
  --accent: #275dad;
  --positive-bg: #eaf7f0;
  --positive: #137a46;
  --negative-bg: #fbecec;
  --negative: #a62929;
  --mixed-bg: #fff6dc;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); }
.site-header, main, .warnings, footer {
  width: min(1180px, calc(100% - 32px));
  margin-inline: auto;
}
.site-header {
  display: flex; justify-content: space-between; gap: 28px;
  align-items: end; padding: 38px 0 22px;
}
h1 { margin: 0 0 8px; font-size: clamp(2rem, 4vw, 3.4rem); }
h2, p { margin-top: 0; }
.subtitle, small, .updated { color: var(--muted); }
.eyebrow {
  margin-bottom: 7px; color: var(--accent); font-size: .75rem;
  font-weight: 800; letter-spacing: .12em;
}
.updated { display: grid; gap: 4px; text-align: right; font-size: .84rem; }
.updated strong { color: var(--text); }
.warnings { display: grid; gap: 8px; margin-bottom: 18px; }
.warning {
  padding: 12px 14px; border: 1px solid #e7c66d;
  border-radius: 10px; background: #fff8df; color: #644b00;
}
main { display: grid; gap: 20px; padding-bottom: 36px; }
.panel, .action-panel, .metric {
  border: 1px solid var(--border); border-radius: 16px; background: var(--panel);
}
.panel { padding: 22px; }
.action-panel {
  display: grid; grid-template-columns: 1.5fr 1fr;
  gap: 24px; padding: 26px; border-width: 2px;
}
.action-positive { border-color: var(--positive); background: var(--positive-bg); }
.action-negative { border-color: var(--negative); background: var(--negative-bg); }
.action-mixed { border-color: #d2a72d; background: var(--mixed-bg); }
.action-neutral { background: #edf1f5; }
.action-panel dl {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 10px; margin: 0;
}
.action-panel dl div {
  padding: 12px; border: 1px solid rgba(100,116,139,.25);
  border-radius: 12px; background: rgba(255,255,255,.45);
}
dt { color: var(--muted); font-size: .78rem; }
dd { margin: 6px 0 0; font-weight: 800; }
.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.metric { padding: 18px; }
.metric span, .metric small { display: block; color: var(--muted); }
.metric strong { display: block; margin: 8px 0 4px; font-size: 1.7rem; }
.panel-heading {
  display: flex; align-items: center; justify-content: space-between; gap: 18px;
}
.panel-heading a { color: var(--accent); font-weight: 700; text-decoration: none; }
.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 12px; }
.data-table {
  width: 100%; border-collapse: collapse; font-size: .9rem; white-space: nowrap;
}
.data-table th, .data-table td {
  padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: right;
}
.data-table th:first-child, .data-table td:first-child { text-align: left; }
.data-table th { background: rgba(100,116,139,.08); }
footer {
  display: flex; justify-content: space-between; gap: 20px;
  padding: 22px 0 34px; color: var(--muted); font-size: .82rem;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0c1118; --panel: #121a24; --text: #edf2f7;
    --muted: #a1acba; --border: #2a3645; --accent: #83b4ff;
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
@media (max-width: 520px) {
  .metrics { grid-template-columns: 1fr; }
}
"""


JS = r"""
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

    statusNode.textContent =
      `Current as of ${new Date(version.generated_at).toLocaleString()}`;
  } catch {
    statusNode.textContent = "Could not check for a newer build";
  }
}

setInterval(checkForUpdate, intervalSeconds * 1000);
checkForUpdate();
"""


def main() -> None:
    SITE.mkdir(parents=True, exist_ok=True)

    daily, market, source = load_data()
    daily, market = add_features(daily, market)
    merged = merge_signals(daily, market)

    if merged.empty:
        raise RuntimeError("No sentiment dates could be matched to later market sessions.")

    latest = daily.iloc[-1]
    analogs, analog_method = find_analogs(merged, latest)
    summary = action_summary(analogs, merged, analog_method)
    study = event_study(merged)

    generated = datetime.now(timezone.utc)
    build_id = generated.strftime("%Y%m%dT%H%M%SZ")
    gaps = daily.index.to_series().diff().dt.days.dropna()
    large_gaps = int((gaps > 7).sum())
    completed_5d = int(merged["forward_5d"].notna().sum())

    warnings: list[str] = []
    if large_gaps:
        warnings.append(
            f"Sentiment history contains {large_gaps} gap(s) longer than seven days."
        )
    if completed_5d < 100:
        warnings.append(
            f"Only {completed_5d} observations have completed five-day outcomes."
        )
    if summary.sample_size < int(CONFIG["minimum_action_sample"]):
        warnings.append("The action is disabled because the analog sample is too small.")

    metrics = [
        {
            "label": "Latest Fear & Greed",
            "value": num(float(latest["fear_greed"])),
            "note": daily.index.max().strftime("%Y-%m-%d"),
        },
        {
            "label": "5-observation change",
            "value": num(
                float(latest["fg_change_5d"])
                if pd.notna(latest["fg_change_5d"])
                else None
            ),
            "note": "Negative means worsening sentiment",
        },
        {
            "label": "Positive after 5 days",
            "value": pct(summary.probability_positive_5d),
            "note": f"{summary.sample_size} historical analogs",
        },
        {
            "label": "Average 5-day outcome",
            "value": pct(summary.average_return_5d),
            "note": f"Excess vs baseline: {pct(summary.excess_average_5d)}",
        },
        {
            "label": "Median 5-day outcome",
            "value": pct(summary.median_return_5d),
            "note": "Less affected by outliers",
        },
        {
            "label": "Average 20-day outcome",
            "value": pct(summary.average_return_20d),
            "note": "Where enough future data exists",
        },
        {
            "label": "Worst 5-day analog",
            "value": pct(summary.worst_return_5d),
            "note": "Historical downside example",
        },
        {
            "label": "Average worst drawdown, next 20D",
            "value": pct(summary.average_max_drawdown_20d),
            "note": "Adverse move after the signal",
        },
    ]

    analog_columns = [
        "signal_date",
        "fear_greed",
        "fg_change_1d",
        "fg_change_3d",
        "fg_change_5d",
        "market_date",
        "entry_price",
        "forward_1d",
        "forward_5d",
        "forward_10d",
        "forward_20d",
        "max_drawdown_20d",
    ]
    analog_export = analogs[analog_columns].copy()

    html = PAGE.render(
        build_id=build_id,
        refresh_seconds=int(CONFIG["refresh_seconds"]),
        source=source,
        generated=generated.strftime("%Y-%m-%d %H:%M UTC"),
        summary=summary,
        warnings=warnings,
        metrics=metrics,
        chart=chart_html(daily, market),
        event_table=dataframe_html(
            study,
            {
                "Positive after 5D",
                "Average 5D",
                "Median 5D",
                "Average 20D",
                "Worst 5D",
                "Average max drawdown 20D",
                "Excess vs baseline 5D",
            },
        ),
        analog_table=dataframe_html(
            analog_export,
            {
                "forward_1d",
                "forward_5d",
                "forward_10d",
                "forward_20d",
                "max_drawdown_20d",
            },
        ),
    )

    (SITE / "index.html").write_text(html, encoding="utf-8")
    (SITE / "styles.css").write_text(CSS, encoding="utf-8")
    (SITE / "app.js").write_text(JS, encoding="utf-8")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")
    study.to_csv(SITE / "event_study.csv", index=False)
    analog_export.to_csv(SITE / "analogs.csv", index=False)
    merged.to_csv(SITE / "full_analysis.csv", index=False)

    report = {
        "generated_at": generated.isoformat(),
        "build_id": build_id,
        "data_source": source,
        "coverage": {
            "first_fear_greed_date": daily.index.min().date().isoformat(),
            "last_fear_greed_date": daily.index.max().date().isoformat(),
            "daily_observations": len(daily),
            "completed_5d_outcomes": completed_5d,
            "gaps_longer_than_7_days": large_gaps,
        },
        "latest": {
            "date": daily.index.max().date().isoformat(),
            "fear_greed": float(latest["fear_greed"]),
            "five_observation_change": (
                None if pd.isna(latest["fg_change_5d"])
                else float(latest["fg_change_5d"])
            ),
        },
        "summary": asdict(summary),
        "warnings": warnings,
    }

    (SITE / "analysis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (SITE / "version.json").write_text(
        json.dumps(
            {
                "generated_at": generated.isoformat(),
                "build_id": build_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Data source: {source}")
    print(f"Coverage: {daily.index.min().date()} through {daily.index.max().date()}")
    print(f"Daily observations: {len(daily)}")
    print(f"Action: {summary.action}")
    print(f"Built: {SITE / 'index.html'}")


if __name__ == "__main__":
    main()