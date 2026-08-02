"""Build a static GitHub Pages dashboard from Fear & Greed and market data."""

from __future__ import annotations

import html
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


def read_table(path: Path) -> pd.DataFrame:
    content = path.read_text(encoding="utf-8-sig", errors="replace")
    parsed: pd.DataFrame | None = None

    for separator in (None, "\t", ",", ";"):
        try:
            candidate = pd.read_csv(
                io.StringIO(content),
                sep=separator,
                engine="python",
            )
            if candidate.shape[1] >= 2:
                parsed = candidate
                break
        except Exception:  # noqa: BLE001
            pass

    if parsed is None:
        raise ValueError(f"Could not parse {path}.")

    parsed.columns = [normalize_column(column) for column in parsed.columns]

    aliases = {
        "date": {"date", "day"},
        "time": {"time"},
        "value": {
            "value",
            "fear_greed",
            "fear_and_greed",
            "fear_greed_index",
            "score",
            "index",
        },
    }

    rename: dict[str, str] = {}
    for target, candidates in aliases.items():
        for column in parsed.columns:
            if column in candidates:
                rename[column] = target
                break

    parsed = parsed.rename(columns=rename)
    if "date" not in parsed.columns or "value" not in parsed.columns:
        raise ValueError("Fear & Greed file requires Date and Value columns.")

    if "time" not in parsed.columns:
        parsed["time"] = "16:00:00"

    parsed["date"] = pd.to_datetime(parsed["date"], errors="coerce").dt.normalize()
    parsed["value"] = pd.to_numeric(parsed["value"], errors="coerce")
    parsed["timestamp"] = pd.to_datetime(
        parsed["date"].dt.strftime("%Y-%m-%d")
        + " "
        + parsed["time"].astype(str),
        errors="coerce",
    )
    parsed = parsed.dropna(subset=["date", "timestamp", "value"])
    parsed = parsed[parsed["value"].between(0, 100)]
    return parsed.sort_values(["date", "timestamp"])


def daily_fear_greed(raw: pd.DataFrame) -> pd.DataFrame:
    method = CONFIG["daily_aggregation"]

    if method == "minimum":
        indexes = raw.groupby("date")["value"].idxmin()
        daily = raw.loc[indexes]
    elif method == "average":
        daily = raw.groupby("date", as_index=False).agg(
            timestamp=("timestamp", "max"),
            value=("value", "mean"),
        )
    else:
        daily = raw.groupby("date", as_index=False).tail(1)

    daily = (
        daily.sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")[["value"]]
        .rename(columns={"value": "fear_greed"})
    )

    for window in (1, 3, 5, 10):
        daily[f"fg_change_{window}d"] = daily["fear_greed"].diff(window)

    return daily


def download_market(start: pd.Timestamp) -> pd.DataFrame:
    ticker = CONFIG["ticker"]
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
        raise RuntimeError(f"No market data returned for {ticker}.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [
            "_".join(str(part) for part in column if str(part))
            for column in raw.columns
        ]

    def find(prefix: str) -> str | None:
        for column in raw.columns:
            normalized = normalize_column(column)
            if normalized == prefix or normalized.startswith(prefix + "_"):
                return str(column)
        return None

    columns = {name: find(name) for name in ("open", "high", "low", "close")}
    if columns["close"] is None:
        raise RuntimeError("Downloaded data has no close column.")

    market = pd.DataFrame(
        index=pd.to_datetime(raw.index).tz_localize(None).normalize()
    )
    for name, source in columns.items():
        market[name] = (
            pd.to_numeric(raw[source], errors="coerce")
            if source is not None
            else np.nan
        )

    market = market.dropna(subset=["close"]).sort_index()
    market.index.name = "market_date"
    market["return_1d"] = market["close"].pct_change()
    market["return_5d"] = market["close"].pct_change(5)
    market["return_20d"] = market["close"].pct_change(20)
    market["high_20d"] = market["close"].rolling(20, min_periods=1).max()
    market["drawdown_from_20d_high"] = market["close"] / market["high_20d"] - 1
    return market


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

    market_positions = {
        pd.Timestamp(index): position
        for position, index in enumerate(market.index)
    }

    for horizon in CONFIG["horizons"]:
        outcomes: list[float] = []
        for row in merged.itertuples():
            position = market_positions.get(pd.Timestamp(row.market_date))
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
        position = market_positions.get(pd.Timestamp(row.market_date))
        if position is None:
            drawdowns.append(np.nan)
            continue
        end = min(position + 19, len(market) - 1)
        lows = market["low"].iloc[position : end + 1].dropna()
        drawdowns.append(
            np.nan
            if lows.empty
            else float(lows.min() / row.entry_price - 1)
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
        min(max(minimum, 12), len(complete)),
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

    if sample:
        confidence_low = wilson_low(int((five > 0).sum()), sample)
    else:
        confidence_low = math.nan

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
        confidence = "Moderate" if sample >= 20 else "Low"
        rationale = (
            "Similar observations had favorable five-day odds and beat "
            "the sample's unconditional five-day average. Use staged buying, "
            "not an all-in decision."
        )
    elif (
        probability is not None
        and probability <= 0.45
        and average_5d is not None
        and average_5d < 0
    ):
        action = "WAIT ON EXTRA BUYING"
        tone = "negative"
        confidence = "Moderate" if sample >= 20 else "Low"
        rationale = (
            "Similar observations were more often followed by additional "
            "short-term weakness than gains."
        )
    else:
        action = "NEUTRAL"
        tone = "mixed"
        confidence = "Low"
        rationale = (
            "Historical outcomes were mixed or not strong enough to support "
            "a directional timing decision."
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


def value(value_: float | None, digits: int = 1) -> str:
    if value_ is None or not np.isfinite(value_):
        return "N/A"
    return f"{value_:.{digits}f}"


def main_chart(daily: pd.DataFrame, market: pd.DataFrame) -> str:
    chart_market = market.loc[market.index >= daily.index.min()]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=chart_market.index,
            y=chart_market["close"],
            name=f'{CONFIG["ticker"]} close',
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
            marker={"size": 5},
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
        height=500,
        margin={"l": 45, "r": 55, "t": 25, "b": 40},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08},
        yaxis={"title": f'{CONFIG["ticker"]} price'},
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


def outcomes_chart(analogs: pd.DataFrame) -> str:
    plot = analogs.dropna(subset=["forward_5d"]).sort_values("signal_date")
    figure = go.Figure(
        go.Bar(
            x=plot["signal_date"],
            y=plot["forward_5d"] * 100,
            name="5-day forward return",
            customdata=np.column_stack(
                [
                    plot["fear_greed"],
                    plot["fg_change_5d"].fillna(np.nan),
                ]
            ),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "5D return: %{y:.2f}%<br>"
                "Fear & Greed: %{customdata[0]:.1f}<br>"
                "5-observation change: %{customdata[1]:.1f}"
                "<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=0, line_width=1)
    figure.update_layout(
        template="plotly_white",
        height=350,
        margin={"l": 45, "r": 20, "t": 20, "b": 40},
        yaxis_title="Forward return (%)",
        xaxis_title="Signal date",
    )
    return figure.to_html(
        full_html=False,
        include_plotlyjs=False,
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
                lambda item: pct(item, 2)
                if pd.notna(item)
                else "—"
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


PAGE_TEMPLATE = Template(r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="robots" content="noindex,nofollow">
  <title>Fear & Greed Market Dashboard</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body data-build-id="{{ build_id }}">
  <header class="site-header">
    <div>
      <p class="eyebrow">RESEARCH DASHBOARD</p>
      <h1>Fear &amp; Greed vs. {{ ticker }}</h1>
      <p class="subtitle">
        Historical analogs, forward returns, downside risk, and a cautious
        rule-based action.
      </p>
    </div>
    <div class="updated">
      <span>Last built</span>
      <strong>{{ generated_display }}</strong>
      <span id="refresh-status">Checking for updates every {{ refresh_minutes }} minutes</span>
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
        {% if metric.note %}<small>{{ metric.note }}</small>{% endif %}
      </article>
      {% endfor %}
    </section>

    <section class="panel">
      <div class="panel-heading">
        <div>
          <p class="eyebrow">MARKET CONTEXT</p>
          <h2>Price and sentiment history</h2>
        </div>
      </div>
      {{ main_chart | safe }}
    </section>

    <section class="two-column">
      <article class="panel">
        <div class="panel-heading">
          <div>
            <p class="eyebrow">CURRENT ANALOGS</p>
            <h2>Five-day historical outcomes</h2>
          </div>
        </div>
        {{ outcomes_chart | safe }}
      </article>

      <article class="panel explanation">
        <p class="eyebrow">HOW TO READ THE ACTION</p>
        <h2>Decision rules</h2>
        <ul>
          <li><strong>Buy gradually:</strong> favorable historical odds and excess return, with enough examples.</li>
          <li><strong>Wait:</strong> similar observations were more often followed by additional weakness.</li>
          <li><strong>Neutral:</strong> mixed evidence; continue a normal schedule rather than market timing.</li>
          <li><strong>Insufficient evidence:</strong> sample too small to support an action.</li>
        </ul>
        <p class="fine-print">
          This is a mechanical research label, not personalized investment
          advice or a guarantee of future performance.
        </p>
      </article>
    </section>

    <section class="panel">
      <div class="panel-heading">
        <div>
          <p class="eyebrow">EVENT STUDY</p>
          <h2>Threshold and sudden-drop backtests</h2>
        </div>
        <a class="download" href="event_study.csv">Download CSV</a>
      </div>
      <div class="table-wrap">{{ event_table | safe }}</div>
    </section>

    <section class="panel">
      <div class="panel-heading">
        <div>
          <p class="eyebrow">HISTORICAL MATCHES</p>
          <h2>Observations most similar to today</h2>
        </div>
        <a class="download" href="analogs.csv">Download CSV</a>
      </div>
      <div class="table-wrap">{{ analog_table | safe }}</div>
    </section>

    <section class="panel methodology">
      <p class="eyebrow">METHODOLOGY AND LIMITATIONS</p>
      <h2>What the dashboard is actually measuring</h2>
      <p>
        Each daily Fear &amp; Greed observation is matched to the next market
        session, and forward returns begin from that session's opening price
        when available. A cooldown reduces repeated counting of one prolonged
        fear episode.
      </p>
      <p>
        Fear &amp; Greed may react to the same market decline it appears to
        predict. Missing sentiment dates, a small number of independent fear
        episodes, source revisions, and changing market regimes can all bias
        the result. The dashboard therefore disables strong language when the
        sample is too small.
      </p>
    </section>
  </main>

  <footer>
    <span>Data source for prices: Yahoo Finance via yfinance.</span>
    <span>Source repository may be private, but ordinary GitHub Pages output is public.</span>
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
  --positive: #137a46;
  --positive-bg: #eaf7f0;
  --negative: #a62929;
  --negative-bg: #fbecec;
  --mixed: #8a5b00;
  --mixed-bg: #fff6dc;
  --neutral-bg: #edf1f5;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
}

.site-header,
main,
.warnings,
footer {
  width: min(1180px, calc(100% - 32px));
  margin-inline: auto;
}

.site-header {
  display: flex;
  justify-content: space-between;
  gap: 32px;
  align-items: flex-end;
  padding: 40px 0 24px;
}

h1, h2, p { margin-top: 0; }
h1 { margin-bottom: 8px; font-size: clamp(2rem, 4vw, 3.5rem); }
h2 { margin-bottom: 10px; }
.subtitle, .fine-print { color: var(--muted); }
.eyebrow {
  margin-bottom: 7px;
  color: var(--accent);
  font-size: .75rem;
  font-weight: 800;
  letter-spacing: .12em;
}

.updated {
  display: grid;
  gap: 4px;
  text-align: right;
  color: var(--muted);
  font-size: .84rem;
}
.updated strong { color: var(--text); font-size: 1rem; }

.warnings { display: grid; gap: 8px; margin-bottom: 18px; }
.warning {
  border: 1px solid #e7c66d;
  border-radius: 10px;
  background: #fff8df;
  color: #644b00;
  padding: 12px 14px;
}

main { display: grid; gap: 20px; padding-bottom: 36px; }

.panel,
.action-panel,
.metric {
  border: 1px solid var(--border);
  border-radius: 16px;
  background: var(--panel);
}

.panel { padding: 22px; }
.panel-heading {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: center;
  margin-bottom: 8px;
}

.action-panel {
  display: grid;
  grid-template-columns: 1.5fr 1fr;
  gap: 24px;
  padding: 26px;
  border-width: 2px;
}
.action-positive { border-color: var(--positive); background: var(--positive-bg); }
.action-negative { border-color: var(--negative); background: var(--negative-bg); }
.action-mixed { border-color: #d2a72d; background: var(--mixed-bg); }
.action-neutral { background: var(--neutral-bg); }

.action-panel dl {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin: 0;
}
.action-panel dl div {
  padding: 12px;
  border: 1px solid rgba(100, 116, 139, .25);
  border-radius: 12px;
  background: rgba(255,255,255,.45);
}
dt { color: var(--muted); font-size: .78rem; }
dd { margin: 6px 0 0; font-weight: 800; }

.metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}
.metric { padding: 18px; }
.metric span, .metric small { display: block; color: var(--muted); }
.metric strong { display: block; margin: 8px 0 4px; font-size: 1.7rem; }

.two-column {
  display: grid;
  grid-template-columns: 1.4fr .8fr;
  gap: 20px;
}
.explanation ul { padding-left: 20px; line-height: 1.6; }

.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 12px;
}
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: .9rem;
  white-space: nowrap;
}
.data-table th,
.data-table td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  text-align: right;
}
.data-table th:first-child,
.data-table td:first-child { text-align: left; }
.data-table th { background: rgba(100,116,139,.08); }
.data-table tr:last-child td { border-bottom: 0; }

.download {
  color: var(--accent);
  text-decoration: none;
  font-weight: 700;
}
.methodology p { max-width: 90ch; line-height: 1.65; }

footer {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  padding: 22px 0 34px;
  border-top: 1px solid var(--border);
  color: var(--muted);
  font-size: .82rem;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0c1118;
    --panel: #121a24;
    --text: #edf2f7;
    --muted: #a1acba;
    --border: #2a3645;
    --accent: #83b4ff;
    --positive-bg: #102a20;
    --negative-bg: #301616;
    --mixed-bg: #2e260f;
    --neutral-bg: #17212d;
  }
  .warning { background: #2e260f; color: #f5d77d; border-color: #65551f; }
  .action-panel dl div { background: rgba(0,0,0,.12); }
}

@media (max-width: 850px) {
  .site-header,
  .action-panel,
  .two-column {
    grid-template-columns: 1fr;
  }
  .site-header { display: grid; align-items: start; }
  .updated { text-align: left; }
  .metrics { grid-template-columns: repeat(2, 1fr); }
  .action-panel dl { grid-template-columns: 1fr; }
  footer { display: grid; }
}

@media (max-width: 520px) {
  .metrics { grid-template-columns: 1fr; }
  .panel, .action-panel { padding: 16px; }
}
"""

JS = r"""
const intervalSeconds = Number(document.body.dataset.refreshSeconds || 300);
const currentBuild = document.body.dataset.buildId;
const statusNode = document.getElementById("refresh-status");

async function checkForUpdate() {
  try {
    const response = await fetch(`version.json?t=${Date.now()}`, {
      cache: "no-store"
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const version = await response.json();

    if (version.build_id && version.build_id !== currentBuild) {
      statusNode.textContent = "New dashboard available — refreshing";
      window.location.reload();
      return;
    }

    statusNode.textContent =
      `Current as of ${new Date(version.generated_at).toLocaleString()}`;
  } catch (error) {
    statusNode.textContent = "Could not check for a newer build";
  }
}

setInterval(checkForUpdate, intervalSeconds * 1000);
checkForUpdate();
"""


def main() -> None:
    SITE.mkdir(parents=True, exist_ok=True)

    fear_path = ROOT / CONFIG["fear_greed_file"]
    raw = read_table(fear_path)
    daily = daily_fear_greed(raw)
    market = download_market(daily.index.min())
    merged = merge_signals(daily, market)

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
            f"Fear & Greed history contains {large_gaps} gap(s) longer than seven days."
        )
    if completed_5d < 30:
        warnings.append(
            f"Only {completed_5d} observations have completed five-day outcomes."
        )
    if summary.sample_size < int(CONFIG["minimum_action_sample"]):
        warnings.append(
            "The action is disabled because there are too few completed analogs."
        )
    warnings.append(
        "A private source repository does not make an ordinary GitHub Pages site private."
    )

    metrics = [
        {
            "label": "Latest Fear & Greed",
            "value": value(float(latest["fear_greed"])),
            "note": daily.index.max().strftime("%Y-%m-%d"),
        },
        {
            "label": "5-observation change",
            "value": value(
                float(latest["fg_change_5d"])
                if pd.notna(latest["fg_change_5d"])
                else None
            ),
            "note": "Negative means worsening sentiment",
        },
        {
            "label": "Positive after 5 days",
            "value": pct(summary.probability_positive_5d),
            "note": f"{summary.sample_size} analogs",
        },
        {
            "label": "Average 5-day outcome",
            "value": pct(summary.average_return_5d),
            "note": f"Excess: {pct(summary.excess_average_5d)}",
        },
        {
            "label": "Median 5-day outcome",
            "value": pct(summary.median_return_5d),
            "note": "Less affected by outliers",
        },
        {
            "label": "Average 20-day outcome",
            "value": pct(summary.average_return_20d),
            "note": "Where available",
        },
        {
            "label": "Worst 5-day analog",
            "value": pct(summary.worst_return_5d),
            "note": "Historical downside example",
        },
        {
            "label": "Average worst drawdown, next 20D",
            "value": pct(summary.average_max_drawdown_20d),
            "note": "Adverse move after entry",
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

    page = PAGE_TEMPLATE.render(
        build_id=build_id,
        ticker=CONFIG["ticker"],
        generated_display=generated.strftime("%Y-%m-%d %H:%M UTC"),
        refresh_minutes=max(1, int(CONFIG["refresh_seconds"]) // 60),
        summary=summary,
        warnings=warnings,
        metrics=metrics,
        main_chart=main_chart(daily, market),
        outcomes_chart=outcomes_chart(analogs),
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

    page = page.replace(
        '<body data-build-id="' + build_id + '">',
        (
            '<body data-build-id="' + build_id + '" '
            f'data-refresh-seconds="{int(CONFIG["refresh_seconds"])}">'
        ),
    )

    (SITE / "index.html").write_text(page, encoding="utf-8")
    (SITE / "styles.css").write_text(CSS, encoding="utf-8")
    (SITE / "app.js").write_text(JS, encoding="utf-8")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")

    study.to_csv(SITE / "event_study.csv", index=False)
    analog_export.to_csv(SITE / "analogs.csv", index=False)
    merged.to_csv(SITE / "full_analysis.csv", index=False)

    report = {
        "generated_at": generated.isoformat(),
        "build_id": build_id,
        "ticker": CONFIG["ticker"],
        "latest_fear_greed_date": daily.index.max().date().isoformat(),
        "latest_fear_greed": float(latest["fear_greed"]),
        "five_observation_change": (
            None
            if pd.isna(latest["fg_change_5d"])
            else float(latest["fg_change_5d"])
        ),
        "summary": asdict(summary),
        "quality": {
            "raw_observations": len(raw),
            "daily_observations": len(daily),
            "completed_5d": completed_5d,
            "large_gaps": large_gaps,
        },
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

    print(f"Built {SITE / 'index.html'}")
    print(f"Action: {summary.action}")
    print(f"Analogs: {summary.sample_size}")


if __name__ == "__main__":
    main()
