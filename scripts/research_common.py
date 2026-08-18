#!/usr/bin/env python3
"""Shared research helpers that call the dashboard engine as the source of truth."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_dashboard as engine  # noqa: E402


@dataclass
class ResearchContext:
    settings: Any
    daily: pd.DataFrame
    market: pd.DataFrame
    events: pd.DataFrame
    data_source: str
    config_path: Path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strategy_version(manifest_path: Path) -> str:
    raw = read_json(manifest_path)
    value = str(raw.get("strategy_version", "")).strip()
    if not value:
        raise ValueError(f"{manifest_path} has no strategy_version")
    return value


def plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def load_context(
    config_path: Path,
    *,
    allow_yahoo: bool = True,
) -> ResearchContext:
    settings = engine.Settings.load(config_path)
    daily, market, source = engine.load_data(
        settings,
        ROOT,
        allow_yahoo=allow_yahoo,
    )
    daily, market = engine.add_features(daily, market)
    events = engine.merge_signals(daily, market, settings)
    return ResearchContext(
        settings=settings,
        daily=daily,
        market=market,
        events=events,
        data_source=source,
        config_path=config_path,
    )


def attach_outcomes(
    history: pd.DataFrame,
    events: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """Attach entry information and realized outcomes to point-in-time decisions."""
    if history.empty:
        return history.copy()

    event_frame = engine.attach_outcome_known_dates(
        events,
        market,
        horizons=(1, 5, 10, 20, 60),
    )
    event_frame["decision_date"] = pd.to_datetime(
        event_frame["signal_date"], errors="coerce"
    ).dt.date.astype(str)

    wanted = [
        "decision_date",
        "entry_date",
        "entry_price",
        "forward_1d",
        "forward_5d",
        "forward_10d",
        "forward_20d",
        "forward_60d",
        "max_drawdown_20d",
        "_forward_1d_known_date",
        "_forward_5d_known_date",
        "_forward_10d_known_date",
        "_forward_20d_known_date",
        "_forward_60d_known_date",
    ]
    available = [column for column in wanted if column in event_frame.columns]
    event_frame = event_frame[available].drop_duplicates(
        subset=["decision_date"],
        keep="last",
    )

    merged = history.copy()
    merged["decision_date"] = merged["decision_date"].astype(str)
    merged = merged.merge(
        event_frame,
        on="decision_date",
        how="left",
        validate="one_to_one",
    )
    return merged


def replay_with_outcomes(
    context: ResearchContext,
    *,
    progress_every: int = 100,
) -> pd.DataFrame:
    history = engine.replay_historical_decisions(
        context.settings,
        context.daily,
        context.market,
        context.events,
        progress_every=progress_every,
    )
    return attach_outcomes(history, context.events, context.market)


def current_prediction(
    context: ResearchContext,
    manifest_path: Path,
) -> dict[str, Any]:
    """Generate the current decision without replaying the full history."""
    current = engine.build_current_context(context.daily, context.market)
    cutoff_market_date = pd.Timestamp(current["market_date"]).normalize()

    prepared = engine.attach_outcome_known_dates(
        context.events,
        context.market,
        horizons=(5, 20),
    )
    eligible = engine.eligible_events_asof(
        prepared,
        cutoff_market_date,
    )
    analogs, method = engine.find_analogs(
        eligible,
        current,
        context.settings,
    )
    baseline = engine.find_regime_baseline(
        eligible,
        current,
        context.settings,
    )
    analogs = engine.mask_analog_outcomes_asof(
        analogs,
        context.market,
        cutoff_market_date,
    )

    verdict = engine.score_analogs(
        analogs,
        baseline,
        current,
        method,
        context.settings,
    )
    guidance = engine.build_position_guidance(
        verdict,
        context.settings,
    )
    timing = engine.score_fast_timing(
        context.daily,
        context.market,
        current,
        context.settings,
    )

    return {
        "decision_date": pd.Timestamp(current["signal_date"]).date().isoformat(),
        "market_date": cutoff_market_date.date().isoformat(),
        "fear_greed": plain(current.get("fear_greed")),
        "fg_change_5": plain(current.get("fg_change_5")),
        "market_regime": verdict.market_regime,
        "market_extension": verdict.market_extension,
        "action": verdict.action,
        "confidence": verdict.confidence,
        "sizing_tier": guidance.tier,
        "sizing_label": guidance.sizing_label,
        "timing_action": timing.action,
        "timing_side": timing.side,
        "timing_score": timing.score,
        "timing_confirmation_count": timing.confirmation_count,
        "timing_confirmation_total": timing.confirmation_total,
        "analog_sample": verdict.sample_size,
        "regime_baseline_sample": verdict.regime_baseline_sample,
        "win_rate_5d": plain(verdict.win_rate_5d),
        "average_5d": plain(verdict.average_5d),
        "average_20d": plain(verdict.average_20d),
        "regime_baseline_5d": plain(verdict.regime_baseline_5d),
        "excess_5d": plain(verdict.excess_5d),
        "excess_ci_low_5d": plain(verdict.excess_ci_low_5d),
        "excess_ci_high_5d": plain(verdict.excess_ci_high_5d),
        "average_drawdown_20d": plain(verdict.average_drawdown_20d),
        "strategy_version": strategy_version(manifest_path),
        "config_sha256": file_sha256(context.config_path),
        "data_source": context.data_source,
    }
