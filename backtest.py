#!/usr/bin/env python3
"""Unified backtest using the exact same point-in-time engine as the dashboard."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_common import (  # noqa: E402
    load_context,
    read_json,
    replay_with_outcomes,
    strategy_version,
)



def _json_safe(value):
    """Convert NumPy/Pandas values and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "strategy_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports",
    )
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--skip-yahoo-fallback", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def decision_exposure(
    row: pd.Series,
    settings: dict[str, Any],
) -> float:
    exposure = float(settings.get("baseline_exposure", 1.0))
    action = str(row.get("action", ""))
    tier = str(row.get("sizing_tier", ""))
    timing = str(row.get("timing_action", ""))

    if action == "BUY GRADUALLY":
        if "Elevated" in tier:
            exposure = float(
                settings.get("buy_elevated_exposure", 1.25)
            )
        else:
            exposure = float(
                settings.get("buy_modest_exposure", 1.15)
            )
    elif action == "WAIT ON BUYING":
        exposure = float(settings.get("wait_exposure", 0.75))

    if timing.startswith("EARLY BUY — FIRST TRANCHE"):
        exposure = max(
            exposure,
            float(
                settings.get(
                    "early_buy_first_tranche_exposure",
                    1.20,
                )
            ),
        )
    elif timing.startswith("EARLY BUY — SMALL START"):
        exposure = max(
            exposure,
            float(settings.get("early_buy_small_exposure", 1.10)),
        )
    elif timing.startswith("EARLY TRIM"):
        if "STRONG" in timing:
            exposure = min(
                exposure,
                float(settings.get("early_trim_strong_exposure", 0.80)),
            )
        else:
            exposure = min(
                exposure,
                float(settings.get("early_trim_exposure", 0.90)),
            )

    lower = float(settings.get("minimum_exposure", 0.50))
    upper = float(settings.get("maximum_exposure", 1.35))
    return float(np.clip(exposure, lower, upper))


def build_daily_backtest(
    market: pd.DataFrame,
    decisions: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    columns = [column for column in ("open", "close") if column in market]
    frame = market[columns].copy().sort_index()
    if "open" not in frame:
        frame["open"] = frame["close"]
    frame["open"] = frame["open"].fillna(frame["close"])
    frame["market_return"] = frame["close"].pct_change().fillna(0.0)
    previous_close = frame["close"].shift(1)
    frame["overnight_return"] = (frame["open"] / previous_close - 1.0).fillna(0.0)
    frame["intraday_return"] = (frame["close"] / frame["open"] - 1.0).fillna(0.0)

    usable = decisions.copy()
    usable["entry_date"] = pd.to_datetime(
        usable["entry_date"],
        errors="coerce",
    )
    usable = usable.dropna(subset=["entry_date"]).sort_values("entry_date")
    usable["target_exposure"] = usable.apply(
        lambda row: decision_exposure(row, settings),
        axis=1,
    )

    exposure_changes = (
        usable.groupby("entry_date")["target_exposure"].last()
    )
    frame["exposure"] = exposure_changes.reindex(frame.index)
    frame["exposure"] = frame["exposure"].ffill().fillna(
        float(settings.get("baseline_exposure", 1.0))
    )

    baseline_exposure = float(settings.get("baseline_exposure", 1.0))
    frame["prior_close_exposure"] = frame["exposure"].shift(1).fillna(
        baseline_exposure
    )
    frame["turnover"] = (
        frame["exposure"] - frame["prior_close_exposure"]
    ).abs()
    cost_bps = float(
        settings.get("transaction_cost_bps_per_1x_turnover", 2.0)
    )
    frame["transaction_cost"] = (
        frame["turnover"] * cost_bps / 10000.0
    )
    # The decision is executable at the next session's open. Existing exposure
    # earns the overnight gap; only the new target earns open-to-close return.
    # Multiplication preserves the interaction between both sub-periods.
    frame["strategy_return_gross"] = (
        (1.0 + frame["prior_close_exposure"] * frame["overnight_return"])
        * (1.0 + frame["exposure"] * frame["intraday_return"])
        - 1.0
    )
    frame["strategy_return"] = (
        frame["strategy_return_gross"] - frame["transaction_cost"]
    )
    return recompute_equity(frame)


def recompute_equity(frame: pd.DataFrame) -> pd.DataFrame:
    """Rebase equity curves after any requested date slice."""
    frame = frame.copy()
    frame["strategy_equity"] = (1.0 + frame["strategy_return"]).cumprod()
    frame["benchmark_equity"] = (1.0 + frame["market_return"]).cumprod()
    return frame


def max_drawdown(equity: pd.Series) -> float:
    running_high = equity.cummax()
    drawdown = equity / running_high - 1.0
    return float(drawdown.min())


def annualized_return(returns: pd.Series) -> float:
    if returns.empty:
        return math.nan
    total = float((1.0 + returns).prod())
    years = len(returns) / 252.0
    if years <= 0 or total <= 0:
        return math.nan
    return float(total ** (1.0 / years) - 1.0)


def performance_metrics(
    frame: pd.DataFrame,
    return_column: str,
    equity_column: str,
) -> dict[str, float]:
    returns = pd.to_numeric(frame[return_column], errors="coerce").dropna()
    equity = pd.to_numeric(frame[equity_column], errors="coerce").dropna()

    ann_return = annualized_return(returns)
    ann_vol = float(returns.std(ddof=1) * math.sqrt(252))
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * math.sqrt(252))
        if returns.std(ddof=1) > 0
        else math.nan
    )
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=1)
    sortino = (
        float(returns.mean() / downside_std * math.sqrt(252))
        if pd.notna(downside_std) and downside_std > 0
        else math.nan
    )
    drawdown = max_drawdown(equity)
    calmar = (
        float(ann_return / abs(drawdown))
        if math.isfinite(ann_return) and drawdown < 0
        else math.nan
    )
    running_high = equity.cummax()
    underwater = equity.lt(running_high)
    underwater_groups = underwater.ne(underwater.shift()).cumsum()
    underwater_lengths = underwater.groupby(underwater_groups).sum()

    yearly = (1.0 + returns).groupby(returns.index.year).prod() - 1.0
    worst_year = int(yearly.idxmin()) if not yearly.empty else None

    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_0rf": sharpe,
        "sortino_0rf": sortino,
        "max_drawdown": drawdown,
        "calmar": calmar,
        "worst_year": worst_year,
        "worst_year_return": (
            float(yearly.min()) if not yearly.empty else math.nan
        ),
        "time_underwater_fraction": float(underwater.mean()),
        "longest_underwater_sessions": (
            int(underwater_lengths.max()) if not underwater_lengths.empty else 0
        ),
    }


def holding_period_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Summarize independent target-exposure holding periods."""
    if frame.empty:
        return {"independent_trades": 0, "worst_trade": math.nan}

    changed = frame["exposure"].ne(frame["exposure"].shift())
    periods = changed.cumsum()
    returns = frame.groupby(periods)["strategy_return"].apply(
        lambda values: float((1.0 + values).prod() - 1.0)
    )
    return {
        "independent_trades": max(int(len(returns)) - 1, 0),
        "worst_trade": float(returns.min()) if not returns.empty else math.nan,
    }


def action_scorecard(decisions: pd.DataFrame) -> pd.DataFrame:
    usable = decisions[
        pd.to_numeric(decisions["forward_5d"], errors="coerce").notna()
    ].copy()
    usable["forward_5d"] = pd.to_numeric(
        usable["forward_5d"],
        errors="coerce",
    )
    usable["forward_20d"] = pd.to_numeric(
        usable["forward_20d"],
        errors="coerce",
    )

    rows = []
    for action, frame in usable.groupby("action"):
        rows.append(
            {
                "action": action,
                "observations": len(frame),
                "win_rate_5d": float((frame["forward_5d"] > 0).mean()),
                "average_5d": float(frame["forward_5d"].mean()),
                "median_5d": float(frame["forward_5d"].median()),
                "average_20d": float(frame["forward_20d"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("action")


def main() -> int:
    args = parse_args()
    raw_config = read_json(args.config)
    overlay_settings = raw_config.get("portfolio_backtest", {})

    context = load_context(
        args.config,
        allow_yahoo=not args.skip_yahoo_fallback,
    )
    decisions = replay_with_outcomes(
        context,
        progress_every=args.progress_every,
    )
    daily = build_daily_backtest(
        context.market,
        decisions,
        overlay_settings,
    )

    if args.start:
        evaluation_start = pd.Timestamp(args.start)
    else:
        entry_dates = pd.to_datetime(decisions["entry_date"], errors="coerce").dropna()
        if entry_dates.empty:
            raise SystemExit("No executable decision entry dates are available")
        evaluation_start = entry_dates.min()
    daily = daily.loc[daily.index >= evaluation_start]
    if args.end:
        daily = daily.loc[daily.index <= pd.Timestamp(args.end)]

    if daily.empty:
        raise SystemExit("No market rows remain after date filtering")

    daily = recompute_equity(daily)

    strategy = performance_metrics(
        daily,
        "strategy_return",
        "strategy_equity",
    )
    benchmark = performance_metrics(
        daily,
        "market_return",
        "benchmark_equity",
    )

    summary = {
        "strategy_version": strategy_version(args.manifest),
        "data_source": context.data_source,
        "start": daily.index.min().date().isoformat(),
        "end": daily.index.max().date().isoformat(),
        "strategy": strategy,
        "benchmark": benchmark,
        "annualized_return_difference": (
            strategy["annualized_return"]
            - benchmark["annualized_return"]
        ),
        "max_drawdown_difference": (
            strategy["max_drawdown"]
            - benchmark["max_drawdown"]
        ),
        "average_exposure": float(daily["exposure"].mean()),
        "maximum_exposure": float(daily["exposure"].max()),
        "minimum_exposure": float(daily["exposure"].min()),
        "total_turnover": float(daily["turnover"].sum()),
        **holding_period_metrics(daily),
        "benchmark_relative_total_return": (
            strategy["total_return"] - benchmark["total_return"]
        ),
        "assumptions": {
            "cash_return": 0.0,
            "borrowing_cost": 0.0,
            "transaction_cost_bps_per_1x_turnover": float(
                overlay_settings.get(
                    "transaction_cost_bps_per_1x_turnover",
                    2.0,
                )
            ),
            "note": (
                "This is a diagnostic exposure-overlay backtest. "
                "It is not a brokerage fill simulator."
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = args.output_dir / "backtest_daily.csv"
    decision_path = args.output_dir / "decision_outcomes.csv"
    scorecard_path = args.output_dir / "action_scorecard.csv"
    summary_path = args.output_dir / "backtest_summary.json"

    daily.to_csv(daily_path)
    decisions.to_csv(decision_path, index=False)
    action_scorecard(decisions).to_csv(scorecard_path, index=False)
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(json.dumps(_json_safe(summary), indent=2, allow_nan=False))
    print(f"\nWrote {daily_path}")
    print(f"Wrote {decision_path}")
    print(f"Wrote {scorecard_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
