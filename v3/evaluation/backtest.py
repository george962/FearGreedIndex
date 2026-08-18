#!/usr/bin/env python3
"""Generic portfolio evaluation utilities for later v3 decision policies.

V3-009 deliberately does not map model predictions to actions. These functions
accept an already-defined executable exposure series so V3-016/V3-017 can use
one common, tested backtest implementation instead of embedding portfolio logic
inside model code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from v3.evaluation.metrics import performance_metrics


def simulate_exposure_strategy(
    market_returns: pd.Series,
    executable_exposure: pd.Series,
    transaction_cost_bps_per_1x_turnover: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    market = pd.Series(market_returns, copy=True).astype(float).rename("market_return")
    exposure = pd.Series(executable_exposure, copy=True).astype(float).rename("exposure")
    frame = pd.concat([market, exposure], axis=1, join="inner").dropna()
    if frame.empty:
        raise ValueError("No overlapping market returns and executable exposure")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Backtest inputs must use a DatetimeIndex")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Backtest inputs contain non-finite values")
    if (frame["exposure"] < 0.0).any():
        raise ValueError("This long-only evaluator does not accept negative exposure")

    frame = frame.sort_index()
    frame["turnover"] = frame["exposure"].diff().abs().fillna(0.0)
    cost_rate = float(transaction_cost_bps_per_1x_turnover) / 10_000.0
    frame["transaction_cost"] = frame["turnover"] * cost_rate
    frame["strategy_return"] = (
        frame["exposure"] * frame["market_return"] - frame["transaction_cost"]
    )
    frame["strategy_equity"] = (1.0 + frame["strategy_return"]).cumprod()
    frame["benchmark_equity"] = (1.0 + frame["market_return"]).cumprod()

    strategy = performance_metrics(frame["strategy_return"])
    benchmark = performance_metrics(frame["market_return"])
    summary: dict[str, Any] = {
        "strategy": strategy,
        "benchmark": benchmark,
        "benchmark_relative_total_return": float(
            strategy["total_return"] - benchmark["total_return"]
        ),
        "annualized_return_difference": float(
            strategy["annualized_return"] - benchmark["annualized_return"]
        ),
        "average_exposure": float(frame["exposure"].mean()),
        "maximum_exposure": float(frame["exposure"].max()),
        "minimum_exposure": float(frame["exposure"].min()),
        "total_turnover": float(frame["turnover"].sum()),
        "transaction_cost_bps_per_1x_turnover": float(
            transaction_cost_bps_per_1x_turnover
        ),
    }
    return frame, summary
