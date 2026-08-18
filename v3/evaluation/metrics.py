#!/usr/bin/env python3
"""Common metric definitions for all v3 candidates and later policies."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TRADING_DAYS_PER_YEAR = 252.0


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    probability: pd.Series | np.ndarray,
    calibration_bins: int = 10,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("classification inputs must have equal nonzero length")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("classification inputs contain non-finite values")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("classification target must be binary")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")

    clipped = np.clip(p, 1e-9, 1.0 - 1e-9)
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))
    base_rate = float(np.mean(y))
    baseline_brier = float(np.mean((base_rate - y) ** 2))
    relative_brier_improvement = (
        float(1.0 - brier / baseline_brier) if baseline_brier > 0.0 else 0.0
    )

    edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    expected_calibration_error = 0.0
    maximum_calibration_error = 0.0
    for index in range(calibration_bins):
        if index == calibration_bins - 1:
            mask = (p >= edges[index]) & (p <= edges[index + 1])
        else:
            mask = (p >= edges[index]) & (p < edges[index + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
        expected_calibration_error += gap * count / len(y)
        maximum_calibration_error = max(maximum_calibration_error, gap)

    return {
        "observations": float(len(y)),
        "actual_rate": base_rate,
        "mean_prediction": float(np.mean(p)),
        "brier_score": brier,
        "baseline_brier_score": baseline_brier,
        "relative_brier_improvement": relative_brier_improvement,
        "log_loss": log_loss,
        "expected_calibration_error": float(expected_calibration_error),
        "maximum_calibration_error": float(maximum_calibration_error),
    }


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    if len(y) != len(pred) or len(y) == 0:
        raise ValueError("regression inputs must have equal nonzero length")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("regression inputs contain non-finite values")

    error = pred - y
    if len(y) >= 2 and len(np.unique(y)) >= 2 and len(np.unique(pred)) >= 2:
        correlation = float(spearmanr(y, pred).statistic)
        if not math.isfinite(correlation):
            correlation = 0.0
    else:
        correlation = 0.0

    return {
        "observations": float(len(y)),
        "mean_actual": float(np.mean(y)),
        "mean_prediction": float(np.mean(pred)),
        "mean_error": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "spearman_rank_correlation": correlation,
    }


def _longest_true_run(mask: pd.Series) -> int:
    longest = 0
    current = 0
    for value in mask.astype(bool).tolist():
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def performance_metrics(returns: pd.Series) -> dict[str, Any]:
    series = pd.Series(returns, copy=True).astype(float).dropna()
    if series.empty:
        raise ValueError("returns are empty")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("returns contain non-finite values")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("returns must use a DatetimeIndex")
    series = series.sort_index()

    equity = (1.0 + series).cumprod()
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    total_return = float(equity.iloc[-1] - 1.0)
    years = len(series) / TRADING_DAYS_PER_YEAR
    annualized_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    annualized_volatility = float(series.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(series) > 1 else 0.0
    daily_std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    sharpe = float(series.mean() / daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)) if daily_std > 0 else 0.0
    downside = series.loc[series < 0.0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = float(series.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)) if downside_std > 0 else 0.0
    max_drawdown = float(drawdown.min())
    calmar = float(annualized_return / abs(max_drawdown)) if max_drawdown < 0.0 else 0.0

    yearly = series.groupby(series.index.year).apply(lambda values: float((1.0 + values).prod() - 1.0))
    worst_year = int(yearly.idxmin())
    worst_year_return = float(yearly.min())
    underwater = drawdown < -1e-15

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_0rf": sharpe,
        "sortino_0rf": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "worst_year": worst_year,
        "worst_year_return": worst_year_return,
        "time_underwater_fraction": float(underwater.mean()),
        "longest_underwater_sessions": int(_longest_true_run(underwater)),
    }
