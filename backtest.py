#!/usr/bin/env python3
"""
backtest.py

Backtests the Fear & Greed threshold strategy (buy when the index is low,
trim when it's high) against real price history for a target ticker
(default TQQQ), and compares it against two baselines:

  1. "All days"   - the average forward return starting from ANY trading
                     day in the sample (i.e. what you'd expect by chance).
  2. "Monthly DCA" - the average forward return starting from a fixed,
                     signal-blind monthly buy schedule.

This answers the question the rest of the repo can't currently answer:
does buying when the Fear & Greed Index is low actually beat doing nothing
special, for this specific ticker?

Usage:
    python backtest.py
    python backtest.py --ticker TQQQ --low 25 --high 75
    python backtest.py --price-csv data/tqqq_prices.csv
    python backtest.py --output data/backtest_signals.csv

Price data:
    By default this tries to fetch daily prices with yfinance. If yfinance
    isn't installed, can't reach the network (e.g. inside some CI/sandbox
    environments), or you'd rather supply your own data, pass
    --price-csv PATH to a CSV with at least "Date" and "Close" columns.

Exit codes:
    0: success
    1: data loading or fetch failure
    2: invalid command-line values
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_FG_CSV = "data/fear_greed_daily.csv"
DEFAULT_TICKER = "TQQQ"
DEFAULT_LOW = 25
DEFAULT_HIGH = 75
DEFAULT_HORIZONS = [5, 20, 60]  # trading days ~ 1wk, 4wk, 12wk


@dataclass
class BacktestResult:
    horizon_days: int
    signal_count: int
    signal_mean_return: float
    signal_win_rate: float
    all_days_mean_return: float
    monthly_dca_mean_return: float


def load_fear_greed(path: str) -> pd.DataFrame:
    """Load the repo's daily Fear & Greed CSV (Date, Value, Rating, ...)."""
    df = pd.read_csv(path)
    if "Date" not in df.columns or "Value" not in df.columns:
        raise ValueError(
            f"{path} must contain at least 'Date' and 'Value' columns, "
            f"found: {list(df.columns)}"
        )
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date", "Value"]]


def load_prices_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    close_col = "Close" if "Close" in df.columns else df.columns[-1]
    df = df[[date_col, close_col]].rename(columns={date_col: "Date", close_col: "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    return df


def load_prices_from_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Run 'pip install yfinance' or pass "
            "--price-csv with your own price data."
        ) from exc

    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty:
        raise RuntimeError(
            f"yfinance returned no data for {ticker} between {start} and {end}. "
            "Check the ticker symbol, your network connection, or supply "
            "--price-csv instead."
        )
    data = data.reset_index()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]
    df = data[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def merge_signal_and_price(fg: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """Align Fear & Greed readings to the nearest trading day's close."""
    price = price.sort_values("Date")
    fg = fg.sort_values("Date")
    merged = pd.merge_asof(fg, price, on="Date", direction="forward")
    merged = merged.dropna(subset=["Close"]).reset_index(drop=True)
    return merged


def forward_return(price: pd.DataFrame, start_idx: int, horizon_days: int) -> float | None:
    end_idx = start_idx + horizon_days
    if end_idx >= len(price):
        return None
    start_price = price.iloc[start_idx]["Close"]
    end_price = price.iloc[end_idx]["Close"]
    if start_price <= 0:
        return None
    return (end_price / start_price) - 1.0


def compute_baselines(price: pd.DataFrame, horizons: list[int]) -> dict[int, dict[str, float]]:
    """All-days average return and monthly-DCA average return, per horizon."""
    price = price.reset_index(drop=True)
    price["_month"] = price["Date"].dt.to_period("M")
    monthly_idx = price.groupby("_month").head(1).index.tolist()

    baselines: dict[int, dict[str, float]] = {}
    for h in horizons:
        all_rets = [
            forward_return(price, i, h)
            for i in range(len(price))
            if forward_return(price, i, h) is not None
        ]
        dca_rets = [
            forward_return(price, i, h)
            for i in monthly_idx
            if forward_return(price, i, h) is not None
        ]
        baselines[h] = {
            "all_days_mean": float(np.mean(all_rets)) if all_rets else float("nan"),
            "monthly_dca_mean": float(np.mean(dca_rets)) if dca_rets else float("nan"),
        }
    return baselines


def run_backtest(
    merged: pd.DataFrame,
    price: pd.DataFrame,
    low: float,
    high: float,
    horizons: list[int],
) -> tuple[list[BacktestResult], list[BacktestResult], pd.DataFrame]:
    """Run buy-signal (low) and trim-signal (high) analysis."""
    price = price.reset_index(drop=True)
    price_index_by_date = {d: i for i, d in enumerate(price["Date"])}

    baselines = compute_baselines(price, horizons)

    def analyze(signal_mask: pd.Series) -> tuple[list[BacktestResult], pd.DataFrame]:
        rows = []
        results = []
        signal_dates = merged.loc[signal_mask, "Date"]
        for h in horizons:
            rets = []
            for d in signal_dates:
                idx = price_index_by_date.get(d)
                if idx is None:
                    continue
                r = forward_return(price, idx, h)
                if r is not None:
                    rets.append(r)
                    if h == horizons[0]:
                        rows.append({"Date": d, "FG_Value": merged.loc[merged["Date"] == d, "Value"].iloc[0]})
            mean_ret = float(np.mean(rets)) if rets else float("nan")
            win_rate = float(np.mean([r > 0 for r in rets])) if rets else float("nan")
            results.append(
                BacktestResult(
                    horizon_days=h,
                    signal_count=len(rets),
                    signal_mean_return=mean_ret,
                    signal_win_rate=win_rate,
                    all_days_mean_return=baselines[h]["all_days_mean"],
                    monthly_dca_mean_return=baselines[h]["monthly_dca_mean"],
                )
            )
        log_df = pd.DataFrame(rows).drop_duplicates(subset="Date") if rows else pd.DataFrame(columns=["Date", "FG_Value"])
        return results, log_df

    buy_results, buy_log = analyze(merged["Value"] <= low)
    sell_results, sell_log = analyze(merged["Value"] >= high)

    buy_log["signal"] = "buy (low fear/greed)"
    sell_log["signal"] = "trim (high fear/greed)"
    combined_log = pd.concat([buy_log, sell_log], ignore_index=True).sort_values("Date")

    return buy_results, sell_results, combined_log


def print_report(ticker: str, low: float, high: float, buy_results, sell_results) -> None:
    horizon_labels = {5: "1 week", 20: "1 month", 60: "~1 quarter"}

    def fmt_pct(x: float) -> str:
        return "n/a" if x != x else f"{x * 100:+.2f}%"

    print(f"\nBacktest: {ticker} vs Fear & Greed Index (low<={low}, high>={high})")
    print("=" * 72)

    print(f"\nBUY SIGNAL  (Fear & Greed <= {low})")
    print(f"{'Horizon':<12}{'Signals':<10}{'Win rate':<12}{'Signal avg':<14}{'All-days avg':<14}{'Monthly DCA avg'}")
    for r in buy_results:
        label = horizon_labels.get(r.horizon_days, f"{r.horizon_days}d")
        win = "n/a" if r.signal_win_rate != r.signal_win_rate else f"{r.signal_win_rate * 100:.0f}%"
        print(
            f"{label:<12}{r.signal_count:<10}{win:<12}{fmt_pct(r.signal_mean_return):<14}"
            f"{fmt_pct(r.all_days_mean_return):<14}{fmt_pct(r.monthly_dca_mean_return)}"
        )

    print(f"\nTRIM SIGNAL (Fear & Greed >= {high})")
    print(f"{'Horizon':<12}{'Signals':<10}{'Win rate':<12}{'Signal avg':<14}{'All-days avg':<14}{'Monthly DCA avg'}")
    for r in sell_results:
        label = horizon_labels.get(r.horizon_days, f"{r.horizon_days}d")
        win = "n/a" if r.signal_win_rate != r.signal_win_rate else f"{r.signal_win_rate * 100:.0f}%"
        print(
            f"{label:<12}{r.signal_count:<10}{win:<12}{fmt_pct(r.signal_mean_return):<14}"
            f"{fmt_pct(r.all_days_mean_return):<14}{fmt_pct(r.monthly_dca_mean_return)}"
        )

    print(
        "\nRead this as: does 'Signal avg' meaningfully beat 'All-days avg' and "
        "'Monthly DCA avg'? If it doesn't, the threshold isn't adding value over "
        "buying on a fixed schedule or a random day for this ticker/period.\n"
        "This is historical only and not a guarantee of future results.\n"
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the Fear & Greed strategy against price history.")
    parser.add_argument("--fg-csv", default=DEFAULT_FG_CSV, help=f"Path to Fear & Greed CSV (default: {DEFAULT_FG_CSV})")
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help=f"Ticker to fetch via yfinance (default: {DEFAULT_TICKER})")
    parser.add_argument("--price-csv", default=None, help="Use a local price CSV instead of fetching via yfinance")
    parser.add_argument("--low", type=float, default=DEFAULT_LOW, help=f"Low (buy) threshold (default: {DEFAULT_LOW})")
    parser.add_argument("--high", type=float, default=DEFAULT_HIGH, help=f"High (trim) threshold (default: {DEFAULT_HIGH})")
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help="Comma-separated forward-looking horizons in trading days (default: 5,20,60)",
    )
    parser.add_argument("--output", default=None, help="Optional path to write the individual signal log as CSV")
    args = parser.parse_args(argv)

    if not (0 <= args.low < args.high <= 100):
        parser.error("--low must be < --high, and both must be between 0 and 100")

    try:
        args.horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    except ValueError:
        parser.error("--horizons must be a comma-separated list of integers, e.g. 5,20,60")
    if not args.horizons:
        parser.error("--horizons must contain at least one value")

    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        fg = load_fear_greed(args.fg_csv)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading Fear & Greed data: {exc}", file=sys.stderr)
        return 1

    try:
        if args.price_csv:
            price = load_prices_from_csv(args.price_csv)
        else:
            start = fg["Date"].min().strftime("%Y-%m-%d")
            end = fg["Date"].max().strftime("%Y-%m-%d")
            price = load_prices_from_yfinance(args.ticker, start, end)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error loading price data: {exc}", file=sys.stderr)
        return 1

    if price.empty:
        print("Error: price data is empty.", file=sys.stderr)
        return 1

    merged = merge_signal_and_price(fg, price)
    if merged.empty:
        print("Error: no overlapping dates between Fear & Greed data and price data.", file=sys.stderr)
        return 1

    buy_results, sell_results, log_df = run_backtest(merged, price, args.low, args.high, args.horizons)
    print_report(args.ticker, args.low, args.high, buy_results, sell_results)

    if args.output:
        log_df.to_csv(args.output, index=False)
        print(f"Signal log written to {args.output} ({len(log_df)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
