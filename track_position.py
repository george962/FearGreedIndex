#!/usr/bin/env python3
"""
track_position.py

Turns generic "market is fearful/greedy" alerts into a personalized one by
combining your actual holdings with the current Fear & Greed reading.

You maintain positions.json by hand after each trade (a couple of fields,
takes 10 seconds) - this script does the math and tells you where you
actually stand relative to your target allocation.

positions.json shape:
{
  "ticker": "TQQQ",
  "target_allocation_pct": 50,
  "total_portfolio_value": 20000,
  "lots": [
    {"date": "2026-06-01", "shares": 30, "price": 62.30},
    {"date": "2026-07-15", "shares": 15, "price": 58.10}
  ]
}

Usage:
    python track_position.py
    python track_position.py --price 73.80
    python track_position.py --price 73.80 --fg-value 22 --fg-rating fear
    python track_position.py --json
    python track_position.py --github-output

Exit codes:
    0: success
    1: file, data, or price-fetch failure
    2: invalid command-line values
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict

DEFAULT_POSITIONS_FILE = "positions.json"


@dataclass
class PositionStatus:
    ticker: str
    shares_held: float
    avg_cost: float
    cost_basis: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_pct: float
    total_portfolio_value: float
    current_allocation_pct: float
    target_allocation_pct: float
    allocation_gap_pct: float
    allocation_gap_dollars: float


def load_positions(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    for field in ("ticker", "target_allocation_pct", "total_portfolio_value", "lots"):
        if field not in data:
            raise ValueError(f"{path} is missing required field '{field}'")
    return data


def fetch_current_price(ticker: str) -> float:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed and no --price was given. Run "
            "'pip install yfinance' or pass --price manually."
        ) from exc

    info = yf.Ticker(ticker).history(period="1d")
    if info.empty:
        raise RuntimeError(
            f"yfinance returned no data for {ticker}. Check the ticker, your "
            "network connection, or pass --price manually."
        )
    return float(info["Close"].iloc[-1])


def compute_status(data: dict, current_price: float) -> PositionStatus:
    lots = data["lots"]
    shares_held = sum(lot["shares"] for lot in lots)
    cost_basis = sum(lot["shares"] * lot["price"] for lot in lots)
    avg_cost = cost_basis / shares_held if shares_held else 0.0

    market_value = shares_held * current_price
    unrealized_pl = market_value - cost_basis
    unrealized_pl_pct = (unrealized_pl / cost_basis) if cost_basis else 0.0

    total_portfolio_value = data["total_portfolio_value"]
    current_allocation_pct = (
        (market_value / total_portfolio_value) * 100 if total_portfolio_value else 0.0
    )
    target_allocation_pct = data["target_allocation_pct"]
    allocation_gap_pct = target_allocation_pct - current_allocation_pct
    allocation_gap_dollars = (allocation_gap_pct / 100) * total_portfolio_value

    return PositionStatus(
        ticker=data["ticker"],
        shares_held=shares_held,
        avg_cost=avg_cost,
        cost_basis=cost_basis,
        current_price=current_price,
        market_value=market_value,
        unrealized_pl=unrealized_pl,
        unrealized_pl_pct=unrealized_pl_pct,
        total_portfolio_value=total_portfolio_value,
        current_allocation_pct=current_allocation_pct,
        target_allocation_pct=target_allocation_pct,
        allocation_gap_pct=allocation_gap_pct,
        allocation_gap_dollars=allocation_gap_dollars,
    )


def suggest_action(status: PositionStatus, fg_value: float | None, fg_rating: str | None) -> str:
    """Combine allocation gap with an optional sentiment reading.

    This is informational only - it reflects your own stated target
    allocation, not a recommendation to buy or sell anything.
    """
    UNDERWEIGHT_TOL = 2.0  # percentage points treated as "on target"

    if abs(status.allocation_gap_pct) <= UNDERWEIGHT_TOL:
        base = "You're within tolerance of your target allocation. No allocation-driven action needed."
    elif status.allocation_gap_pct > 0:
        base = (
            f"You're underweight target by {status.allocation_gap_pct:.1f} percentage points "
            f"(~${status.allocation_gap_dollars:,.0f}, ~{status.allocation_gap_dollars / status.current_price:.1f} shares "
            f"at current price)."
        )
    else:
        base = (
            f"You're overweight target by {abs(status.allocation_gap_pct):.1f} percentage points "
            f"(~${abs(status.allocation_gap_dollars):,.0f}, ~{abs(status.allocation_gap_dollars) / status.current_price:.1f} shares "
            f"at current price)."
        )

    if fg_value is None:
        return base

    rating = (fg_rating or "").lower()
    if fg_value <= 25 or "fear" in rating:
        sentiment = "Market sentiment is in fear."
        if status.allocation_gap_pct > UNDERWEIGHT_TOL:
            tag = " This combination (fear + underweight) is the setup your dip-buy rules are built for."
        elif status.allocation_gap_pct < -UNDERWEIGHT_TOL:
            tag = " You're already overweight, so a fear reading alone doesn't add a reason to buy more."
        else:
            tag = ""
    elif fg_value >= 75 or "greed" in rating:
        sentiment = "Market sentiment is in greed."
        if status.allocation_gap_pct < -UNDERWEIGHT_TOL:
            tag = " This combination (greed + overweight) is the setup your trim rules are built for."
        elif status.allocation_gap_pct > UNDERWEIGHT_TOL:
            tag = " You're still underweight despite greed, so this isn't a strong add signal on its own."
        else:
            tag = ""
    else:
        sentiment = "Market sentiment is in the normal range."
        tag = ""

    return f"{base} {sentiment}{tag}"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track your position vs target allocation.")
    parser.add_argument("--positions-file", default=DEFAULT_POSITIONS_FILE, help=f"Path to positions JSON (default: {DEFAULT_POSITIONS_FILE})")
    parser.add_argument("--price", type=float, default=None, help="Current price override (skips yfinance fetch)")
    parser.add_argument("--fg-value", type=float, default=None, help="Current Fear & Greed value, to include a combined suggestion")
    parser.add_argument("--fg-rating", default=None, help="Current Fear & Greed rating label (e.g. 'fear', 'greed')")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a human-readable report")
    parser.add_argument("--github-output", action="store_true", help="Write key values to $GITHUB_OUTPUT if available")
    return parser.parse_args(argv)


def write_github_output(status: PositionStatus, action: str) -> None:
    out_path = os.environ.get("GITHUB_OUTPUT")
    if not out_path:
        return
    with open(out_path, "a") as f:
        f.write(f"current_allocation_pct={status.current_allocation_pct:.2f}\n")
        f.write(f"target_allocation_pct={status.target_allocation_pct:.2f}\n")
        f.write(f"allocation_gap_pct={status.allocation_gap_pct:.2f}\n")
        f.write(f"unrealized_pl_pct={status.unrealized_pl_pct * 100:.2f}\n")
        f.write(f"suggested_action={action}\n")


def print_report(status: PositionStatus, action: str) -> None:
    print(f"\nPosition status: {status.ticker}")
    print("=" * 50)
    print(f"Shares held:          {status.shares_held:g}")
    print(f"Average cost:         ${status.avg_cost:,.2f}")
    print(f"Current price:        ${status.current_price:,.2f}")
    print(f"Market value:         ${status.market_value:,.2f}")
    print(f"Unrealized P/L:       ${status.unrealized_pl:,.2f} ({status.unrealized_pl_pct * 100:+.2f}%)")
    print(f"Current allocation:   {status.current_allocation_pct:.1f}% of portfolio")
    print(f"Target allocation:    {status.target_allocation_pct:.1f}%")
    print(f"Allocation gap:       {status.allocation_gap_pct:+.1f} pts (${status.allocation_gap_dollars:+,.0f})")
    print(f"\n{action}\n")
    print("This is a status report based on your own stated target allocation, not investment advice.\n")


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        data = load_positions(args.positions_file)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error loading {args.positions_file}: {exc}", file=sys.stderr)
        return 1

    try:
        current_price = args.price if args.price is not None else fetch_current_price(data["ticker"])
    except RuntimeError as exc:
        print(f"Error fetching price: {exc}", file=sys.stderr)
        return 1

    if not data["lots"]:
        print("No lots recorded in positions file yet - nothing to report.", file=sys.stderr)
        return 1

    status = compute_status(data, current_price)
    action = suggest_action(status, args.fg_value, args.fg_rating)

    if args.github_output:
        write_github_output(status, action)

    if args.json:
        payload = asdict(status)
        payload["suggested_action"] = action
        print(json.dumps(payload, indent=2))
    else:
        print_report(status, action)

    return 0


if __name__ == "__main__":
    sys.exit(main())
