from __future__ import annotations

import argparse
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import FearGreedMarketData as market_data


class FearGreedMarketDataTests(unittest.TestCase):
    def make_args(self, root: Path) -> argparse.Namespace:
        return argparse.Namespace(
            fear_greed_input=root / "data" / "fear_greed_daily.csv",
            spx_cache=root / "data" / "spx_daily.csv",
            output=root / "data" / "fear_greed_spx_daily.csv",
            symbol="^GSPC",
            context_days=450,
            overlap_days=10,
            timeout=30,
            force_refresh=False,
            json=False,
        )

    def write_fear_greed(self, path: Path, rows: list[list[str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["Date,Value,Rating,Source Timestamp UTC"]
        lines.extend(",".join(row) for row in rows)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def make_market_history(self, periods: int = 420) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-02", periods=periods, tz="America/New_York")
        base = pd.Series(range(periods), dtype=float) + 4500.0
        return pd.DataFrame(
            {
                "Open": base.to_numpy(),
                "High": (base + 12).to_numpy(),
                "Low": (base - 8).to_numpy(),
                "Close": (base + 5).to_numpy(),
                "Adj Close": (base + 5).to_numpy(),
                "Volume": [0] * periods,
            },
            index=dates,
        )

    def test_no_score_rows_skip_yfinance_entirely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root)
            self.write_fear_greed(
                args.fear_greed_input,
                [["2026-07-31", "", "fear", "2026-07-31T20:00:00Z"]],
            )

            with patch.object(
                market_data,
                "fetch_spx_history",
                side_effect=AssertionError("yfinance must not be called"),
            ):
                result = market_data.run(args, today=date(2026, 8, 1))

            self.assertEqual(result["status"], "no_fear_greed_data")
            self.assertFalse(result["market_request_made"])
            self.assertFalse(args.spx_cache.exists())
            self.assertFalse(args.output.exists())

    def test_builds_cache_and_analysis_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root)
            self.write_fear_greed(
                args.fear_greed_input,
                [
                    ["2025-07-01", "24", "extreme_fear", "2025-07-01T20:00:00Z"],
                    ["2025-07-02", "29", "fear", "2025-07-02T20:00:00Z"],
                    ["2025-07-03", "35", "fear", "2025-07-03T20:00:00Z"],
                ],
            )
            history = self.make_market_history()

            with patch.object(
                market_data,
                "fetch_spx_history",
                return_value=market_data.normalize_downloaded_history(
                    history, "^GSPC"
                ),
            ) as mocked_fetch:
                result = market_data.run(args, today=date(2025, 8, 1))

            self.assertTrue(result["market_request_made"])
            mocked_fetch.assert_called_once()
            self.assertTrue(args.spx_cache.exists())
            self.assertTrue(args.output.exists())

            output = pd.read_csv(args.output)
            self.assertEqual(len(output), 3)
            self.assertIn("spx_close", output.columns)
            self.assertIn("spx_return_20d_pct", output.columns)
            self.assertIn("spx_volatility_20d_annualized_pct", output.columns)
            self.assertIn("outcome_forward_return_20d_pct", output.columns)
            self.assertIn(
                "outcome_forward_max_drawdown_20d_pct", output.columns
            )
            self.assertEqual(output.loc[1, "fear_greed_change_1d"], 5.0)

    def test_weekend_observation_matches_previous_trading_day(self) -> None:
        fear = pd.DataFrame(
            {
                "fear_greed_date_utc": pd.to_datetime(["2026-07-11"]),
                "fear_greed_market_date": pd.to_datetime(["2026-07-11"]),
                "fear_greed_value": [20.0],
                "fear_greed_rating": ["extreme fear"],
                "fear_greed_source_timestamp_utc": pd.to_datetime(
                    ["2026-07-11T12:00:00Z"], utc=True
                ),
            }
        )
        cache = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-07-09", "2026-07-10"]),
                "symbol": ["^GSPC", "^GSPC"],
                "open": [6100.0, 6120.0],
                "high": [6130.0, 6150.0],
                "low": [6080.0, 6100.0],
                "close": [6120.0, 6140.0],
                "adj_close": [6120.0, 6140.0],
                "volume": [0, 0],
                "data_source": [market_data.DATA_SOURCE] * 2,
            }
        )

        output = market_data.build_analysis_dataset(fear, cache)
        self.assertEqual(output.loc[0, "spx_date"], pd.Timestamp("2026-07-10"))
        self.assertEqual(output.loc[0, "spx_match_lag_calendar_days"], 1)
        self.assertFalse(bool(output.loc[0, "spx_same_trading_day"]))

    def test_current_cache_skips_market_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root)
            self.write_fear_greed(
                args.fear_greed_input,
                [["2025-07-03", "35", "fear", "2025-07-03T20:00:00Z"]],
            )
            normalized = market_data.normalize_downloaded_history(
                self.make_market_history(), "^GSPC"
            )
            args.spx_cache.parent.mkdir(parents=True, exist_ok=True)
            args.spx_cache.write_text(
                market_data.render_csv(normalized), encoding="utf-8"
            )

            with patch.object(
                market_data,
                "fetch_spx_history",
                side_effect=AssertionError("current cache should be reused"),
            ):
                result = market_data.run(args, today=date(2025, 7, 31))

            self.assertFalse(result["market_request_made"])
            self.assertTrue(args.output.exists())


if __name__ == "__main__":
    unittest.main()
