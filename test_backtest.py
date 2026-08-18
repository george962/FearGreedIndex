import unittest

import pandas as pd

from backtest import build_daily_backtest, recompute_equity


class BacktestAlignmentTests(unittest.TestCase):
    def test_next_open_decision_does_not_capture_overnight_gap(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        market = pd.DataFrame(
            {
                "open": [100.0, 110.0],
                "close": [100.0, 110.0],
            },
            index=dates,
        )
        decisions = pd.DataFrame(
            {
                "entry_date": [dates[1]],
                "action": ["WAIT ON BUYING"],
                "sizing_tier": [""],
                "timing_action": [""],
            }
        )
        frame = build_daily_backtest(
            market,
            decisions,
            {
                "baseline_exposure": 1.0,
                "wait_exposure": 0.5,
                "transaction_cost_bps_per_1x_turnover": 0.0,
            },
        )

        # The old implementation incorrectly earned only half of this gap.
        self.assertAlmostEqual(frame.loc[dates[1], "strategy_return"], 0.10)

    def test_new_exposure_applies_to_entry_day_intraday_return(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        market = pd.DataFrame(
            {"open": [100.0, 100.0], "close": [100.0, 110.0]},
            index=dates,
        )
        decisions = pd.DataFrame(
            {
                "entry_date": [dates[1]],
                "action": ["WAIT ON BUYING"],
                "sizing_tier": [""],
                "timing_action": [""],
            }
        )
        frame = build_daily_backtest(
            market,
            decisions,
            {
                "baseline_exposure": 1.0,
                "wait_exposure": 0.5,
                "transaction_cost_bps_per_1x_turnover": 0.0,
            },
        )
        self.assertAlmostEqual(frame.loc[dates[1], "strategy_return"], 0.05)

    def test_equity_is_rebased_after_date_slice(self):
        frame = pd.DataFrame(
            {"strategy_return": [0.10], "market_return": [0.10]},
            index=pd.to_datetime(["2024-01-03"]),
        )
        rebased = recompute_equity(frame)
        self.assertAlmostEqual(rebased.iloc[-1]["strategy_equity"], 1.10)


if __name__ == "__main__":
    unittest.main()
