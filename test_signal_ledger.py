import unittest

import pandas as pd

from scripts.signal_ledger import (
    append_prediction,
    prediction_hash,
    update_matured_outcomes,
)


def sample_prediction():
    return {
        "decision_date": "2026-08-17",
        "market_date": "2026-08-17",
        "fear_greed": 59.97,
        "fg_change_5": -4.42,
        "market_regime": "near_high_uptrend",
        "market_extension": "Moderate",
        "action": "HOLD / NO EXTRA BUYING",
        "confidence": "Low",
        "sizing_tier": "Stay at your baseline",
        "sizing_label": "Normal plan only, no discretionary add",
        "timing_action": "NEUTRAL / NO TIMING EDGE",
        "timing_side": "NEUTRAL",
        "timing_score": 30,
        "timing_confirmation_count": 0,
        "timing_confirmation_total": 0,
        "analog_sample": 20,
        "regime_baseline_sample": 40,
        "win_rate_5d": 0.55,
        "average_5d": 0.004,
        "average_20d": 0.01,
        "regime_baseline_5d": 0.003,
        "excess_5d": 0.001,
        "excess_ci_low_5d": -0.002,
        "excess_ci_high_5d": 0.004,
        "average_drawdown_20d": -0.03,
        "strategy_version": "feargreed-v2.0.0",
        "config_sha256": "abc",
        "data_source": "test",
    }


class SignalLedgerTests(unittest.TestCase):
    def test_identical_prediction_is_not_duplicated(self):
        empty = pd.DataFrame()
        first, appended = append_prediction(
            empty,
            sample_prediction(),
            recorded_at_utc="2026-08-18T00:00:00+00:00",
        )
        self.assertTrue(appended)
        second, appended = append_prediction(
            first,
            sample_prediction(),
            recorded_at_utc="2026-08-18T01:00:00+00:00",
        )
        self.assertFalse(appended)
        self.assertEqual(len(second), 1)

    def test_changed_prediction_becomes_new_revision(self):
        prediction = sample_prediction()
        first, _ = append_prediction(
            pd.DataFrame(),
            prediction,
            recorded_at_utc="2026-08-18T00:00:00+00:00",
        )
        changed = dict(prediction)
        changed["action"] = "BUY GRADUALLY"
        second, appended = append_prediction(
            first,
            changed,
            recorded_at_utc="2026-08-18T01:00:00+00:00",
        )
        self.assertTrue(appended)
        self.assertEqual(len(second), 2)
        self.assertNotEqual(
            prediction_hash(prediction),
            prediction_hash(changed),
        )

    def test_outcomes_are_filled_without_changing_prediction_hash(self):
        prediction = sample_prediction()
        ledger, _ = append_prediction(
            pd.DataFrame(),
            prediction,
            recorded_at_utc="2026-08-18T00:00:00+00:00",
        )
        original_hash = ledger.iloc[0]["prediction_sha256"]

        events = pd.DataFrame(
            {
                "signal_date": [pd.Timestamp("2026-08-17")],
                "entry_date": [pd.Timestamp("2026-08-18")],
                "entry_price": [100.0],
                "forward_5d": [0.03],
                "forward_20d": [0.08],
                "max_drawdown_20d": [-0.02],
            }
        )
        updated, changed = update_matured_outcomes(
            ledger,
            events,
            updated_at_utc="2026-09-20T00:00:00+00:00",
        )
        self.assertEqual(changed, 1)
        self.assertEqual(
            updated.iloc[0]["prediction_sha256"],
            original_hash,
        )
        self.assertEqual(updated.iloc[0]["realized_5d"], "0.03")


if __name__ == "__main__":
    unittest.main()
