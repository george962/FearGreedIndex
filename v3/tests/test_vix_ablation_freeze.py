from __future__ import annotations

import unittest

import pandas as pd

from v3.evaluation.vix_ablation import ABLATION_AS_OF, freeze_model_dataset_asof


class VixAblationFreezeTests(unittest.TestCase):
    @staticmethod
    def _dataset() -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "decision_date": pd.to_datetime(
                    ["2026-08-01", "2026-08-02", "2026-08-19"]
                ),
                "entry_date": pd.to_datetime(
                    ["2026-08-03", "2026-08-03", "2026-08-20"]
                ),
                "feature_x": [1.0, 2.0, 3.0],
            }
        )
        known_dates = {
            5: ["2026-08-10", "2026-08-11", "2026-08-27"],
            20: ["2026-08-28", "2026-08-31", "2026-09-18"],
            60: ["2026-10-26", "2026-10-27", "2026-11-16"],
        }
        for horizon in (5, 20, 60):
            frame[f"forward_return_{horizon}d"] = [0.01, -0.02, 0.03]
            frame[f"forward_positive_{horizon}d"] = pd.array(
                [True, False, True], dtype="boolean"
            )
            frame[f"max_drawdown_{horizon}d"] = [-0.01, -0.03, -0.02]
            frame[f"_forward_{horizon}d_known_date"] = pd.to_datetime(
                known_dates[horizon]
            )
        frame["further_5pct_decline_20d"] = pd.array(
            [False, True, False], dtype="boolean"
        )
        return frame

    def test_post_cutoff_decisions_are_removed(self) -> None:
        frozen = freeze_model_dataset_asof(self._dataset(), ABLATION_AS_OF)
        self.assertEqual(len(frozen), 2)
        self.assertTrue((frozen["decision_date"] <= ABLATION_AS_OF).all())
        self.assertEqual(
            frozen["decision_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2026-08-01", "2026-08-02"],
        )

    def test_outcomes_known_by_cutoff_are_preserved(self) -> None:
        original = self._dataset()
        frozen = freeze_model_dataset_asof(original, ABLATION_AS_OF)
        self.assertEqual(frozen["forward_return_5d"].tolist(), [0.01, -0.02])
        self.assertEqual(
            frozen["forward_positive_5d"].astype("boolean").tolist(),
            [True, False],
        )
        self.assertEqual(frozen["max_drawdown_5d"].tolist(), [-0.01, -0.03])
        self.assertEqual(
            frozen["_forward_5d_known_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2026-08-10", "2026-08-11"],
        )

    def test_outcomes_maturing_after_cutoff_are_censored(self) -> None:
        frozen = freeze_model_dataset_asof(self._dataset(), ABLATION_AS_OF)
        for horizon in (20, 60):
            self.assertTrue(frozen[f"forward_return_{horizon}d"].isna().all())
            self.assertTrue(frozen[f"forward_positive_{horizon}d"].isna().all())
            self.assertTrue(frozen[f"max_drawdown_{horizon}d"].isna().all())
            self.assertTrue(frozen[f"_forward_{horizon}d_known_date"].isna().all())
        self.assertTrue(frozen["further_5pct_decline_20d"].isna().all())


if __name__ == "__main__":
    unittest.main()
