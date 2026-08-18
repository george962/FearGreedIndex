from __future__ import annotations

import unittest

import pandas as pd

from v3.evaluation.tournament import (
    build_tournament_report,
    summarize_experiments,
)


class TournamentTests(unittest.TestCase):
    @staticmethod
    def _full_candidate_rows(
        experiment_id: str,
        model_name: str,
        relative_brier: float,
        return_spearman: float,
        drawdown_spearman: float,
        error_scale: float,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for fold in ("2024", "2025", "2026_ytd"):
            for horizon in (5, 20, 60):
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "model_name": model_name,
                        "fold": fold,
                        "target_type": "classification",
                        "target": f"p_up_{horizon}d",
                        "horizon": horizon,
                        "brier_score": 0.20 * error_scale,
                        "log_loss": 0.60 * error_scale,
                        "expected_calibration_error": 0.10 * error_scale,
                        "relative_brier_improvement": relative_brier,
                    }
                )
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "model_name": model_name,
                        "fold": fold,
                        "target_type": "return_regression",
                        "target": f"return_{horizon}d",
                        "horizon": horizon,
                        "mae": 0.02 * error_scale,
                        "rmse": 0.03 * error_scale,
                        "spearman_rank_correlation": return_spearman,
                    }
                )
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "model_name": model_name,
                    "fold": fold,
                    "target_type": "drawdown_regression",
                    "target": "drawdown_20d",
                    "horizon": 20,
                    "mae": 0.015 * error_scale,
                    "rmse": 0.02 * error_scale,
                    "spearman_rank_correlation": drawdown_spearman,
                }
            )
        return rows

    def test_first_place_can_still_fail_promotion(self) -> None:
        rows = self._full_candidate_rows(
            "EXP-A", "weak_but_best", -0.05, -0.02, -0.01, 0.8
        )
        rows += self._full_candidate_rows(
            "EXP-B", "weaker", -0.20, -0.10, -0.05, 1.2
        )
        scoreboard = summarize_experiments(pd.DataFrame(rows))
        best = scoreboard.sort_values("overall_full_candidate_rank").iloc[0]
        self.assertEqual(best["experiment_id"], "EXP-A")
        self.assertFalse(bool(best["promotion_ready"]))
        self.assertIn("classification_worse_than_base_rate", best["promotion_gate_reason"])
        report = build_tournament_report(scoreboard)
        self.assertEqual(report["best_ranked_full_candidate"], "EXP-A")
        self.assertIsNone(report["champion"])

    def test_strong_robust_full_candidate_can_pass_gate(self) -> None:
        rows = self._full_candidate_rows(
            "EXP-GOOD", "good", 0.10, 0.20, 0.15, 0.8
        )
        rows += self._full_candidate_rows(
            "EXP-WEAK", "weak", -0.10, -0.05, -0.02, 1.2
        )
        scoreboard = summarize_experiments(pd.DataFrame(rows))
        good = scoreboard.set_index("experiment_id").loc["EXP-GOOD"]
        self.assertTrue(bool(good["promotion_ready"]))
        report = build_tournament_report(scoreboard)
        self.assertEqual(report["champion"], "EXP-GOOD")

    def test_component_only_model_never_passes_full_candidate_gate(self) -> None:
        rows = []
        for fold in ("2024", "2025", "2026_ytd"):
            rows.append(
                {
                    "experiment_id": "EXP-CLASS",
                    "model_name": "classification_only",
                    "fold": fold,
                    "target_type": "classification",
                    "target": "p_up_20d",
                    "horizon": 20,
                    "brier_score": 0.10,
                    "log_loss": 0.30,
                    "expected_calibration_error": 0.05,
                    "relative_brier_improvement": 0.20,
                }
            )
        scoreboard = summarize_experiments(pd.DataFrame(rows))
        row = scoreboard.iloc[0]
        self.assertFalse(bool(row["full_candidate"]))
        self.assertFalse(bool(row["promotion_ready"]))
        self.assertEqual(row["promotion_gate_reason"], "incomplete_output_interface")


if __name__ == "__main__":
    unittest.main()
