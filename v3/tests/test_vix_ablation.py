from __future__ import annotations

import unittest

import pandas as pd

from v3.evaluation.vix_ablation import (
    compare_metric_frames,
    summarize_lane_improvements,
    vix_family_decision,
)


class VixAblationTests(unittest.TestCase):
    @staticmethod
    def _metric_rows(
        experiment_id: str,
        model_name: str,
        *,
        brier: float,
        return_spearman: float,
        drawdown_spearman: float,
        sample_prefix: str = "same",
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for fold_index, fold in enumerate(("2024", "2025", "2026_ytd"), start=1):
            for horizon in (5, 20, 60):
                sample = f"{sample_prefix}-class-{fold}-{horizon}"
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "model_name": model_name,
                        "fold": fold,
                        "target_type": "classification",
                        "target": f"p_up_{horizon}d",
                        "horizon": horizon,
                        "sample_sha256": sample,
                        "brier_score": brier + fold_index * 0.001,
                        "relative_brier_improvement": 0.0,
                    }
                )
                sample = f"{sample_prefix}-return-{fold}-{horizon}"
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "model_name": model_name,
                        "fold": fold,
                        "target_type": "return_regression",
                        "target": f"return_{horizon}d",
                        "horizon": horizon,
                        "sample_sha256": sample,
                        "spearman_rank_correlation": return_spearman,
                        "rmse": 0.03,
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
                    "sample_sha256": f"{sample_prefix}-dd-{fold}-20",
                    "spearman_rank_correlation": drawdown_spearman,
                    "rmse": 0.02,
                }
            )
        return rows

    def test_comparison_uses_correct_metric_directions(self) -> None:
        baseline = pd.DataFrame(
            self._metric_rows(
                "EXP-003",
                "gb",
                brier=0.20,
                return_spearman=0.10,
                drawdown_spearman=0.05,
            )
        )
        vix = baseline.copy()
        class_mask = vix["target_type"].eq("classification")
        return_mask = vix["target_type"].eq("return_regression")
        dd_mask = vix["target_type"].eq("drawdown_regression")
        vix.loc[class_mask, "brier_score"] -= 0.01
        vix.loc[class_mask, "relative_brier_improvement"] += 0.02
        vix.loc[return_mask, "spearman_rank_correlation"] += 0.05
        vix.loc[dd_mask, "spearman_rank_correlation"] += 0.03
        comparison = compare_metric_frames(baseline, vix)
        self.assertTrue((comparison["primary_improvement"] > 0.0).all())
        self.assertTrue(comparison["vix_primary_improved"].all())

    def test_sample_hash_mismatch_is_rejected(self) -> None:
        baseline = pd.DataFrame(
            self._metric_rows(
                "EXP-003",
                "gb",
                brier=0.20,
                return_spearman=0.10,
                drawdown_spearman=0.05,
            )
        )
        vix = baseline.copy()
        vix.loc[0, "sample_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "different realized-date samples"):
            compare_metric_frames(baseline, vix)

    def test_family_is_retained_only_with_two_robust_lanes_in_both_models(self) -> None:
        rows: list[dict[str, object]] = []
        for experiment_id, model_name in (("EXP-003", "gb"), ("EXP-004", "rf")):
            for target_type in (
                "classification",
                "return_regression",
                "drawdown_regression",
            ):
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "model_name": model_name,
                        "target_type": target_type,
                        "primary_metric": (
                            "brier_score"
                            if target_type == "classification"
                            else "spearman_rank_correlation"
                        ),
                        "aggregate_primary_improvement": (
                            0.01 if target_type != "drawdown_regression" else -0.01
                        ),
                        "improved_folds": 2 if target_type != "drawdown_regression" else 1,
                        "total_folds": 3,
                        "robust_improvement": target_type != "drawdown_regression",
                    }
                )
        decision = vix_family_decision(pd.DataFrame(rows))
        self.assertTrue(decision["retain_vix"])
        self.assertEqual(decision["robust_lane_count"], 2)

        rows[-2]["robust_improvement"] = False  # EXP-004 return lane
        decision = vix_family_decision(pd.DataFrame(rows))
        self.assertFalse(decision["retain_vix"])
        self.assertEqual(decision["robust_lane_count"], 1)

    def test_lane_summary_requires_two_positive_folds(self) -> None:
        baseline = pd.DataFrame(
            self._metric_rows(
                "EXP-003",
                "gb",
                brier=0.20,
                return_spearman=0.10,
                drawdown_spearman=0.05,
            )
        )
        vix = baseline.copy()
        class_mask = vix["target_type"].eq("classification")
        vix.loc[class_mask, "brier_score"] -= 0.01
        comparison = compare_metric_frames(baseline, vix)
        lanes = summarize_lane_improvements(comparison)
        classification = lanes.loc[lanes["target_type"] == "classification"].iloc[0]
        self.assertTrue(bool(classification["robust_improvement"]))
        self.assertEqual(int(classification["improved_folds"]), 3)


if __name__ == "__main__":
    unittest.main()
