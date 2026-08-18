#!/usr/bin/env python3
"""Build the V3-010 model tournament without confusing rank with readiness."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v3.evaluation.walk_forward import (
    DEFAULT_METRICS as DEFAULT_COMMON_METRICS,
    run_common_evaluation,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_CSV = ROOT / "v3" / "reports" / "tournament.csv"
DEFAULT_OUTPUT_JSON = ROOT / "v3" / "reports" / "tournament.json"

FULL_REQUIRED_TARGET_TYPES = {
    "classification",
    "return_regression",
    "drawdown_regression",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_COMMON_METRICS)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--regenerate-common-evaluation",
        action="store_true",
        help="Regenerate V3-009 outputs before building the tournament.",
    )
    return parser.parse_args()


def _mean(group: pd.DataFrame, column: str) -> float | None:
    if column not in group:
        return None
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _positive_cells(group: pd.DataFrame, column: str) -> int:
    if column not in group:
        return 0
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    return int((values > 0.0).sum())


def _positive_folds(group: pd.DataFrame, column: str) -> int:
    if column not in group:
        return 0
    count = 0
    for _, fold_group in group.groupby("fold", sort=True):
        values = pd.to_numeric(fold_group[column], errors="coerce").dropna()
        if len(values) and float(values.mean()) > 0.0:
            count += 1
    return count


def summarize_experiments(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (experiment_id, model_name), experiment in metrics.groupby(
        ["experiment_id", "model_name"], sort=True
    ):
        target_types = set(experiment["target_type"].dropna().astype(str))
        row: dict[str, Any] = {
            "experiment_id": experiment_id,
            "model_name": model_name,
            "full_candidate": FULL_REQUIRED_TARGET_TYPES.issubset(target_types),
        }

        classification = experiment.loc[experiment["target_type"] == "classification"]
        row.update(
            {
                "classification_cells": int(len(classification)),
                "classification_mean_brier": _mean(classification, "brier_score"),
                "classification_mean_log_loss": _mean(classification, "log_loss"),
                "classification_mean_ece": _mean(
                    classification, "expected_calibration_error"
                ),
                "classification_mean_relative_brier_improvement": _mean(
                    classification, "relative_brier_improvement"
                ),
                "classification_positive_relative_brier_cells": _positive_cells(
                    classification, "relative_brier_improvement"
                ),
                "classification_positive_relative_brier_folds": _positive_folds(
                    classification, "relative_brier_improvement"
                ),
            }
        )

        returns = experiment.loc[experiment["target_type"] == "return_regression"]
        row.update(
            {
                "return_cells": int(len(returns)),
                "return_mean_mae": _mean(returns, "mae"),
                "return_mean_rmse": _mean(returns, "rmse"),
                "return_mean_spearman": _mean(
                    returns, "spearman_rank_correlation"
                ),
                "return_positive_spearman_cells": _positive_cells(
                    returns, "spearman_rank_correlation"
                ),
                "return_positive_spearman_folds": _positive_folds(
                    returns, "spearman_rank_correlation"
                ),
            }
        )

        drawdown = experiment.loc[experiment["target_type"] == "drawdown_regression"]
        row.update(
            {
                "drawdown_cells": int(len(drawdown)),
                "drawdown_mean_mae": _mean(drawdown, "mae"),
                "drawdown_mean_rmse": _mean(drawdown, "rmse"),
                "drawdown_mean_spearman": _mean(
                    drawdown, "spearman_rank_correlation"
                ),
                "drawdown_positive_spearman_folds": _positive_folds(
                    drawdown, "spearman_rank_correlation"
                ),
            }
        )
        rows.append(row)

    scoreboard = pd.DataFrame(rows).sort_values("experiment_id").reset_index(drop=True)
    return add_lane_ranks(scoreboard)


def _normalized_rank(
    frame: pd.DataFrame,
    columns: list[tuple[str, bool]],
    eligible: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    indexes = frame.index[eligible]
    scores = pd.Series(np.nan, index=frame.index, dtype=float)
    ranks = pd.Series(np.nan, index=frame.index, dtype=float)
    if len(indexes) == 0:
        return scores, ranks

    metric_ranks: list[pd.Series] = []
    for column, ascending in columns:
        values = pd.to_numeric(frame.loc[indexes, column], errors="coerce")
        metric_ranks.append(values.rank(method="average", ascending=ascending))
    raw_score = pd.concat(metric_ranks, axis=1).mean(axis=1)
    lane_rank = raw_score.rank(method="min", ascending=True)
    count = len(indexes)
    normalized = (
        (lane_rank - 1.0) / (count - 1.0) if count > 1 else lane_rank * 0.0
    )
    scores.loc[indexes] = normalized
    ranks.loc[indexes] = lane_rank
    return scores, ranks


def add_lane_ranks(scoreboard: pd.DataFrame) -> pd.DataFrame:
    frame = scoreboard.copy()

    class_ok = frame["classification_cells"].gt(0)
    frame["classification_lane_score"], frame["classification_rank"] = _normalized_rank(
        frame,
        [
            ("classification_mean_brier", True),
            ("classification_mean_log_loss", True),
            ("classification_mean_ece", True),
        ],
        class_ok,
    )

    return_ok = frame["return_cells"].gt(0)
    frame["return_lane_score"], frame["return_rank"] = _normalized_rank(
        frame,
        [
            ("return_mean_mae", True),
            ("return_mean_rmse", True),
            ("return_mean_spearman", False),
        ],
        return_ok,
    )

    drawdown_ok = frame["drawdown_cells"].gt(0)
    frame["drawdown_lane_score"], frame["drawdown_rank"] = _normalized_rank(
        frame,
        [
            ("drawdown_mean_mae", True),
            ("drawdown_mean_rmse", True),
            ("drawdown_mean_spearman", False),
        ],
        drawdown_ok,
    )

    frame["overall_full_candidate_score"] = np.nan
    full = frame["full_candidate"].astype(bool)
    if full.any():
        frame.loc[full, "overall_full_candidate_score"] = frame.loc[
            full,
            [
                "classification_lane_score",
                "return_lane_score",
                "drawdown_lane_score",
            ],
        ].mean(axis=1)
        frame.loc[full, "overall_full_candidate_rank"] = frame.loc[
            full, "overall_full_candidate_score"
        ].rank(method="min", ascending=True)
    else:
        frame["overall_full_candidate_rank"] = np.nan

    promotion: list[bool] = []
    reasons: list[str] = []
    for _, row in frame.iterrows():
        failures: list[str] = []
        if not bool(row["full_candidate"]):
            failures.append("incomplete_output_interface")
        else:
            if not (
                pd.notna(row["classification_mean_relative_brier_improvement"])
                and float(row["classification_mean_relative_brier_improvement"]) > 0.0
            ):
                failures.append("classification_worse_than_base_rate")
            if int(row["classification_positive_relative_brier_folds"]) < 2:
                failures.append("classification_not_robust_across_folds")
            if not (
                pd.notna(row["return_mean_spearman"])
                and float(row["return_mean_spearman"]) > 0.0
            ):
                failures.append("return_rank_correlation_nonpositive")
            if int(row["return_positive_spearman_folds"]) < 2:
                failures.append("return_signal_not_robust_across_folds")
            if not (
                pd.notna(row["drawdown_mean_spearman"])
                and float(row["drawdown_mean_spearman"]) > 0.0
            ):
                failures.append("drawdown_rank_correlation_nonpositive")
            if int(row["drawdown_positive_spearman_folds"]) < 2:
                failures.append("drawdown_signal_not_robust_across_folds")
        promotion.append(len(failures) == 0)
        reasons.append("PASS" if not failures else ";".join(failures))

    frame["promotion_ready"] = promotion
    frame["promotion_gate_reason"] = reasons
    frame["status"] = np.where(frame["promotion_ready"], "PROMOTION_READY", "NOT_PROMOTION_READY")
    return frame


def build_tournament_report(scoreboard: pd.DataFrame) -> dict[str, Any]:
    full = scoreboard.loc[scoreboard["full_candidate"].astype(bool)].copy()
    full = full.sort_values(
        ["overall_full_candidate_rank", "experiment_id"], na_position="last"
    )
    best_full = None if full.empty else str(full.iloc[0]["experiment_id"])
    ready = scoreboard.loc[scoreboard["promotion_ready"].astype(bool)]
    champion = None
    if len(ready) == 1:
        champion = str(ready.iloc[0]["experiment_id"])
    elif len(ready) > 1:
        eligible_full = ready.loc[ready["full_candidate"].astype(bool)].sort_values(
            ["overall_full_candidate_rank", "experiment_id"]
        )
        champion = None if eligible_full.empty else str(eligible_full.iloc[0]["experiment_id"])

    return {
        "status": "TOURNAMENT_COMPLETE",
        "scoreboard_version": "v3-tournament-001",
        "experiments": scoreboard["experiment_id"].tolist(),
        "best_ranked_full_candidate": best_full,
        "promotion_ready_experiments": ready["experiment_id"].tolist(),
        "champion": champion,
        "champion_selected": champion is not None,
        "decision_policy_status": "NOT_STARTED_V3_016",
        "trading_score_status": "DEFERRED_UNTIL_DECISION_POLICY",
        "interpretation": (
            "Rank compares tested models. Promotion readiness is a separate absolute gate; "
            "first place does not imply validated predictive edge."
        ),
    }


def run_tournament(
    metrics_path: Path = DEFAULT_COMMON_METRICS,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    regenerate_common_evaluation: bool = False,
) -> dict[str, Any]:
    if regenerate_common_evaluation or not metrics_path.exists():
        run_common_evaluation()
    metrics = pd.read_csv(metrics_path)
    scoreboard = summarize_experiments(metrics)
    report = build_tournament_report(scoreboard)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    scoreboard.to_csv(output_csv, index=False)
    output_json.write_text(
        json.dumps(
            {
                **report,
                "scoreboard": scoreboard.replace({np.nan: None}).to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(scoreboard.to_string(index=False))
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    args = parse_args()
    report = run_tournament(
        metrics_path=args.metrics,
        output_csv=args.output_csv,
        output_json=args.output_json,
        regenerate_common_evaluation=args.regenerate_common_evaluation,
    )
    return 0 if report["status"] == "TOURNAMENT_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
