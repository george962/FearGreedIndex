#!/usr/bin/env python3
"""Anchored walk-forward validation and probability calibration for FearGreedIndex."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_common import (  # noqa: E402
    load_context,
    read_json,
    replay_with_outcomes,
    strategy_version,
)



def _json_safe(value):
    """Convert NumPy/Pandas values and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "strategy_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports",
    )
    parser.add_argument("--skip-yahoo-fallback", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return non-zero when the configured acceptance checks fail",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
    )
    return parser.parse_args()


def binary_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    clipped = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    target = y.astype(float)
    return float(
        -np.mean(
            target * np.log(clipped)
            + (1.0 - target) * np.log(1.0 - clipped)
        )
    )


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p.astype(float) - y.astype(float)) ** 2))


def _group_key(row: pd.Series, columns: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")) for column in columns)


def _fit_table(
    train: pd.DataFrame,
    columns: list[str],
    *,
    global_probability: float,
    shrinkage_strength: float,
) -> dict[tuple[str, ...], dict[str, float]]:
    table: dict[tuple[str, ...], dict[str, float]] = {}
    if train.empty:
        return table

    grouped = train.groupby(columns, dropna=False)
    for keys, frame in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        y = (pd.to_numeric(frame["forward_5d"], errors="coerce") > 0).astype(float)
        n = int(len(y))
        wins = float(y.sum())
        probability = (
            wins + shrinkage_strength * global_probability
        ) / (n + shrinkage_strength)
        table[tuple(str(value) for value in keys)] = {
            "n": n,
            "probability": float(probability),
        }
    return table


def fit_calibrator(
    train: pd.DataFrame,
    *,
    minimum_group_sample: int,
    shrinkage_strength: float,
) -> dict[str, Any]:
    usable = train[pd.to_numeric(train["forward_5d"], errors="coerce").notna()].copy()
    if usable.empty:
        raise ValueError("Training fold has no completed 5-day outcomes")

    y = (pd.to_numeric(usable["forward_5d"], errors="coerce") > 0).astype(float)
    global_probability = float(y.mean())

    levels = [
        ["action", "market_regime", "timing_side"],
        ["action", "market_regime"],
        ["action"],
        ["timing_side"],
    ]
    tables = {
        "|".join(columns): _fit_table(
            usable,
            columns,
            global_probability=global_probability,
            shrinkage_strength=shrinkage_strength,
        )
        for columns in levels
    }

    return {
        "global_probability": global_probability,
        "minimum_group_sample": int(minimum_group_sample),
        "levels": levels,
        "tables": tables,
    }


def calibrated_probability(
    row: pd.Series,
    calibrator: dict[str, Any],
) -> tuple[float, str, int]:
    minimum = int(calibrator["minimum_group_sample"])

    for columns in calibrator["levels"]:
        name = "|".join(columns)
        key = _group_key(row, columns)
        item = calibrator["tables"][name].get(key)
        if item and int(item["n"]) >= minimum:
            return float(item["probability"]), name, int(item["n"])

    return (
        float(calibrator["global_probability"]),
        "global",
        0,
    )


def evaluate_fold(
    history: pd.DataFrame,
    fold: dict[str, Any],
    *,
    minimum_group_sample: int,
    shrinkage_strength: float,
    minimum_test_rows: int,
    minimum_relative_brier_improvement: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    dates = pd.to_datetime(history["decision_date"], errors="coerce")
    train_end = pd.Timestamp(fold["train_end"])
    test_start = pd.Timestamp(fold["test_start"])
    test_end = pd.Timestamp(fold["test_end"])

    train = history.loc[dates <= train_end].copy()
    test = history.loc[(dates >= test_start) & (dates <= test_end)].copy()
    test = test[pd.to_numeric(test["forward_5d"], errors="coerce").notna()].copy()

    calibrator = fit_calibrator(
        train,
        minimum_group_sample=minimum_group_sample,
        shrinkage_strength=shrinkage_strength,
    )

    probabilities = []
    sources = []
    source_samples = []
    for _, row in test.iterrows():
        probability, source, sample = calibrated_probability(row, calibrator)
        probabilities.append(probability)
        sources.append(source)
        source_samples.append(sample)

    test["predicted_p_up_5d"] = probabilities
    test["calibration_source"] = sources
    test["calibration_sample"] = source_samples
    test["fold"] = str(fold["name"])

    if test.empty:
        summary = {
            "fold": str(fold["name"]),
            "train_rows": int(len(train)),
            "test_rows": 0,
            "status": "INSUFFICIENT TEST DATA",
        }
        return summary, test

    y = (pd.to_numeric(test["forward_5d"], errors="coerce") > 0).astype(int).to_numpy()
    p = test["predicted_p_up_5d"].to_numpy(float)
    baseline_p = np.full(len(test), float(calibrator["global_probability"]))

    brier = brier_score(y, p)
    baseline_brier = brier_score(y, baseline_p)
    relative_brier_improvement = (
        0.0
        if baseline_brier <= 0
        else 1.0 - brier / baseline_brier
    )

    test_return = pd.to_numeric(test["forward_5d"], errors="coerce")
    overall_mean = float(test_return.mean())
    overall_win_rate = float((test_return > 0).mean())

    buy = test[test["action"].eq("BUY GRADUALLY")]
    wait = test[test["action"].eq("WAIT ON BUYING")]

    buy_mean = (
        float(pd.to_numeric(buy["forward_5d"], errors="coerce").mean())
        if not buy.empty
        else math.nan
    )
    wait_mean = (
        float(pd.to_numeric(wait["forward_5d"], errors="coerce").mean())
        if not wait.empty
        else math.nan
    )

    enough_rows = len(test) >= minimum_test_rows
    calibration_ok = (
        relative_brier_improvement >= minimum_relative_brier_improvement
    )

    status = "PASS" if enough_rows and calibration_ok else "REVIEW"

    summary = {
        "fold": str(fold["name"]),
        "train_end": str(fold["train_end"]),
        "test_start": str(fold["test_start"]),
        "test_end": str(fold["test_end"]),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "actual_up_rate_5d": overall_win_rate,
        "average_return_5d": overall_mean,
        "brier_score": brier,
        "baseline_brier_score": baseline_brier,
        "relative_brier_improvement": float(relative_brier_improvement),
        "log_loss": binary_log_loss(y, p),
        "mean_predicted_probability": float(np.mean(p)),
        "calibration_gap": float(np.mean(p) - np.mean(y)),
        "buy_gradually_rows": int(len(buy)),
        "buy_gradually_average_5d": buy_mean,
        "buy_gradually_excess_vs_fold": (
            buy_mean - overall_mean if math.isfinite(buy_mean) else math.nan
        ),
        "wait_rows": int(len(wait)),
        "wait_average_5d": wait_mean,
        "wait_excess_vs_fold": (
            wait_mean - overall_mean if math.isfinite(wait_mean) else math.nan
        ),
        "status": status,
    }
    return summary, test


def main() -> int:
    args = parse_args()
    raw_config = read_json(args.config)
    validation = raw_config.get("validation", {})
    folds = validation.get("folds", [])
    if not folds:
        raise SystemExit("config.json has no validation.folds")

    context = load_context(
        args.config,
        allow_yahoo=not args.skip_yahoo_fallback,
    )
    history = replay_with_outcomes(
        context,
        progress_every=args.progress_every,
    )

    summaries = []
    predictions = []

    for fold in folds:
        summary, predicted = evaluate_fold(
            history,
            fold,
            minimum_group_sample=int(
                validation.get("minimum_group_sample", 20)
            ),
            shrinkage_strength=float(
                validation.get("shrinkage_strength", 20.0)
            ),
            minimum_test_rows=int(
                validation.get("minimum_test_rows", 60)
            ),
            minimum_relative_brier_improvement=float(
                validation.get(
                    "minimum_relative_brier_improvement",
                    0.0,
                )
            ),
        )
        summaries.append(summary)
        if not predicted.empty:
            predictions.append(predicted)

    summary_frame = pd.DataFrame(summaries)
    prediction_frame = (
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "walk_forward_summary.csv"
    predictions_csv = args.output_dir / "walk_forward_predictions.csv"
    report_json = args.output_dir / "walk_forward.json"

    summary_frame.to_csv(summary_csv, index=False)
    prediction_frame.to_csv(predictions_csv, index=False)

    report = {
        "strategy_version": strategy_version(args.manifest),
        "data_source": context.data_source,
        "folds": summaries,
        "all_folds_pass": bool(
            len(summary_frame)
            and summary_frame["status"].eq("PASS").all()
        ),
    }
    report_json.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(summary_frame.to_string(index=False))
    print(f"\nWrote {summary_csv}")
    print(f"Wrote {predictions_csv}")
    print(f"Wrote {report_json}")

    if args.strict and not report["all_folds_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
