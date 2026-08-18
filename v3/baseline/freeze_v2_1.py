#!/usr/bin/env python3
"""Generate the immutable v2.1 baseline package used by v3 comparisons.

The expensive historical decision replay is performed exactly once. The frozen
walk-forward evaluator and portfolio backtest both consume that same replay so
V3-001 preserves v2.1 semantics without duplicating the point-in-time engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest as v2_backtest  # noqa: E402
from scripts import strategy_validation as v2_validation  # noqa: E402
from scripts.research_common import (  # noqa: E402
    load_context,
    read_json,
    replay_with_outcomes,
    strategy_version,
)

DEFAULT_OUTPUT = ROOT / "reports" / "baseline_v2_1"
RUNTIME_FILES = (
    "FearGreed.py",
    "FearGreedHistory.py",
    "FearGreedMarketData.py",
    "backtest.py",
    "config.json",
    "strategy_manifest.json",
    "scripts/build_dashboard.py",
    "scripts/research_common.py",
    "scripts/strategy_validation.py",
)
REQUIRED_REPORTS = (
    "backtest_summary.json",
    "walk_forward_summary.csv",
    "action_scorecard.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def build_walk_forward_summary(
    history: pd.DataFrame,
    raw_config: dict[str, Any],
) -> pd.DataFrame:
    validation = raw_config.get("validation", {})
    folds = validation.get("folds", [])
    if not folds:
        raise ValueError("config.json has no validation.folds")

    summaries: list[dict[str, Any]] = []
    for fold in folds:
        summary, _ = v2_validation.evaluate_fold(
            history,
            fold,
            minimum_group_sample=int(validation.get("minimum_group_sample", 20)),
            shrinkage_strength=float(validation.get("shrinkage_strength", 20.0)),
            minimum_test_rows=int(validation.get("minimum_test_rows", 60)),
            minimum_relative_brier_improvement=float(
                validation.get("minimum_relative_brier_improvement", 0.0)
            ),
        )
        summaries.append(summary)
    return pd.DataFrame(summaries)


def build_backtest_summary(
    context: Any,
    history: pd.DataFrame,
    raw_config: dict[str, Any],
    manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    overlay_settings = raw_config.get("portfolio_backtest", {})
    daily = v2_backtest.build_daily_backtest(
        context.market,
        history,
        overlay_settings,
    )

    entry_dates = pd.to_datetime(history["entry_date"], errors="coerce").dropna()
    if entry_dates.empty:
        raise ValueError("No executable decision entry dates are available")
    evaluation_start = entry_dates.min()
    daily = daily.loc[daily.index >= evaluation_start]
    if daily.empty:
        raise ValueError("No market rows remain after baseline date filtering")
    daily = v2_backtest.recompute_equity(daily)

    strategy = v2_backtest.performance_metrics(
        daily,
        "strategy_return",
        "strategy_equity",
    )
    benchmark = v2_backtest.performance_metrics(
        daily,
        "market_return",
        "benchmark_equity",
    )

    summary: dict[str, Any] = {
        "strategy_version": strategy_version(manifest_path),
        "data_source": context.data_source,
        "start": daily.index.min().date().isoformat(),
        "end": daily.index.max().date().isoformat(),
        "strategy": strategy,
        "benchmark": benchmark,
        "annualized_return_difference": (
            strategy["annualized_return"] - benchmark["annualized_return"]
        ),
        "max_drawdown_difference": (
            strategy["max_drawdown"] - benchmark["max_drawdown"]
        ),
        "average_exposure": float(daily["exposure"].mean()),
        "maximum_exposure": float(daily["exposure"].max()),
        "minimum_exposure": float(daily["exposure"].min()),
        "total_turnover": float(daily["turnover"].sum()),
        **v2_backtest.holding_period_metrics(daily),
        "benchmark_relative_total_return": (
            strategy["total_return"] - benchmark["total_return"]
        ),
        "assumptions": {
            "cash_return": 0.0,
            "borrowing_cost": 0.0,
            "transaction_cost_bps_per_1x_turnover": float(
                overlay_settings.get("transaction_cost_bps_per_1x_turnover", 2.0)
            ),
            "note": (
                "This is a diagnostic exposure-overlay backtest. "
                "It is not a brokerage fill simulator."
            ),
        },
    }
    return daily, summary


def main() -> int:
    args = parse_args()
    config_path = ROOT / "config.json"
    manifest_path = ROOT / "strategy_manifest.json"
    raw_config = read_json(config_path)
    version = strategy_version(manifest_path)

    if raw_config.get("enable_tactical_sizing") is not False:
        raise SystemExit(
            "V3-001 requires the frozen v2.1 benchmark to keep tactical sizing disabled"
        )

    context = load_context(config_path, allow_yahoo=False)
    history = replay_with_outcomes(
        context,
        progress_every=args.progress_every,
    )
    if history.empty:
        raise SystemExit("Frozen v2.1 replay produced no historical decisions")

    walk_forward = build_walk_forward_summary(history, raw_config)
    _, backtest_summary = build_backtest_summary(
        context,
        history,
        raw_config,
        manifest_path,
    )
    scorecard = v2_backtest.action_scorecard(history)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    backtest_path = args.output_dir / "backtest_summary.json"
    walk_forward_path = args.output_dir / "walk_forward_summary.csv"
    scorecard_path = args.output_dir / "action_scorecard.csv"

    backtest_path.write_text(
        json.dumps(json_safe(backtest_summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    walk_forward.to_csv(walk_forward_path, index=False)
    scorecard.to_csv(scorecard_path, index=False)

    missing = [name for name in REQUIRED_REPORTS if not (args.output_dir / name).exists()]
    if missing:
        raise SystemExit(f"Baseline generation missing outputs: {missing}")

    file_hashes = {relative: sha256(ROOT / relative) for relative in RUNTIME_FILES}
    input_hashes = {
        "data/fear_greed_daily.csv": sha256(ROOT / "data" / "fear_greed_daily.csv"),
        "data/spx_daily.csv": sha256(ROOT / "data" / "spx_daily.csv"),
    }
    report_hashes = {
        name: sha256(args.output_dir / name)
        for name in REQUIRED_REPORTS
    }
    all_folds_pass = bool(
        len(walk_forward)
        and "status" in walk_forward
        and walk_forward["status"].eq("PASS").all()
    )

    freeze_manifest = {
        "strategy_version": version,
        "dataset_start": backtest_summary.get("start"),
        "dataset_end": backtest_summary.get("end"),
        "tactical_sizing_enabled": False,
        "walk_forward_all_folds_pass": all_folds_pass,
        "historical_decision_rows": int(len(history)),
        "runtime_sha256": file_hashes,
        "input_sha256": input_hashes,
        "report_sha256": report_hashes,
        "generator_sha256": sha256(Path(__file__).resolve()),
        "generation_command": "python v3/baseline/freeze_v2_1.py",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(freeze_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    readme = (
        f"# v2.1 Frozen Baseline\n\n"
        f"Strategy version: `{version}`\n\n"
        "This directory is the permanent benchmark package for v3 research. "
        "It is generated exclusively from checked-in data using the frozen v2.1 runtime.\n\n"
        f"Evaluation coverage: `{backtest_summary.get('start')}` through "
        f"`{backtest_summary.get('end')}`.\n\n"
        "Tactical sizing is explicitly required to remain disabled while this "
        "benchmark is generated. The walk-forward status is recorded as evidence, "
        "not used as a condition for whether the benchmark may be frozen.\n\n"
        "## Reproduce\n\n```bash\npython v3/baseline/freeze_v2_1.py\n```\n\n"
        "The hashes in `manifest.json` identify the exact runtime, inputs, generator, "
        "and reports. Any methodology change belongs in v3 rather than changing this benchmark.\n"
    )
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Frozen v2.1 baseline written to {args.output_dir}")
    print(f"Historical decisions: {len(history)}")
    print(f"Walk-forward all folds pass: {all_folds_pass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
