#!/usr/bin/env python3
"""Extend v3-features-001 with point-in-time QQQ/SPY relative-strength features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_FEATURES = ROOT / "v3" / "data" / "features_daily.parquet"
DEFAULT_BASE_REGISTRY = ROOT / "v3" / "features" / "feature_registry.json"
DEFAULT_FAMILY_REGISTRY = ROOT / "v3" / "features" / "relative_strength_features.json"
DEFAULT_SOURCE = ROOT / "v3" / "data" / "qqq_spy_daily.csv.gz"
DEFAULT_OUTPUT = ROOT / "v3" / "data" / "features_daily_relative_strength.parquet"
DEFAULT_OUTPUT_REGISTRY = ROOT / "v3" / "reports" / "feature_registry_relative_strength.json"
DEFAULT_REPORT = ROOT / "v3" / "reports" / "relative_strength_features_missingness.json"

RELATIVE_STRENGTH_FEATURES = (
    "qqq_return_1",
    "qqq_return_5",
    "qqq_return_20",
    "qqq_return_60",
    "qqq_spy_relative_return_1",
    "qqq_spy_relative_return_5",
    "qqq_spy_relative_return_20",
    "qqq_spy_relative_return_60",
    "qqq_spy_ratio_distance_ma_20",
    "qqq_spy_ratio_distance_ma_60",
    "qqq_spy_relative_acceleration_5_vs_20",
    "qqq_spy_relative_acceleration_20_vs_60",
)
BASE_METADATA = {
    "decision_date",
    "fear_greed_date",
    "open",
    "high",
    "low",
    "close",
}
EXPANDED_METADATA = BASE_METADATA | {"relative_strength_date"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-features", type=Path, default=DEFAULT_BASE_FEATURES)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-registry", type=Path, default=DEFAULT_OUTPUT_REGISTRY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_base_features(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow").copy()
    frame["decision_date"] = pd.to_datetime(
        frame["decision_date"], errors="raise"
    ).dt.normalize()
    if frame["decision_date"].duplicated().any():
        raise ValueError("Baseline feature table has duplicate decision dates")
    return frame.sort_values("decision_date").reset_index(drop=True)


def load_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "qqq_close", "spy_close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"QQQ/SPY snapshot missing columns: {missing}")
    frame = frame[["date", "qqq_close", "spy_close"]].copy()
    frame["relative_strength_date"] = pd.to_datetime(
        frame.pop("date"), errors="raise"
    ).dt.normalize()
    for column in ("qqq_close", "spy_close"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        if (frame[column] <= 0.0).any():
            raise ValueError(f"{column} contains non-positive values")
    frame = frame.sort_values("relative_strength_date").reset_index(drop=True)
    if frame["relative_strength_date"].duplicated().any():
        raise ValueError("QQQ/SPY snapshot contains duplicate dates")
    return frame


def _distance_from_mean(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 2)).mean()
    return (series / mean.replace(0.0, np.nan) - 1.0).replace(
        [np.inf, -np.inf], np.nan
    )


def build_source_history(source: pd.DataFrame) -> pd.DataFrame:
    history = source.sort_values("relative_strength_date").copy()
    qqq = history["qqq_close"]
    spy = history["spy_close"]

    for horizon in (1, 5, 20, 60):
        qqq_growth = qqq / qqq.shift(horizon)
        spy_growth = spy / spy.shift(horizon)
        history[f"qqq_return_{horizon}"] = qqq_growth - 1.0
        history[f"qqq_spy_relative_return_{horizon}"] = (
            qqq_growth / spy_growth.replace(0.0, np.nan) - 1.0
        )

    ratio = qqq / spy.replace(0.0, np.nan)
    history["qqq_spy_ratio_distance_ma_20"] = _distance_from_mean(ratio, 20)
    history["qqq_spy_ratio_distance_ma_60"] = _distance_from_mean(ratio, 60)
    history["qqq_spy_relative_acceleration_5_vs_20"] = (
        history["qqq_spy_relative_return_5"]
        - history["qqq_spy_relative_return_20"]
    )
    history["qqq_spy_relative_acceleration_20_vs_60"] = (
        history["qqq_spy_relative_return_20"]
        - history["qqq_spy_relative_return_60"]
    )
    return history[["relative_strength_date", *RELATIVE_STRENGTH_FEATURES]]


def build_expanded_features(
    base_features: pd.DataFrame,
    source: pd.DataFrame,
) -> pd.DataFrame:
    base = base_features.sort_values("decision_date").copy()
    history = build_source_history(source)
    expanded = pd.merge_asof(
        base,
        history,
        left_on="decision_date",
        right_on="relative_strength_date",
        direction="backward",
        allow_exact_matches=True,
    )
    future = (
        expanded["relative_strength_date"].notna()
        & expanded["relative_strength_date"].gt(expanded["decision_date"])
    )
    if future.any():
        raise AssertionError("Future QQQ/SPY source date joined to a decision")
    if expanded["decision_date"].duplicated().any():
        raise AssertionError("Expanded feature table has duplicate decision dates")
    return expanded


def build_output_registry(
    base_registry_path: Path = DEFAULT_BASE_REGISTRY,
    family_registry_path: Path = DEFAULT_FAMILY_REGISTRY,
) -> dict[str, object]:
    base = json.loads(base_registry_path.read_text(encoding="utf-8"))
    family = json.loads(family_registry_path.read_text(encoding="utf-8"))
    base_features = list(base.get("features", []))
    added_features = list(family.get("features", []))
    names = [item["name"] for item in [*base_features, *added_features]]
    if len(names) != len(set(names)):
        raise ValueError("Relative-strength registry contains duplicate feature names")
    return {
        "version": family["version"],
        "parent_version": family["parent_version"],
        "decision_time_convention": family["decision_time_convention"],
        "features": [*base_features, *added_features],
    }


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in EXPANDED_METADATA]


def run_build(
    *,
    base_features_path: Path = DEFAULT_BASE_FEATURES,
    source_path: Path = DEFAULT_SOURCE,
    output_path: Path = DEFAULT_OUTPUT,
    output_registry_path: Path = DEFAULT_OUTPUT_REGISTRY,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, object]:
    base = load_base_features(base_features_path)
    source = load_source(source_path)
    expanded = build_expanded_features(base, source)

    assert_frame_equal(
        expanded[base.columns].reset_index(drop=True),
        base.reset_index(drop=True),
        check_dtype=True,
    )

    registry = build_output_registry()
    expected = [item["name"] for item in registry["features"]]
    actual = model_feature_columns(expanded)
    if set(expected) != set(actual):
        raise ValueError(
            "Generated relative-strength registry does not match feature table; "
            f"missing={sorted(set(actual).difference(expected))}, "
            f"stale={sorted(set(expected).difference(actual))}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_registry_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_parquet(output_path, index=False, engine="pyarrow")
    output_registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8"
    )

    report: dict[str, object] = {
        "feature_version": registry["version"],
        "rows": int(len(expanded)),
        "start": expanded["decision_date"].min().date().isoformat(),
        "end": expanded["decision_date"].max().date().isoformat(),
        "baseline_feature_count": int(len(actual) - len(RELATIVE_STRENGTH_FEATURES)),
        "relative_strength_feature_count": int(len(RELATIVE_STRENGTH_FEATURES)),
        "total_feature_count": int(len(actual)),
        "future_source_rows": int(
            (
                expanded["relative_strength_date"].notna()
                & expanded["relative_strength_date"].gt(expanded["decision_date"])
            ).sum()
        ),
        "missingness": {
            column: {
                "missing_rows": int(expanded[column].isna().sum()),
                "missing_fraction": float(expanded[column].isna().mean()),
            }
            for column in ("relative_strength_date", *RELATIVE_STRENGTH_FEATURES)
        },
        "input_sha256": {
            "baseline_features": _sha256(base_features_path),
            "qqq_spy_snapshot": _sha256(source_path),
        },
        "output_sha256": _sha256(output_path),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> int:
    args = parse_args()
    report = run_build(
        base_features_path=args.base_features,
        source_path=args.source,
        output_path=args.output,
        output_registry_path=args.output_registry,
        report_path=args.report,
    )
    print(
        f"Wrote {args.output} ({report['rows']} rows, "
        f"{report['total_feature_count']} features)"
    )
    print(f"Wrote {args.output_registry}")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
