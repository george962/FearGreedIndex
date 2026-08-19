#!/usr/bin/env python3
"""Extend v3-features-001 with lagged broad-U.S.-dollar features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from v3.features.build_features import _rolling_percentile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_FEATURES = ROOT / "v3" / "data" / "features_daily.parquet"
DEFAULT_BASE_REGISTRY = ROOT / "v3" / "features" / "feature_registry.json"
DEFAULT_FAMILY_REGISTRY = ROOT / "v3" / "features" / "dollar_features.json"
DEFAULT_SOURCE = ROOT / "v3" / "data" / "dollar_daily.csv.gz"
DEFAULT_OUTPUT = ROOT / "v3" / "data" / "features_daily_dollar.parquet"
DEFAULT_OUTPUT_REGISTRY = ROOT / "v3" / "reports" / "feature_registry_dollar.json"
DEFAULT_REPORT = ROOT / "v3" / "reports" / "dollar_features_missingness.json"

DOLLAR_FEATURES = (
    "dollar_level",
    "dollar_return_1",
    "dollar_return_5",
    "dollar_return_20",
    "dollar_return_60",
    "dollar_percentile_60",
    "dollar_percentile_252",
    "dollar_distance_ma_20",
    "dollar_distance_ma_60",
    "dollar_acceleration_5_vs_20",
    "dollar_acceleration_20_vs_60",
)
BASE_METADATA = {"decision_date", "fear_greed_date", "open", "high", "low", "close"}
EXPANDED_METADATA = BASE_METADATA | {"dollar_observation_date", "dollar_available_date"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-features", type=Path, default=DEFAULT_BASE_FEATURES)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-registry", type=Path, default=DEFAULT_OUTPUT_REGISTRY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def load_base(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow").copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
    frame = frame.sort_values("decision_date").reset_index(drop=True)
    if frame["decision_date"].duplicated().any():
        raise ValueError("Baseline feature table has duplicate decision dates")
    return frame


def load_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "dollar_index"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dollar snapshot missing columns: {missing}")
    result = pd.DataFrame(
        {
            "dollar_observation_date": pd.to_datetime(frame["date"], errors="raise").dt.normalize(),
            "dollar_index": pd.to_numeric(frame["dollar_index"], errors="raise").astype(float),
        }
    ).sort_values("dollar_observation_date").reset_index(drop=True)
    if result.empty or result["dollar_observation_date"].duplicated().any():
        raise ValueError("Dollar snapshot is empty or has duplicate dates")
    if (result["dollar_index"] <= 0.0).any():
        raise ValueError("Dollar index contains non-positive values")
    return result


def _distance_from_mean(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 2)).mean()
    return (series / mean.replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)


def build_history(source: pd.DataFrame) -> pd.DataFrame:
    history = source.copy().sort_values("dollar_observation_date")
    level = history["dollar_index"]
    history["dollar_available_date"] = history["dollar_observation_date"] + pd.Timedelta(days=1)
    history["dollar_level"] = level
    for horizon in (1, 5, 20, 60):
        history[f"dollar_return_{horizon}"] = level.pct_change(horizon)
    history["dollar_percentile_60"] = _rolling_percentile(level, 60)
    history["dollar_percentile_252"] = _rolling_percentile(level, 252)
    history["dollar_distance_ma_20"] = _distance_from_mean(level, 20)
    history["dollar_distance_ma_60"] = _distance_from_mean(level, 60)
    history["dollar_acceleration_5_vs_20"] = history["dollar_return_5"] - history["dollar_return_20"]
    history["dollar_acceleration_20_vs_60"] = history["dollar_return_20"] - history["dollar_return_60"]
    return history[["dollar_observation_date", "dollar_available_date", *DOLLAR_FEATURES]]


def build_expanded_features(base: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    history = build_history(source).sort_values("dollar_available_date")
    expanded = pd.merge_asof(
        base.sort_values("decision_date"),
        history,
        left_on="decision_date",
        right_on="dollar_available_date",
        direction="backward",
        allow_exact_matches=True,
    )
    bad = expanded["dollar_available_date"].notna() & expanded["dollar_available_date"].gt(expanded["decision_date"])
    if bad.any():
        raise AssertionError("Future/unavailable dollar observation joined to a decision")
    same_day = expanded["dollar_observation_date"].notna() & expanded["dollar_observation_date"].ge(expanded["decision_date"])
    if same_day.any():
        raise AssertionError("Dollar observation was used without the conservative one-day lag")
    return expanded


def output_registry() -> dict[str, object]:
    base = json.loads(DEFAULT_BASE_REGISTRY.read_text(encoding="utf-8"))
    family = json.loads(DEFAULT_FAMILY_REGISTRY.read_text(encoding="utf-8"))
    features = [*base.get("features", []), *family.get("features", [])]
    names = [item["name"] for item in features]
    if len(names) != len(set(names)):
        raise ValueError("Dollar registry contains duplicate feature names")
    return {
        "version": family["version"],
        "parent_version": family["parent_version"],
        "decision_time_convention": family["decision_time_convention"],
        "features": features,
    }


def run_build(
    *,
    base_features_path: Path = DEFAULT_BASE_FEATURES,
    source_path: Path = DEFAULT_SOURCE,
    output_path: Path = DEFAULT_OUTPUT,
    output_registry_path: Path = DEFAULT_OUTPUT_REGISTRY,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, object]:
    base = load_base(base_features_path)
    expanded = build_expanded_features(base, load_source(source_path))
    assert_frame_equal(expanded[base.columns].reset_index(drop=True), base.reset_index(drop=True), check_dtype=True)
    registry = output_registry()
    expected = [item["name"] for item in registry["features"]]
    actual = [column for column in expanded.columns if column not in EXPANDED_METADATA]
    if set(expected) != set(actual):
        raise ValueError("Dollar feature registry does not match generated table")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_registry_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_parquet(output_path, index=False, engine="pyarrow")
    output_registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    report = {
        "feature_version": registry["version"],
        "rows": int(len(expanded)),
        "baseline_feature_count": int(len(actual) - len(DOLLAR_FEATURES)),
        "dollar_feature_count": int(len(DOLLAR_FEATURES)),
        "total_feature_count": int(len(actual)),
        "future_available_rows": int((expanded["dollar_available_date"].notna() & expanded["dollar_available_date"].gt(expanded["decision_date"])).sum()),
        "same_day_observation_rows": int((expanded["dollar_observation_date"].notna() & expanded["dollar_observation_date"].ge(expanded["decision_date"])).sum()),
        "missingness": {column: {"missing_rows": int(expanded[column].isna().sum()), "missing_fraction": float(expanded[column].isna().mean())} for column in ("dollar_observation_date", "dollar_available_date", *DOLLAR_FEATURES)},
        "input_sha256": {"baseline_features": _sha256(base_features_path), "dollar_snapshot": _sha256(source_path)},
        "output_sha256": _sha256(output_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    report = run_build(base_features_path=args.base_features, source_path=args.source, output_path=args.output, output_registry_path=args.output_registry, report_path=args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
