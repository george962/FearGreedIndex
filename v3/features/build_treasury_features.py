#!/usr/bin/env python3
"""Extend v3-features-001 with point-in-time Treasury-rate features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from v3.features.build_features import _rolling_percentile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_FEATURES = ROOT / "v3" / "data" / "features_daily.parquet"
DEFAULT_BASE_REGISTRY = ROOT / "v3" / "features" / "feature_registry.json"
DEFAULT_FAMILY_REGISTRY = ROOT / "v3" / "features" / "treasury_features.json"
DEFAULT_SOURCE = ROOT / "v3" / "data" / "treasury_daily.csv.gz"
DEFAULT_OUTPUT = ROOT / "v3" / "data" / "features_daily_treasury.parquet"
DEFAULT_OUTPUT_REGISTRY = ROOT / "v3" / "reports" / "feature_registry_treasury.json"
DEFAULT_REPORT = ROOT / "v3" / "reports" / "treasury_features_missingness.json"

TREASURY_FEATURES = (
    "treasury_2y_level",
    "treasury_10y_level",
    "treasury_10y_2y_slope",
    "treasury_2y_change_1",
    "treasury_2y_change_5",
    "treasury_2y_change_20",
    "treasury_10y_change_1",
    "treasury_10y_change_5",
    "treasury_10y_change_20",
    "treasury_slope_change_5",
    "treasury_slope_change_20",
    "treasury_10y_percentile_252",
)
BASE_METADATA = {
    "decision_date",
    "fear_greed_date",
    "open",
    "high",
    "low",
    "close",
}
EXPANDED_METADATA = BASE_METADATA | {"treasury_date"}


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
    if "decision_date" not in frame:
        raise ValueError("Baseline feature table has no decision_date")
    frame["decision_date"] = pd.to_datetime(
        frame["decision_date"], errors="raise"
    ).dt.normalize()
    frame = frame.sort_values("decision_date").reset_index(drop=True)
    if frame["decision_date"].duplicated().any():
        raise ValueError("Baseline feature table has duplicate decision dates")
    return frame


def load_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "dgs2", "dgs10"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Treasury snapshot missing columns: {missing}")
    result = frame[["date", "dgs2", "dgs10"]].copy()
    result["treasury_date"] = pd.to_datetime(
        result.pop("date"), errors="raise"
    ).dt.normalize()
    for column in ("dgs2", "dgs10"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    result = result.sort_values("treasury_date").reset_index(drop=True)
    if result.empty:
        raise ValueError("Treasury snapshot is empty")
    if result["treasury_date"].duplicated().any():
        raise ValueError("Treasury snapshot contains duplicate dates")
    return result


def build_source_history(source: pd.DataFrame) -> pd.DataFrame:
    history = source.sort_values("treasury_date").copy()
    two = history["dgs2"]
    ten = history["dgs10"]
    slope = ten - two

    history["treasury_2y_level"] = two
    history["treasury_10y_level"] = ten
    history["treasury_10y_2y_slope"] = slope
    for horizon in (1, 5, 20):
        history[f"treasury_2y_change_{horizon}"] = two.diff(horizon)
        history[f"treasury_10y_change_{horizon}"] = ten.diff(horizon)
    history["treasury_slope_change_5"] = slope.diff(5)
    history["treasury_slope_change_20"] = slope.diff(20)
    history["treasury_10y_percentile_252"] = _rolling_percentile(ten, 252)
    return history[["treasury_date", *TREASURY_FEATURES]]


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
        right_on="treasury_date",
        direction="backward",
        allow_exact_matches=True,
    )
    future = (
        expanded["treasury_date"].notna()
        & expanded["treasury_date"].gt(expanded["decision_date"])
    )
    if future.any():
        raise AssertionError("Future Treasury observation joined to a decision")
    if expanded["decision_date"].duplicated().any():
        raise AssertionError("Expanded Treasury feature table has duplicate dates")
    return expanded


def build_output_registry(
    base_registry_path: Path = DEFAULT_BASE_REGISTRY,
    family_registry_path: Path = DEFAULT_FAMILY_REGISTRY,
) -> dict[str, object]:
    base = json.loads(base_registry_path.read_text(encoding="utf-8"))
    family = json.loads(family_registry_path.read_text(encoding="utf-8"))
    features = [*base.get("features", []), *family.get("features", [])]
    names = [item["name"] for item in features]
    if len(names) != len(set(names)):
        raise ValueError("Treasury registry contains duplicate feature names")
    return {
        "version": family["version"],
        "parent_version": family["parent_version"],
        "decision_time_convention": family["decision_time_convention"],
        "features": features,
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
            "Generated Treasury registry does not match feature table; "
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
        "baseline_feature_count": int(len(actual) - len(TREASURY_FEATURES)),
        "treasury_feature_count": int(len(TREASURY_FEATURES)),
        "total_feature_count": int(len(actual)),
        "future_source_rows": int(
            (
                expanded["treasury_date"].notna()
                & expanded["treasury_date"].gt(expanded["decision_date"])
            ).sum()
        ),
        "missingness": {
            column: {
                "missing_rows": int(expanded[column].isna().sum()),
                "missing_fraction": float(expanded[column].isna().mean()),
            }
            for column in ("treasury_date", *TREASURY_FEATURES)
        },
        "input_sha256": {
            "baseline_features": _sha256(base_features_path),
            "treasury_snapshot": _sha256(source_path),
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
