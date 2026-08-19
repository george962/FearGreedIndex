#!/usr/bin/env python3
"""Build the combined feature set from all independently retained V3 families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from v3.features.build_dollar_features import (
    DOLLAR_FEATURES,
    DEFAULT_SOURCE as DOLLAR_SOURCE,
    build_expanded_features as build_dollar_expanded,
    load_source as load_dollar_source,
)
from v3.features.build_relative_strength_features import (
    DEFAULT_SOURCE as RELATIVE_SOURCE,
    RELATIVE_STRENGTH_FEATURES,
    build_expanded_features as build_relative_expanded,
    load_source as load_relative_source,
)
from v3.features.build_treasury_features import (
    DEFAULT_SOURCE as TREASURY_SOURCE,
    TREASURY_FEATURES,
    build_expanded_features as build_treasury_expanded,
    load_source as load_treasury_source,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_FEATURES = ROOT / "v3" / "data" / "features_daily.parquet"
DEFAULT_BASE_REGISTRY = ROOT / "v3" / "features" / "feature_registry.json"
DEFAULT_RELATIVE_REGISTRY = ROOT / "v3" / "features" / "relative_strength_features.json"
DEFAULT_TREASURY_REGISTRY = ROOT / "v3" / "features" / "treasury_features.json"
DEFAULT_DOLLAR_REGISTRY = ROOT / "v3" / "features" / "dollar_features.json"
DEFAULT_OUTPUT = ROOT / "v3" / "data" / "features_daily_retained_combined.parquet"
DEFAULT_OUTPUT_REGISTRY = ROOT / "v3" / "reports" / "feature_registry_retained_combined.json"
DEFAULT_REPORT = ROOT / "v3" / "reports" / "retained_combined_features_missingness.json"
FEATURE_VERSION = "v3-features-006-retained-combined"

SOURCE_METADATA = (
    "relative_strength_date",
    "treasury_date",
    "dollar_observation_date",
    "dollar_available_date",
)
COMBINED_FEATURES = (
    *RELATIVE_STRENGTH_FEATURES,
    *TREASURY_FEATURES,
    *DOLLAR_FEATURES,
)


def load_base(path: Path = DEFAULT_BASE_FEATURES) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow").copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
    return frame.sort_values("decision_date").reset_index(drop=True)


def build_combined_features(base: pd.DataFrame) -> pd.DataFrame:
    relative = build_relative_expanded(base, load_relative_source(RELATIVE_SOURCE))
    treasury = build_treasury_expanded(base, load_treasury_source(TREASURY_SOURCE))
    dollar = build_dollar_expanded(base, load_dollar_source(DOLLAR_SOURCE))

    combined = base.copy()
    for frame, columns in (
        (relative, ("relative_strength_date", *RELATIVE_STRENGTH_FEATURES)),
        (treasury, ("treasury_date", *TREASURY_FEATURES)),
        (dollar, ("dollar_observation_date", "dollar_available_date", *DOLLAR_FEATURES)),
    ):
        combined = combined.merge(
            frame[["decision_date", *columns]],
            on="decision_date",
            how="left",
            validate="one_to_one",
        )

    assert_frame_equal(
        combined[base.columns].reset_index(drop=True),
        base.reset_index(drop=True),
        check_dtype=True,
    )
    if combined["decision_date"].duplicated().any():
        raise ValueError("Combined retained feature table has duplicate decision dates")
    return combined


def build_registry() -> dict[str, object]:
    base = json.loads(DEFAULT_BASE_REGISTRY.read_text(encoding="utf-8"))
    relative = json.loads(DEFAULT_RELATIVE_REGISTRY.read_text(encoding="utf-8"))
    treasury = json.loads(DEFAULT_TREASURY_REGISTRY.read_text(encoding="utf-8"))
    dollar = json.loads(DEFAULT_DOLLAR_REGISTRY.read_text(encoding="utf-8"))
    features = [
        *base.get("features", []),
        *relative.get("features", []),
        *treasury.get("features", []),
        *dollar.get("features", []),
    ]
    names = [item["name"] for item in features]
    if len(names) != len(set(names)):
        raise ValueError("Combined retained registry contains duplicate feature names")
    return {
        "version": FEATURE_VERSION,
        "parent_version": "v3-features-001",
        "component_versions": [
            relative["version"],
            treasury["version"],
            dollar["version"],
        ],
        "decision_time_convention": (
            "Each component retains its independently validated point-in-time source "
            "and availability rule; the combined table does not alter source timing."
        ),
        "features": features,
    }


def run_build(
    output: Path = DEFAULT_OUTPUT,
    registry_output: Path = DEFAULT_OUTPUT_REGISTRY,
    report_output: Path = DEFAULT_REPORT,
) -> dict[str, object]:
    base = load_base()
    combined = build_combined_features(base)
    registry = build_registry()
    feature_names = [item["name"] for item in registry["features"]]
    if len(feature_names) != 76:
        raise ValueError(f"Expected 76 combined model features, got {len(feature_names)}")
    missing = sorted(set(feature_names).difference(combined.columns))
    if missing:
        raise ValueError(f"Combined feature table missing registry columns: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    registry_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False, engine="pyarrow")
    registry_output.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    report = {
        "feature_version": FEATURE_VERSION,
        "rows": int(len(combined)),
        "baseline_feature_count": 41,
        "relative_strength_feature_count": len(RELATIVE_STRENGTH_FEATURES),
        "treasury_feature_count": len(TREASURY_FEATURES),
        "dollar_feature_count": len(DOLLAR_FEATURES),
        "total_feature_count": len(feature_names),
        "missingness": {
            column: {
                "missing_rows": int(combined[column].isna().sum()),
                "missing_fraction": float(combined[column].isna().mean()),
            }
            for column in (*SOURCE_METADATA, *COMBINED_FEATURES)
        },
    }
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registry-output", type=Path, default=DEFAULT_OUTPUT_REGISTRY)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(json.dumps(run_build(args.output, args.registry_output, args.report_output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
