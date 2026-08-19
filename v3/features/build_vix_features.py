#!/usr/bin/env python3
"""Extend the frozen v3 baseline feature table with point-in-time VIX features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v3.features.build_features import _rolling_percentile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_FEATURES = ROOT / "v3" / "data" / "features_daily.parquet"
DEFAULT_VIX = ROOT / "v3" / "data" / "vix_daily.csv.gz"
DEFAULT_OUTPUT = ROOT / "v3" / "data" / "features_daily_vix.parquet"
DEFAULT_REGISTRY = ROOT / "v3" / "features" / "feature_registry_vix.json"
DEFAULT_REPORT = ROOT / "v3" / "reports" / "vix_features_missingness.json"

VIX_FEATURE_COLUMNS = (
    "vix_level",
    "vix_change_1",
    "vix_change_5",
    "vix_pct_change_5",
    "vix_percentile_60",
    "vix_percentile_252",
    "vix_distance_ma_20",
    "vix_distance_ma_60",
)
BASE_METADATA = {
    "decision_date",
    "fear_greed_date",
    "open",
    "high",
    "low",
    "close",
}
EXPANDED_METADATA = BASE_METADATA | {"vix_date"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-features", type=Path, default=DEFAULT_BASE_FEATURES)
    parser.add_argument("--vix", type=Path, default=DEFAULT_VIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base_features(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow").copy()
    if "decision_date" not in frame:
        raise ValueError("Baseline feature table has no decision_date")
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
    if frame["decision_date"].duplicated().any():
        raise ValueError("Baseline feature table has duplicate decision dates")
    if not frame["decision_date"].is_monotonic_increasing:
        frame = frame.sort_values("decision_date").reset_index(drop=True)
    return frame


def load_vix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "vix_close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"VIX snapshot missing columns: {sorted(missing)}")
    result = frame[["date", "vix_close"]].copy()
    result["vix_date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["vix_close"] = pd.to_numeric(result["vix_close"], errors="raise").astype(float)
    result = result.drop(columns=["date"]).sort_values("vix_date").reset_index(drop=True)
    if result["vix_date"].duplicated().any():
        raise ValueError("VIX snapshot has duplicate dates")
    if (result["vix_close"] <= 0.0).any():
        raise ValueError("VIX snapshot contains non-positive close values")
    return result


def _distance_from_mean(series: pd.Series, window: int) -> pd.Series:
    minimum = min(window, max(5, window // 2))
    mean = series.rolling(window, min_periods=minimum).mean()
    return (series / mean.replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)


def build_vix_history(vix: pd.DataFrame) -> pd.DataFrame:
    history = vix.sort_values("vix_date").copy()
    close = history["vix_close"]
    history["vix_level"] = close
    history["vix_change_1"] = close.diff(1)
    history["vix_change_5"] = close.diff(5)
    history["vix_pct_change_5"] = close.pct_change(5)
    history["vix_percentile_60"] = _rolling_percentile(close, 60)
    history["vix_percentile_252"] = _rolling_percentile(close, 252)
    history["vix_distance_ma_20"] = _distance_from_mean(close, 20)
    history["vix_distance_ma_60"] = _distance_from_mean(close, 60)
    return history[["vix_date", *VIX_FEATURE_COLUMNS]]


def build_expanded_features(
    base_features: pd.DataFrame,
    vix: pd.DataFrame,
) -> pd.DataFrame:
    base = base_features.sort_values("decision_date").copy()
    vix_history = build_vix_history(vix)
    expanded = pd.merge_asof(
        base,
        vix_history,
        left_on="decision_date",
        right_on="vix_date",
        direction="backward",
        allow_exact_matches=True,
    )
    future = (
        expanded["vix_date"].notna()
        & expanded["decision_date"].notna()
        & expanded["vix_date"].gt(expanded["decision_date"])
    )
    if future.any():
        raise AssertionError("Future VIX observation joined to a decision date")
    if expanded["decision_date"].duplicated().any():
        raise AssertionError("Expanded VIX feature table has duplicate decision dates")
    if not expanded["decision_date"].is_monotonic_increasing:
        raise AssertionError("Expanded VIX feature table is not chronologically sorted")
    return expanded


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in EXPANDED_METADATA]


def validate_registry(registry_path: Path, columns: list[str]) -> dict[str, object]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registered = [item["name"] for item in registry.get("features", [])]
    missing = sorted(set(columns).difference(registered))
    stale = sorted(set(registered).difference(columns))
    if missing or stale:
        raise ValueError(
            "VIX feature registry does not match generated features; "
            f"missing={missing}, stale={stale}"
        )
    return registry


def build_report(
    frame: pd.DataFrame,
    *,
    base_path: Path,
    vix_path: Path,
    output_path: Path,
    registry: dict[str, object],
) -> dict[str, object]:
    features = model_feature_columns(frame)
    return {
        "feature_version": registry.get("version"),
        "rows": int(len(frame)),
        "start": frame["decision_date"].min().date().isoformat() if len(frame) else None,
        "end": frame["decision_date"].max().date().isoformat() if len(frame) else None,
        "baseline_feature_count": int(len([c for c in features if c not in VIX_FEATURE_COLUMNS])),
        "vix_feature_count": int(len(VIX_FEATURE_COLUMNS)),
        "total_feature_count": int(len(features)),
        "vix_source_date_future_rows": int(
            (
                frame["vix_date"].notna()
                & frame["vix_date"].gt(frame["decision_date"])
            ).sum()
        ),
        "vix_missingness": {
            column: {
                "missing_rows": int(frame[column].isna().sum()),
                "missing_fraction": float(frame[column].isna().mean()),
            }
            for column in ("vix_date", *VIX_FEATURE_COLUMNS)
        },
        "input_sha256": {
            "baseline_features": _sha256(base_path),
            "vix_snapshot": _sha256(vix_path),
        },
        "output_sha256": _sha256(output_path),
    }


def run_build(
    *,
    base_features_path: Path = DEFAULT_BASE_FEATURES,
    vix_path: Path = DEFAULT_VIX,
    output_path: Path = DEFAULT_OUTPUT,
    registry_path: Path = DEFAULT_REGISTRY,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, object]:
    base = load_base_features(base_features_path)
    vix = load_vix(vix_path)
    expanded = build_expanded_features(base, vix)
    features = model_feature_columns(expanded)
    registry = validate_registry(registry_path, features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_parquet(output_path, index=False, engine="pyarrow")
    report = build_report(
        expanded,
        base_path=base_features_path,
        vix_path=vix_path,
        output_path=output_path,
        registry=registry,
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    report = run_build(
        base_features_path=args.base_features,
        vix_path=args.vix,
        output_path=args.output,
        registry_path=args.registry,
        report_path=args.report,
    )
    print(
        f"Wrote {args.output} ({report['rows']} rows, "
        f"{report['total_feature_count']} features)"
    )
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
