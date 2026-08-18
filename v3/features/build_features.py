#!/usr/bin/env python3
"""Build the v3 point-in-time daily feature dataset.

All feature columns are constructed from information dated on or before the
``decision_date``. Forward-looking outcomes intentionally live in the labels
pipeline instead of this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FG = ROOT / "data" / "fear_greed_daily.csv"
DEFAULT_MARKET = ROOT / "data" / "spx_daily.csv"
DEFAULT_OUTPUT = ROOT / "v3" / "data" / "features_daily.parquet"
DEFAULT_MISSINGNESS = ROOT / "v3" / "reports" / "features_missingness.json"
DEFAULT_REGISTRY = ROOT / "v3" / "features" / "feature_registry.json"

FORBIDDEN_FEATURE_TOKENS = (
    "forward_",
    "future_",
    "target",
    "label",
    "outcome",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fear-greed", type=Path, default=DEFAULT_FG)
    parser.add_argument("--market", type=Path, default=DEFAULT_MARKET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--missingness-output", type=Path, default=DEFAULT_MISSINGNESS)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of the current value inside its trailing window."""

    def rank_last(values: np.ndarray) -> float:
        if len(values) == 0 or not np.isfinite(values[-1]):
            return np.nan
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return np.nan
        current = values[-1]
        less = np.sum(finite < current)
        equal = np.sum(finite == current)
        return float((less + 0.5 * equal) / len(finite))

    minimum = min(window, max(5, window // 4))
    return series.rolling(window=window, min_periods=minimum).apply(
        rank_last,
        raw=True,
    )


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan) - 1.0
    return result.replace([np.inf, -np.inf], np.nan)


def load_fear_greed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Date", "Value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Fear & Greed data missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["fear_greed_date"] = pd.to_datetime(frame["Date"], errors="raise").dt.normalize()
    frame["fear_greed"] = pd.to_numeric(frame["Value"], errors="raise").astype(float)
    frame = frame.sort_values("fear_greed_date")
    if frame["fear_greed_date"].duplicated().any():
        duplicates = frame.loc[
            frame["fear_greed_date"].duplicated(keep=False), "fear_greed_date"
        ].dt.date.astype(str).unique()
        raise ValueError(f"Duplicate Fear & Greed dates: {duplicates[:5].tolist()}")
    return frame[["fear_greed_date", "fear_greed"]]


def load_market(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Market data missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["decision_date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    frame = frame.sort_values("decision_date")
    if frame["decision_date"].duplicated().any():
        duplicates = frame.loc[
            frame["decision_date"].duplicated(keep=False), "decision_date"
        ].dt.date.astype(str).unique()
        raise ValueError(f"Duplicate market dates: {duplicates[:5].tolist()}")
    return frame[["decision_date", "open", "high", "low", "close"]]


def build_feature_frame(fear_greed: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    fg = fear_greed.sort_values("fear_greed_date").copy()
    mkt = market.sort_values("decision_date").copy()

    frame = pd.merge_asof(
        mkt,
        fg,
        left_on="decision_date",
        right_on="fear_greed_date",
        direction="backward",
        allow_exact_matches=True,
    )
    if (frame["fear_greed_date"] > frame["decision_date"]).fillna(False).any():
        raise AssertionError("Future Fear & Greed observation joined to a decision date")

    fg_level = frame["fear_greed"]
    for lag in (1, 3, 5, 10):
        frame[f"fg_change_{lag}"] = fg_level.diff(lag)

    for window in (5, 20):
        minimum = min(window, max(2, window // 2))
        rolling_min = fg_level.rolling(window, min_periods=minimum).min()
        rolling_max = fg_level.rolling(window, min_periods=minimum).max()
        frame[f"fg_min_{window}"] = rolling_min
        frame[f"fg_max_{window}"] = rolling_max
        frame[f"fg_distance_from_min_{window}"] = fg_level - rolling_min
        frame[f"fg_distance_from_max_{window}"] = fg_level - rolling_max

    for window in (60, 252):
        frame[f"fg_percentile_{window}"] = _rolling_percentile(fg_level, window)

    frame["fg_acceleration_1_vs_5"] = frame["fg_change_1"] - frame["fg_change_5"] / 5.0
    frame["fg_reversal_1_vs_5"] = np.where(
        frame["fg_change_1"].notna() & frame["fg_change_5"].notna(),
        (np.sign(frame["fg_change_1"]) != np.sign(frame["fg_change_5"])).astype(float),
        np.nan,
    )

    close = frame["close"]
    daily_return = close.pct_change()
    for horizon in (1, 3, 5, 10, 20, 60):
        frame[f"spx_return_{horizon}"] = close.pct_change(horizon)

    for window in (20, 50, 200):
        minimum = min(window, max(5, window // 2))
        moving_average = close.rolling(window, min_periods=minimum).mean()
        frame[f"spx_distance_ma_{window}"] = _safe_ratio(close, moving_average)

    for window in (20, 60, 252):
        minimum = min(window, max(5, window // 2))
        trailing_high = close.rolling(window, min_periods=minimum).max()
        trailing_low = close.rolling(window, min_periods=minimum).min()
        frame[f"spx_distance_high_{window}"] = _safe_ratio(close, trailing_high)
        frame[f"spx_drawdown_{window}"] = _safe_ratio(close, trailing_high)
        frame[f"spx_rebound_from_low_{window}"] = _safe_ratio(close, trailing_low)

    for window in (5, 20, 60):
        minimum = min(window, max(3, window // 2))
        frame[f"spx_realized_vol_{window}"] = (
            daily_return.rolling(window, min_periods=minimum).std(ddof=1) * np.sqrt(252.0)
        )

    frame["interaction_fg_x_drawdown_20"] = frame["fear_greed"] * frame["spx_drawdown_20"]
    frame["interaction_fg_x_vol_20"] = frame["fear_greed"] * frame["spx_realized_vol_20"]
    frame["interaction_fg_change_5_x_spx_return_5"] = (
        frame["fg_change_5"] * frame["spx_return_5"]
    )

    # Keep the market-history warmup for rolling calculations, but do not emit
    # decision rows before the first available Fear & Greed observation.
    frame = frame.loc[frame["fear_greed_date"].notna()].copy()

    metadata_columns = [
        "decision_date",
        "fear_greed_date",
        "open",
        "high",
        "low",
        "close",
    ]
    generated_features = [column for column in frame.columns if column not in metadata_columns]
    forbidden = [
        column
        for column in generated_features
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise AssertionError(f"Forward/target columns leaked into features: {forbidden}")

    frame = frame[metadata_columns + generated_features].sort_values("decision_date").reset_index(drop=True)
    if frame["decision_date"].duplicated().any():
        raise AssertionError("Feature frame has duplicate decision dates")
    if not frame["decision_date"].is_monotonic_increasing:
        raise AssertionError("Feature frame is not chronologically sorted")
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    metadata = {"decision_date", "fear_greed_date", "open", "high", "low", "close"}
    return [column for column in frame.columns if column not in metadata]


def missingness_report(frame: pd.DataFrame, features: Iterable[str]) -> dict[str, object]:
    columns = list(features)
    return {
        "rows": int(len(frame)),
        "start": frame["decision_date"].min().date().isoformat() if len(frame) else None,
        "end": frame["decision_date"].max().date().isoformat() if len(frame) else None,
        "columns": {
            column: {
                "missing_rows": int(frame[column].isna().sum()),
                "missing_fraction": float(frame[column].isna().mean()),
            }
            for column in columns
        },
    }


def validate_registry(registry_path: Path, columns: list[str]) -> None:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registered = [item["name"] for item in registry.get("features", [])]
    missing = sorted(set(columns).difference(registered))
    stale = sorted(set(registered).difference(columns))
    if missing or stale:
        raise ValueError(
            "feature_registry.json does not match generated features; "
            f"missing={missing}, stale={stale}"
        )


def main() -> int:
    args = parse_args()
    fear_greed = load_fear_greed(args.fear_greed)
    market = load_market(args.market)
    frame = build_feature_frame(fear_greed, market)
    features = feature_columns(frame)
    validate_registry(args.registry, features)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.missingness_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False, engine="pyarrow")

    report = missingness_report(frame, features)
    report["input_sha256"] = {
        "fear_greed": _sha256(args.fear_greed),
        "market": _sha256(args.market),
    }
    report["output_sha256"] = _sha256(args.output)
    args.missingness_output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {args.output} ({len(frame)} rows, {len(features)} features)")
    print(f"Wrote {args.missingness_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
