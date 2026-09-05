#!/usr/bin/env python3
"""Build the DATA-001 long-history core regime feature matrix.

The feature family is intentionally small and transparent. It contains only
S&P 500 price/volatility state, Cboe VIX state, and U.S. Treasury 2Y/10Y state.
CNN Fear & Greed is excluded by construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "v3" / "data" / "long_history"
DEFAULT_SPX = DATA_DIR / "spx_daily.csv.gz"
DEFAULT_VIX = DATA_DIR / "vix_daily.csv.gz"
DEFAULT_TREASURY = DATA_DIR / "treasury_daily.csv.gz"
DEFAULT_OUTPUT = DATA_DIR / "core_features.parquet"
DEFAULT_REGISTRY = ROOT / "v3" / "reports" / "feature_registry_long_history_core.json"

FEATURE_VERSION = "v3-long-history-core-001"
RESEARCH_CUTOFF = pd.Timestamp("2026-08-18")


def trailing_percentile(series: pd.Series, window: int) -> pd.Series:
    """Causal percentile rank of today's value within the trailing window.

    The current observation is included because it is known at the decision date.
    Ties use the weak <= rank, making the definition deterministic.
    """

    def rank_last(values: np.ndarray) -> float:
        current = values[-1]
        return float(np.mean(values <= current))

    return series.rolling(window, min_periods=window).apply(rank_last, raw=True)


def load_sources(
    spx_path: Path = DEFAULT_SPX,
    vix_path: Path = DEFAULT_VIX,
    treasury_path: Path = DEFAULT_TREASURY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spx = pd.read_csv(spx_path)
    vix = pd.read_csv(vix_path)
    treasury = pd.read_csv(treasury_path)

    for name, frame in (("spx", spx), ("vix", vix), ("treasury", treasury)):
        if "date" not in frame:
            raise ValueError(f"DATA-001 {name} source has no date column")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
            raise ValueError(f"DATA-001 {name} dates must be unique and increasing")
        if frame["date"].max() > RESEARCH_CUTOFF:
            raise ValueError(f"DATA-001 {name} source exceeds frozen cutoff")

    for column in ("open", "high", "low", "close", "adj_close"):
        spx[column] = pd.to_numeric(spx[column], errors="raise").astype(float)
    vix["vix_close"] = pd.to_numeric(vix["vix_close"], errors="raise").astype(float)
    for column in ("dgs2", "dgs10"):
        treasury[column] = pd.to_numeric(treasury[column], errors="raise").astype(float)
    return spx, vix, treasury


def align_sources(spx: pd.DataFrame, vix: pd.DataFrame, treasury: pd.DataFrame) -> pd.DataFrame:
    base = spx.copy().sort_values("date").reset_index(drop=True)
    vix_aligned = vix.rename(columns={"date": "vix_observation_date"}).copy()
    base = base.merge(
        vix_aligned,
        left_on="date",
        right_on="vix_observation_date",
        how="left",
        validate="one_to_one",
    )

    treasury_aligned = treasury.rename(columns={"date": "treasury_observation_date"}).copy()
    base = pd.merge_asof(
        base.sort_values("date"),
        treasury_aligned.sort_values("treasury_observation_date"),
        left_on="date",
        right_on="treasury_observation_date",
        direction="backward",
        allow_exact_matches=True,
    )
    base = base.rename(columns={"date": "decision_date"})
    return base.sort_values("decision_date").reset_index(drop=True)


def build_feature_frame(aligned: pd.DataFrame) -> pd.DataFrame:
    result = aligned.copy().sort_values("decision_date").reset_index(drop=True)
    close = pd.to_numeric(result["close"], errors="raise").astype(float)
    high = pd.to_numeric(result["high"], errors="raise").astype(float)
    log_return_1d = np.log(close).diff()

    for horizon in (1, 3, 5, 10, 20, 60):
        result[f"spx_return_{horizon}"] = close / close.shift(horizon) - 1.0

    for window in (20, 50, 200):
        ma = close.rolling(window, min_periods=window).mean()
        result[f"spx_distance_ma_{window}"] = close / ma - 1.0

    for window in (20, 60, 252):
        rolling_high_price = high.rolling(window, min_periods=window).max()
        rolling_high_close = close.rolling(window, min_periods=window).max()
        rolling_low_close = close.rolling(window, min_periods=window).min()
        result[f"spx_distance_high_{window}"] = close / rolling_high_price - 1.0
        result[f"spx_drawdown_{window}"] = close / rolling_high_close - 1.0
        result[f"spx_rebound_{window}"] = close / rolling_low_close - 1.0

    for window in (5, 20, 60):
        result[f"spx_realized_vol_{window}"] = (
            log_return_1d.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252.0)
        )

    vix = pd.to_numeric(result["vix_close"], errors="coerce").astype(float)
    result["vix_level"] = vix
    for horizon in (1, 5, 20):
        result[f"vix_change_{horizon}"] = vix - vix.shift(horizon)
        result[f"vix_return_{horizon}"] = vix / vix.shift(horizon) - 1.0
    for window in (60, 252):
        result[f"vix_percentile_{window}"] = trailing_percentile(vix, window)
    for window in (20, 60):
        result[f"vix_distance_ma_{window}"] = vix / vix.rolling(
            window, min_periods=window
        ).mean() - 1.0

    dgs2 = pd.to_numeric(result["dgs2"], errors="coerce").astype(float)
    dgs10 = pd.to_numeric(result["dgs10"], errors="coerce").astype(float)
    slope = dgs10 - dgs2
    result["treasury_2y_level"] = dgs2
    result["treasury_10y_level"] = dgs10
    result["treasury_10y_2y_slope"] = slope
    for horizon in (1, 5, 20):
        result[f"treasury_2y_change_{horizon}"] = dgs2 - dgs2.shift(horizon)
        result[f"treasury_10y_change_{horizon}"] = dgs10 - dgs10.shift(horizon)
        result[f"treasury_slope_change_{horizon}"] = slope - slope.shift(horizon)
    result["treasury_2y_percentile_252"] = trailing_percentile(dgs2, 252)
    result["treasury_10y_percentile_252"] = trailing_percentile(dgs10, 252)
    result["treasury_slope_percentile_252"] = trailing_percentile(slope, 252)

    return result


def load_registry(path: Path = DEFAULT_REGISTRY) -> tuple[str, list[str]]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    version = str(registry["feature_set_version"])
    features = [str(item["name"]) for item in registry["features"]]
    if version != FEATURE_VERSION:
        raise ValueError(f"DATA-001 registry version mismatch: {version}")
    if len(features) != len(set(features)):
        raise ValueError("DATA-001 registry contains duplicate features")
    return version, features


def build_dataset(
    *,
    spx_path: Path = DEFAULT_SPX,
    vix_path: Path = DEFAULT_VIX,
    treasury_path: Path = DEFAULT_TREASURY,
    registry_path: Path = DEFAULT_REGISTRY,
) -> pd.DataFrame:
    spx, vix, treasury = load_sources(spx_path, vix_path, treasury_path)
    aligned = align_sources(spx, vix, treasury)
    frame = build_feature_frame(aligned)
    _, features = load_registry(registry_path)
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise ValueError(f"DATA-001 feature builder missing registry features: {missing}")

    metadata = [
        "decision_date",
        "vix_observation_date",
        "treasury_observation_date",
    ]
    output = frame.loc[:, metadata + features].copy()
    if output["decision_date"].duplicated().any():
        raise ValueError("DATA-001 feature dataset contains duplicate decision dates")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spx", type=Path, default=DEFAULT_SPX)
    parser.add_argument("--vix", type=Path, default=DEFAULT_VIX)
    parser.add_argument("--treasury", type=Path, default=DEFAULT_TREASURY)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = build_dataset(
        spx_path=args.spx,
        vix_path=args.vix,
        treasury_path=args.treasury,
        registry_path=args.registry,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False, engine="pyarrow")
    complete_rows = int(frame.dropna().shape[0])
    print(
        f"Wrote {args.output} ({len(frame)} rows, {len(frame.columns) - 3} features, "
        f"{complete_rows} complete rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
