#!/usr/bin/env python3
"""Build executable-entry multi-horizon v3 labels.

A decision on date T is entered at the next tradable session open. Each horizon
contains that entry session as session 1, so a 5-session target is the close of
the fifth tradable session beginning with the entry date.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES = ROOT / "v3" / "data" / "features_daily.parquet"
DEFAULT_MARKET = ROOT / "data" / "spx_daily.csv"
DEFAULT_LABELS = ROOT / "v3" / "data" / "labels_daily.parquet"
DEFAULT_MODEL_DATASET = ROOT / "v3" / "data" / "model_dataset.parquet"
HORIZONS = (5, 20, 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--market", type=Path, default=DEFAULT_MARKET)
    parser.add_argument("--labels-output", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_DATASET)
    return parser.parse_args()


def load_market(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Market data missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("Market data has duplicate dates")
    return frame[["date", "open", "high", "low", "close"]]


def build_labels(decision_dates: pd.Series, market: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(decision_dates, errors="raise").dt.normalize()
    market_dates = market["date"].to_numpy(dtype="datetime64[ns]")
    opens = market["open"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    closes = market["close"].to_numpy(float)

    rows: list[dict[str, object]] = []
    for decision_date in dates:
        decision_np = np.datetime64(decision_date.to_datetime64(), "ns")
        entry_index = int(np.searchsorted(market_dates, decision_np, side="right"))
        row: dict[str, object] = {
            "decision_date": decision_date,
            "entry_date": pd.NaT,
            "entry_price": np.nan,
        }
        if entry_index >= len(market):
            for horizon in HORIZONS:
                row[f"forward_return_{horizon}d"] = np.nan
                row[f"forward_positive_{horizon}d"] = pd.NA
                row[f"max_drawdown_{horizon}d"] = np.nan
                row[f"_forward_{horizon}d_known_date"] = pd.NaT
            row["further_5pct_decline_20d"] = pd.NA
            rows.append(row)
            continue

        entry_price = float(opens[entry_index])
        row["entry_date"] = pd.Timestamp(market_dates[entry_index])
        row["entry_price"] = entry_price

        for horizon in HORIZONS:
            target_index = entry_index + horizon - 1
            if target_index >= len(market):
                row[f"forward_return_{horizon}d"] = np.nan
                row[f"forward_positive_{horizon}d"] = pd.NA
                row[f"max_drawdown_{horizon}d"] = np.nan
                row[f"_forward_{horizon}d_known_date"] = pd.NaT
                continue

            target_close = float(closes[target_index])
            forward_return = target_close / entry_price - 1.0
            path_low = float(np.nanmin(lows[entry_index : target_index + 1]))
            max_drawdown = path_low / entry_price - 1.0

            row[f"forward_return_{horizon}d"] = forward_return
            row[f"forward_positive_{horizon}d"] = bool(forward_return > 0.0)
            row[f"max_drawdown_{horizon}d"] = max_drawdown
            row[f"_forward_{horizon}d_known_date"] = pd.Timestamp(
                market_dates[target_index]
            )

        dd20 = row.get("max_drawdown_20d")
        row["further_5pct_decline_20d"] = (
            pd.NA if pd.isna(dd20) else bool(float(dd20) <= -0.05)
        )
        rows.append(row)

    labels = pd.DataFrame(rows)
    for horizon in HORIZONS:
        labels[f"forward_positive_{horizon}d"] = labels[
            f"forward_positive_{horizon}d"
        ].astype("boolean")
    labels["further_5pct_decline_20d"] = labels[
        "further_5pct_decline_20d"
    ].astype("boolean")
    return labels


def main() -> int:
    args = parse_args()
    features = pd.read_parquet(args.features, engine="pyarrow")
    if "decision_date" not in features:
        raise ValueError("Feature dataset has no decision_date")
    features = features.copy()
    features["decision_date"] = pd.to_datetime(features["decision_date"], errors="raise").dt.normalize()
    if features["decision_date"].duplicated().any():
        raise ValueError("Feature dataset has duplicate decision dates")

    market = load_market(args.market)
    labels = build_labels(features["decision_date"], market)
    model_dataset = features.merge(labels, on="decision_date", how="left", validate="one_to_one")

    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(args.labels_output, index=False, engine="pyarrow")
    model_dataset.to_parquet(args.model_output, index=False, engine="pyarrow")
    print(f"Wrote {args.labels_output} ({len(labels)} rows)")
    print(f"Wrote {args.model_output} ({len(model_dataset)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
