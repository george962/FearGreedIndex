#!/usr/bin/env python3
"""Build DATA-001 next-session-open labels and EXP-006-compatible target.

The executable-entry convention is identical to the existing V3 label builder:
a decision on session T enters at the next tradable S&P 500 session open.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v3.labels.build_labels import build_labels, load_market

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "v3" / "data" / "long_history"
DEFAULT_FEATURES = DATA_DIR / "core_features.parquet"
DEFAULT_MARKET = DATA_DIR / "spx_daily.csv.gz"
DEFAULT_LABELS = DATA_DIR / "core_labels.parquet"
DEFAULT_MODEL = DATA_DIR / "core_model_dataset.parquet"

RETURN_GOOD = 0.02
DRAWDOWN_BAD = -0.05


def add_favorable_entry_target(labels: pd.DataFrame) -> pd.DataFrame:
    result = labels.copy()
    known = pd.to_datetime(result["_forward_20d_known_date"], errors="coerce")
    forward_return = pd.to_numeric(result["forward_return_20d"], errors="coerce")
    drawdown = pd.to_numeric(result["max_drawdown_20d"], errors="coerce")
    mature = known.notna() & forward_return.notna() & drawdown.notna()

    favorable = pd.Series(pd.NA, index=result.index, dtype="boolean")
    favorable.loc[mature] = (
        forward_return.loc[mature].ge(RETURN_GOOD)
        & drawdown.loc[mature].gt(DRAWDOWN_BAD)
    )
    result["favorable_entry_20d"] = favorable
    return result


def build_long_history_labels(features: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "decision_date" not in features:
        raise ValueError("DATA-001 features have no decision_date")
    decisions = pd.to_datetime(features["decision_date"], errors="raise").dt.normalize()
    if decisions.duplicated().any():
        raise ValueError("DATA-001 features contain duplicate decision dates")

    labels = build_labels(decisions, market)
    labels = add_favorable_entry_target(labels)
    model = features.merge(labels, on="decision_date", how="left", validate="one_to_one")
    return labels, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--market", type=Path, default=DEFAULT_MARKET)
    parser.add_argument("--labels-output", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features = pd.read_parquet(args.features, engine="pyarrow")
    market = load_market(args.market)
    labels, model = build_long_history_labels(features, market)

    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(args.labels_output, index=False, engine="pyarrow")
    model.to_parquet(args.model_output, index=False, engine="pyarrow")
    mature = int(labels["favorable_entry_20d"].notna().sum())
    print(f"Wrote {args.labels_output} ({len(labels)} rows; {mature} mature 20d targets)")
    print(f"Wrote {args.model_output} ({len(model)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
