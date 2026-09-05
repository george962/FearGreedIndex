#!/usr/bin/env python3
"""Validate DATA-001 long-history core sources, features, and labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v3.data_sources.fetch_long_history_core import RESEARCH_CUTOFF, verify as verify_sources

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "v3" / "data" / "long_history"
FEATURES = DATA_DIR / "core_features.parquet"
LABELS = DATA_DIR / "core_labels.parquet"
MODEL = DATA_DIR / "core_model_dataset.parquet"
MARKET = DATA_DIR / "spx_daily.csv.gz"
REGISTRY = ROOT / "v3" / "reports" / "feature_registry_long_history_core.json"
REPORT = ROOT / "v3" / "reports" / "data001_summary.json"

FORBIDDEN_FEATURE_TOKENS = (
    "fear_greed",
    "favorable_entry",
    "forward_return",
    "forward_positive",
    "opportunity_state",
    "known_date",
    "entry_price",
)


def load_registry() -> list[str]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry.get("feature_set_version") != "v3-long-history-core-001":
        raise ValueError("DATA-001 feature registry version changed")
    if registry.get("cnn_fear_greed_included") is not False:
        raise ValueError("DATA-001 registry must exclude Fear & Greed")
    features = [str(item["name"]) for item in registry["features"]]
    if len(features) != 47 or len(features) != len(set(features)):
        raise ValueError("DATA-001 registry must contain exactly 47 unique features")
    for feature in features:
        lowered = feature.lower()
        if any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS):
            raise ValueError(f"Outcome/Fear & Greed vocabulary entered feature registry: {feature}")
    return features


def validate_source_dates(features: pd.DataFrame) -> None:
    decision = pd.to_datetime(features["decision_date"], errors="raise").dt.normalize()
    vix_date = pd.to_datetime(features["vix_observation_date"], errors="coerce").dt.normalize()
    treasury_date = pd.to_datetime(features["treasury_observation_date"], errors="coerce").dt.normalize()

    if decision.duplicated().any() or not decision.is_monotonic_increasing:
        raise ValueError("DATA-001 decision dates must be unique and increasing")
    if decision.max() > pd.Timestamp(RESEARCH_CUTOFF):
        raise ValueError("DATA-001 features exceed frozen research cutoff")
    if (vix_date.notna() & vix_date.ne(decision)).any():
        raise ValueError("DATA-001 VIX observations must match the decision session exactly")
    if (treasury_date.notna() & treasury_date.gt(decision)).any():
        raise ValueError("DATA-001 Treasury as-of join used a future observation")
    age = (decision - treasury_date).dt.days
    if (age.dropna() < 0).any() or (age.dropna() > 7).any():
        raise ValueError("DATA-001 Treasury observation lag is outside 0-7 calendar days")


def validate_labels(features: pd.DataFrame, labels: pd.DataFrame, market: pd.DataFrame) -> None:
    market = market.copy()
    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    market = market.sort_values("date").reset_index(drop=True)
    market_dates = market["date"].to_numpy(dtype="datetime64[ns]")
    market_open = pd.to_numeric(market["open"], errors="raise").to_numpy(float)

    labels = labels.copy()
    labels["decision_date"] = pd.to_datetime(labels["decision_date"], errors="raise").dt.normalize()
    labels["entry_date"] = pd.to_datetime(labels["entry_date"], errors="coerce").dt.normalize()
    if len(labels) != len(features):
        raise ValueError("DATA-001 feature/label row counts differ")

    for row in labels.itertuples(index=False):
        decision_np = np.datetime64(pd.Timestamp(row.decision_date).to_datetime64(), "ns")
        entry_index = int(np.searchsorted(market_dates, decision_np, side="right"))
        if entry_index >= len(market_dates):
            if pd.notna(row.entry_date):
                raise ValueError("DATA-001 label has entry after market snapshot ended")
            continue
        expected_date = pd.Timestamp(market_dates[entry_index])
        if pd.Timestamp(row.entry_date) != expected_date:
            raise ValueError("DATA-001 entry date is not the next tradable session")
        if not np.isclose(float(row.entry_price), float(market_open[entry_index]), rtol=0.0, atol=1e-10):
            raise ValueError("DATA-001 entry price is not the next-session open")

    known = pd.to_datetime(labels["_forward_20d_known_date"], errors="coerce")
    mature_inputs = (
        pd.to_numeric(labels["forward_return_20d"], errors="coerce").notna()
        & pd.to_numeric(labels["max_drawdown_20d"], errors="coerce").notna()
        & known.notna()
    )
    if labels.loc[mature_inputs, "favorable_entry_20d"].isna().any():
        raise ValueError("DATA-001 mature 20d row lacks favorable_entry_20d")
    if labels.loc[~mature_inputs, "favorable_entry_20d"].notna().any():
        raise ValueError("DATA-001 favorable_entry_20d assigned before maturity")


def run() -> dict[str, object]:
    source_report = verify_sources()
    feature_names = load_registry()
    features = pd.read_parquet(FEATURES, engine="pyarrow").copy()
    labels = pd.read_parquet(LABELS, engine="pyarrow").copy()
    model = pd.read_parquet(MODEL, engine="pyarrow").copy()
    market = pd.read_csv(MARKET)

    expected_columns = ["decision_date", "vix_observation_date", "treasury_observation_date"] + feature_names
    if list(features.columns) != expected_columns:
        raise ValueError("DATA-001 feature columns differ from frozen registry/order")
    validate_source_dates(features)
    validate_labels(features, labels, market)
    if len(model) != len(features):
        raise ValueError("DATA-001 model dataset row count differs from features")

    complete = features.dropna(subset=feature_names).copy()
    if complete.empty:
        raise ValueError("DATA-001 has no complete feature rows")
    complete_dates = pd.to_datetime(complete["decision_date"], errors="raise").dt.normalize()
    eras = {
        "1990s": ("1990-01-01", "1999-12-31"),
        "2000s": ("2000-01-01", "2009-12-31"),
        "2010s": ("2010-01-01", "2019-12-31"),
        "2020s": ("2020-01-01", RESEARCH_CUTOFF),
    }
    era_counts: dict[str, int] = {}
    for name, (start, end) in eras.items():
        count = int(complete_dates.between(pd.Timestamp(start), pd.Timestamp(end)).sum())
        era_counts[name] = count
        if count < 100:
            raise ValueError(f"DATA-001 lacks adequate complete rows in {name}: {count}")

    report: dict[str, object] = {
        "dataset_id": "DATA-001",
        "status": "PASS",
        "feature_set_version": "v3-long-history-core-001",
        "feature_count": len(feature_names),
        "feature_rows": int(len(features)),
        "complete_feature_rows": int(len(complete)),
        "complete_feature_start": complete_dates.min().date().isoformat(),
        "complete_feature_end": complete_dates.max().date().isoformat(),
        "era_complete_row_counts": era_counts,
        "mature_favorable_entry_20d_rows": int(labels["favorable_entry_20d"].notna().sum()),
        "cnn_fear_greed_included": False,
        "interactions_included": False,
        "source_validation": source_report,
        "champion_selected": False,
        "v3_019_eligible": False,
        "sizing_multiplier": 1.0,
        "core_001_evaluated": False,
        "note": "DATA-001 validates data infrastructure only; no model has been fit and no historical outcomes have been inspected for method selection.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = run()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
