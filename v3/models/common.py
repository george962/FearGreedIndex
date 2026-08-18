#!/usr/bin/env python3
"""Shared point-in-time contracts for v3 model candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DATASET = ROOT / "v3" / "data" / "model_dataset.parquet"
DEFAULT_FEATURE_REGISTRY = ROOT / "v3" / "features" / "feature_registry.json"
HORIZONS = (5, 20, 60)


def load_feature_registry(path: Path = DEFAULT_FEATURE_REGISTRY) -> tuple[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = str(payload.get("version", "")).strip()
    features = [str(item["name"]) for item in payload.get("features", [])]
    if not version:
        raise ValueError("Feature registry has no version")
    if not features:
        raise ValueError("Feature registry has no features")
    if len(features) != len(set(features)):
        raise ValueError("Feature registry contains duplicate feature names")
    return version, features


def load_model_dataset(path: Path = DEFAULT_MODEL_DATASET) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow").copy()
    required = {"decision_date"}
    for horizon in HORIZONS:
        required.update(
            {
                f"forward_positive_{horizon}d",
                f"_forward_{horizon}d_known_date",
            }
        )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Model dataset missing required columns: {missing}")

    frame["decision_date"] = pd.to_datetime(
        frame["decision_date"], errors="raise"
    ).dt.normalize()
    for horizon in HORIZONS:
        column = f"_forward_{horizon}d_known_date"
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()

    frame = frame.sort_values("decision_date").reset_index(drop=True)
    if frame["decision_date"].duplicated().any():
        raise ValueError("Model dataset has duplicate decision dates")
    return frame


def validate_feature_columns(frame: pd.DataFrame, features: Iterable[str]) -> list[str]:
    names = list(features)
    missing = sorted(set(names).difference(frame.columns))
    if missing:
        raise ValueError(f"Model dataset missing registered features: {missing}")
    return names


def eligible_training_mask(
    frame: pd.DataFrame,
    horizon: int,
    cutoff: pd.Timestamp | str,
) -> pd.Series:
    """Rows legally trainable at a chronological cutoff.

    A row is eligible only when both the decision and the outcome-known date are
    on or before the training cutoff. This is stricter than filtering by the
    decision date alone and prevents partially matured labels from leaking into
    training.
    """

    if horizon not in HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon}")
    cutoff_ts = pd.Timestamp(cutoff).normalize()
    target = f"forward_positive_{horizon}d"
    known = f"_forward_{horizon}d_known_date"
    return (
        frame["decision_date"].le(cutoff_ts)
        & frame[known].notna()
        & frame[known].le(cutoff_ts)
        & frame[target].notna()
    )


def fold_test_mask(frame: pd.DataFrame, fold: dict[str, Any]) -> pd.Series:
    start = pd.Timestamp(fold["test_start"]).normalize()
    end = pd.Timestamp(fold["test_end"]).normalize()
    return frame["decision_date"].between(start, end, inclusive="both")
