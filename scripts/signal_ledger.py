#!/usr/bin/env python3
"""Create an immutable decision ledger and fill outcomes only after they mature."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_common import (  # noqa: E402
    current_prediction,
    load_context,
    plain,
)


IMMUTABLE_FIELDS = [
    "decision_date",
    "market_date",
    "fear_greed",
    "fg_change_5",
    "market_regime",
    "market_extension",
    "action",
    "confidence",
    "sizing_tier",
    "sizing_label",
    "timing_action",
    "timing_side",
    "timing_score",
    "timing_confirmation_count",
    "timing_confirmation_total",
    "analog_sample",
    "regime_baseline_sample",
    "win_rate_5d",
    "average_5d",
    "average_20d",
    "regime_baseline_5d",
    "excess_5d",
    "excess_ci_low_5d",
    "excess_ci_high_5d",
    "average_drawdown_20d",
    "strategy_version",
    "config_sha256",
    "data_source",
]

LEDGER_COLUMNS = [
    "prediction_sha256",
    "recorded_at_utc",
    *IMMUTABLE_FIELDS,
    "entry_date",
    "entry_price",
    "realized_1d",
    "realized_5d",
    "realized_10d",
    "realized_20d",
    "realized_60d",
    "realized_max_drawdown_20d",
    "outcomes_last_updated_utc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "strategy_manifest.json",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "data" / "signal_ledger.csv",
    )
    parser.add_argument("--skip-yahoo-fallback", action="store_true")
    return parser.parse_args()


def _json_scalar(value: Any) -> Any:
    value = plain(value)
    if isinstance(value, float):
        return round(value, 12)
    return value


def prediction_hash(row: dict[str, Any]) -> str:
    canonical = {
        field: _json_scalar(row.get(field))
        for field in IMMUTABLE_FIELDS
    }
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in LEDGER_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[LEDGER_COLUMNS].copy()


def append_prediction(
    ledger: pd.DataFrame,
    prediction: dict[str, Any],
    *,
    recorded_at_utc: str,
) -> tuple[pd.DataFrame, bool]:
    row = {column: "" for column in LEDGER_COLUMNS}
    for field in IMMUTABLE_FIELDS:
        value = prediction.get(field)
        row[field] = "" if value is None else value
    row["recorded_at_utc"] = recorded_at_utc
    row["prediction_sha256"] = prediction_hash(prediction)

    if (
        not ledger.empty
        and ledger["prediction_sha256"].astype(str).eq(
            row["prediction_sha256"]
        ).any()
    ):
        return ledger, False

    output = pd.concat(
        [ledger, pd.DataFrame([row])],
        ignore_index=True,
    )
    return output[LEDGER_COLUMNS], True


def _event_lookup(events: pd.DataFrame) -> dict[str, pd.Series]:
    output: dict[str, pd.Series] = {}
    for _, row in events.iterrows():
        decision_date = pd.Timestamp(row["signal_date"]).date().isoformat()
        output[decision_date] = row
    return output


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return "" if not math.isfinite(number) else f"{number:.12g}"
    return str(value)


def update_matured_outcomes(
    ledger: pd.DataFrame,
    events: pd.DataFrame,
    *,
    updated_at_utc: str,
) -> tuple[pd.DataFrame, int]:
    if ledger.empty:
        return ledger, 0

    lookup = _event_lookup(events)
    output = ledger.copy()
    changed_rows = 0

    mapping = {
        "entry_date": "entry_date",
        "entry_price": "entry_price",
        "realized_1d": "forward_1d",
        "realized_5d": "forward_5d",
        "realized_10d": "forward_10d",
        "realized_20d": "forward_20d",
        "realized_60d": "forward_60d",
        "realized_max_drawdown_20d": "max_drawdown_20d",
    }

    for index, ledger_row in output.iterrows():
        event = lookup.get(str(ledger_row["decision_date"]))
        if event is None:
            continue

        row_changed = False
        for ledger_column, event_column in mapping.items():
            if event_column not in event.index:
                continue
            new_value = _format_value(event.get(event_column))
            old_value = str(ledger_row.get(ledger_column, ""))
            if new_value and not old_value:
                output.at[index, ledger_column] = new_value
                row_changed = True

        if row_changed:
            output.at[index, "outcomes_last_updated_utc"] = updated_at_utc
            changed_rows += 1

    return output[LEDGER_COLUMNS], changed_rows


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    context = load_context(
        args.config,
        allow_yahoo=not args.skip_yahoo_fallback,
    )
    prediction = current_prediction(
        context,
        args.manifest,
    )
    now = datetime.now(timezone.utc).isoformat()

    ledger = load_ledger(args.ledger)
    ledger, appended = append_prediction(
        ledger,
        prediction,
        recorded_at_utc=now,
    )
    ledger, matured = update_matured_outcomes(
        ledger,
        context.events,
        updated_at_utc=now,
    )

    atomic_write_csv(ledger, args.ledger)

    print(
        json.dumps(
            {
                "ledger": str(args.ledger),
                "prediction_appended": appended,
                "matured_rows_updated": matured,
                "latest_prediction_sha256": prediction_hash(prediction),
                "strategy_version": prediction["strategy_version"],
                "action": prediction["action"],
                "timing_action": prediction["timing_action"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
