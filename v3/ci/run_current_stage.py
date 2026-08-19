#!/usr/bin/env python3
"""One-time V3-013 snapshot-integrity repair stage.

This temporary stage does not run or inspect the relative-strength ablation. It
only measures the already checked-in QQQ/SPY snapshot so its manifest can be
made atomically consistent. It will be replaced by the strict ablation stage
before V3-013 is merged.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "v3" / "data" / "qqq_spy_daily.csv.gz"
OUTPUT = ROOT / "v3" / "reports" / "relative_strength_snapshot_repair.json"


def main() -> int:
    snapshot_payload = SNAPSHOT.read_bytes()
    normalized_payload = gzip.decompress(snapshot_payload)
    frame = pd.read_csv(SNAPSHOT)
    required = {"date", "qqq_close", "spy_close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SystemExit(f"Checked-in QQQ/SPY snapshot missing columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame.empty or frame["date"].duplicated().any():
        raise SystemExit("Checked-in QQQ/SPY snapshot is empty or has duplicate dates")
    if not frame["date"].is_monotonic_increasing:
        raise SystemExit("Checked-in QQQ/SPY snapshot dates are not sorted")
    for column in ("qqq_close", "spy_close"):
        values = pd.to_numeric(frame[column], errors="raise")
        if values.isna().any() or (values <= 0.0).any():
            raise SystemExit(f"Checked-in snapshot has invalid {column} values")

    report = {
        "status": "SNAPSHOT_INTEGRITY_MEASURED",
        "rows": int(len(frame)),
        "start": frame["date"].min().date().isoformat(),
        "end": frame["date"].max().date().isoformat(),
        "normalized_sha256": hashlib.sha256(normalized_payload).hexdigest(),
        "snapshot_sha256": hashlib.sha256(snapshot_payload).hexdigest(),
        "normalized_columns": ["date", "qqq_close", "spy_close"],
        "snapshot_format": "gzip csv; deterministic mtime=0 expected",
        "note": "One-time measurement only; strict manifest enforcement returns before ablation.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
