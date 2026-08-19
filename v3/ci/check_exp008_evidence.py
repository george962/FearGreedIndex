#!/usr/bin/env python3
"""Verify frozen EXP-008 evidence and manifest contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "v3" / "experiments" / "EXP-008" / "manifest.json"
EVALUATION = ROOT / "v3" / "reports" / "exp008_evaluation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))

    if manifest.get("experiment_id") != "EXP-008":
        raise SystemExit("EXP-008 manifest experiment ID drift")
    if manifest.get("status") != "complete_reject":
        raise SystemExit("EXP-008 manifest must remain complete_reject")
    if manifest.get("sentiment_state", {}).get("extreme_fear_max") != 25.0:
        raise SystemExit("EXP-008 extreme-fear threshold drift")
    if manifest.get("sentiment_state", {}).get("extreme_greed_min") != 75.0:
        raise SystemExit("EXP-008 extreme-greed threshold drift")

    expected = manifest.get("evidence", {})
    if not expected:
        raise SystemExit("EXP-008 manifest has no frozen evidence hashes")
    for relative, expected_hash in expected.items():
        path = ROOT / relative
        if not path.exists():
            raise SystemExit(f"Missing EXP-008 evidence file: {relative}")
        actual = sha256(path)
        if actual != expected_hash:
            raise SystemExit(
                f"EXP-008 evidence hash mismatch for {relative}: {actual} != {expected_hash}"
            )

    if evaluation.get("decision") != "DO_NOT_ADVANCE_SENTIMENT_EXTREMES_UNDER_EXP_008":
        raise SystemExit("EXP-008 frozen decision drift")
    if evaluation.get("experiment_viability_pass") is not False:
        raise SystemExit("EXP-008 must remain a rejected viability result")
    if evaluation.get("v3_019_eligible") is not False:
        raise SystemExit("EXP-008 must not unlock V3-019")
    if float(evaluation.get("current_sizing_multiplier")) != 1.0:
        raise SystemExit("EXP-008 must not change sizing")

    viability = evaluation.get("viability", {})
    if int(viability.get("hypothesized_prevalence_ordering_folds", -1)) != 0:
        raise SystemExit("EXP-008 prevalence-ordering evidence drift")
    if viability.get("sample_hashes_match_exp006") is not True:
        raise SystemExit("EXP-008 must preserve EXP-006 realized-date samples")

    print("EXP-008 frozen evidence: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
