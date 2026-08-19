#!/usr/bin/env python3
"""Record immutable lineage for the V3-017 minimal sizing layer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "v3" / "policy" / "sizing_v1.json"
ENGINE = ROOT / "v3" / "policy" / "sizing_policy.py"
OUTPUT = ROOT / "v3" / "reports" / "sizing_policy_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = {
        "sizing_version": config["sizing_version"],
        "status": config["status"],
        "config_sha256": sha256(CONFIG),
        "engine_sha256": sha256(ENGINE),
        "baseline_multiplier": config["baseline_multiplier"],
        "strong_add_multiplier": config["strong_add_multiplier"],
        "requires_promotion_ready_prediction": config["requires_promotion_ready_prediction"],
        "underweight_allowed": False,
        "larger_sizing_allowed": False,
        "current_candidate_activation": "BLOCKED",
        "empirical_sizing_test_status": "DEFERRED_UNTIL_PREDICTION_PROMOTION_GATE_PASSES"
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
