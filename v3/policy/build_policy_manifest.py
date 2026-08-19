#!/usr/bin/env python3
"""Record immutable lineage for the frozen V3-016 research policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "v3" / "policy" / "policy_v1.json"
ENGINE = ROOT / "v3" / "policy" / "decision_policy.py"
OUTPUT = ROOT / "v3" / "reports" / "decision_policy_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    manifest = {
        "policy_version": policy["policy_version"],
        "policy_status": policy["status"],
        "threshold_origin": policy["threshold_origin"],
        "policy_config_sha256": sha256(POLICY),
        "policy_engine_sha256": sha256(ENGINE),
        "action_vocabulary": list(policy["action_semantics"].keys()),
        "sizing_defined": False,
        "sell_or_underweight_allowed": False,
        "champion_required_for_production": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
