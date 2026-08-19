#!/usr/bin/env python3
"""Hash the frozen V3-018 gate config and engine for immutable lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "v3" / "evaluation" / "champion_gates_v1.json"
DEFAULT_ENGINE = ROOT / "v3" / "evaluation" / "champion_gates.py"
DEFAULT_OUTPUT = ROOT / "v3" / "reports" / "champion_gate_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("gate_version") != "v3-champion-gates-001":
        raise SystemExit("Unexpected V3-018 gate version")
    if config.get("frozen_before_current_candidate_evaluation") is not True:
        raise SystemExit("Champion gate config is not marked pre-registered/frozen")
    if config.get("fail_closed") is not True:
        raise SystemExit("Champion gate config must be fail-closed")
    manifest = {
        "gate_version": config["gate_version"],
        "status": config["status"],
        "pre_registered_issue": config["pre_registered_issue"],
        "config_sha256": sha256(args.config),
        "engine_sha256": sha256(args.engine),
        "promotion_rule": config["promotion_rule"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
