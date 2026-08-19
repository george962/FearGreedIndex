#!/usr/bin/env python3
"""V3-017 minimal sizing layer gated by prediction-promotion status."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "v3" / "policy" / "sizing_v1.json"
VALID_ACTIONS = {"STRONG ADD", "ADD MODESTLY", "BASELINE", "WAIT FOR BETTER ENTRY"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if float(config["minimum_multiplier"]) < 1.0:
        raise ValueError("V3-017 may not underweight below 1.00x")
    if float(config["maximum_multiplier"]) > 1.10:
        raise ValueError("V3-017 may not exceed 1.10x")
    return config


def size_action(
    action: str,
    *,
    promotion_ready_prediction: bool,
    experiment_id: str,
    decision_policy_version: str,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unknown decision action: {action}")
    if not experiment_id.strip():
        raise ValueError("experiment_id is required")
    if not decision_policy_version.strip():
        raise ValueError("decision_policy_version is required")

    multiplier = float(config["baseline_multiplier"])
    reason = "BASELINE_SIZING"
    activation_blocked = False

    if action == config["eligible_action"]:
        if promotion_ready_prediction:
            multiplier = float(config["strong_add_multiplier"])
            reason = "VALIDATED_STRONG_ADD_1_10X"
        else:
            reason = "PREDICTION_PROMOTION_GATE_NOT_MET"
            activation_blocked = True

    if multiplier < 1.0 or multiplier > 1.10:
        raise AssertionError("V3-017 produced an out-of-policy multiplier")

    return {
        "sizing_version": config["sizing_version"],
        "sizing_status": config["status"],
        "sizing_config_sha256": _sha256(config_path),
        "action": action,
        "multiplier": multiplier,
        "reason_code": reason,
        "promotion_ready_prediction": bool(promotion_ready_prediction),
        "activation_blocked": activation_blocked,
        "experiment_id": experiment_id,
        "decision_policy_version": decision_policy_version,
        "underweight_allowed": False,
        "larger_sizing_allowed": False,
    }
