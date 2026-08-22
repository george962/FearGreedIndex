#!/usr/bin/env python3
"""Build a research-only V3 challenger snapshot and inject it into the static dashboard.

This module is intentionally downstream of the production dashboard build. It may
read frozen V3 research artifacts and append-only EVID-001 feature rows, but it
must not change the production decision engine, action, sizing, or analysis.json.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from v3.evaluation.exp006_opportunity import AS_OF, add_opportunity_targets, eligible_training_mask
from v3.evaluation.stab001_past_only import select_stable_features
from v3.evaluation.stab003_consensus_abstention import (
    build_consensus,
    consensus_score,
    evid001_outcomes_are_sealed,
    select_short_memory_features,
)
from v3.evaluation.stab004_rolling_normalization import (
    MIN_REPRESENTATIVES,
    redundancy_clusters,
    rolling_score_percentiles,
)
from v3.models.common import load_feature_registry, validate_feature_columns

HISTORICAL_DATASET = ROOT / "v3" / "data" / "model_dataset_treasury.parquet"
FEATURE_REGISTRY = ROOT / "v3" / "reports" / "feature_registry_treasury.json"
FORWARD_LEDGER = ROOT / "v3" / "evidence" / "forward_feature_ledger.csv"
STAB004_EVALUATION = ROOT / "v3" / "reports" / "stab004_evaluation.json"
DEFAULT_SITE = ROOT / "site"
OUTPUT_NAME = "v3_challenger.json"
PANEL_MARKER = 'id="v3-research-challenger"'


def _plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if pd.isna(value):
        return None
    return value


def build_shadow_snapshot() -> dict[str, Any]:
    """Score the append-only forward feature ledger without opening forward outcomes."""
    if not evid001_outcomes_are_sealed():
        raise ValueError("V3 shadow dashboard refuses to run after EVID-001 outcomes are opened")

    evaluation = json.loads(STAB004_EVALUATION.read_text(encoding="utf-8"))
    if evaluation.get("method_id") != "STAB-004":
        raise ValueError("Unexpected V3 challenger method")
    if evaluation.get("evid001_outcomes_opened") is not False:
        raise ValueError("Frozen STAB-004 evidence says EVID-001 was opened")
    if evaluation.get("champion_selected") is not False or evaluation.get("v3_019_eligible") is not False:
        raise ValueError("Research-only challenger cannot be a promoted champion")
    if float(evaluation.get("current_sizing_multiplier", 0.0)) != 1.0:
        raise ValueError("Research-only challenger cannot change sizing")

    feature_version, features = load_feature_registry(FEATURE_REGISTRY)
    historical = pd.read_parquet(HISTORICAL_DATASET, engine="pyarrow").copy()
    historical["decision_date"] = pd.to_datetime(historical["decision_date"], errors="raise").dt.normalize()
    historical = historical.loc[historical["decision_date"] <= AS_OF].sort_values("decision_date").reset_index(drop=True)
    historical = add_opportunity_targets(historical)
    validate_feature_columns(historical, features)

    train = historical.loc[eligible_training_mask(historical, AS_OF)].copy()
    if train.empty:
        raise ValueError("No legally mature pre-cutoff training rows for V3 shadow scoring")

    forward = pd.read_csv(FORWARD_LEDGER)
    if forward.empty:
        return {
            "mode": "RESEARCH_ONLY",
            "production_effect": "NONE",
            "method_id": "STAB-004",
            "method_status": evaluation.get("status"),
            "latest_decision_date": None,
            "call_state": "NO_FORWARD_FEATURE_ROW",
            "rolling_percentile": None,
            "reference_count": 0,
            "representative_features": [],
            "note": "EVID-001 has no post-cutoff feature rows yet.",
        }

    forward["decision_date"] = pd.to_datetime(forward["decision_date"], errors="raise").dt.normalize()
    if not forward["decision_date"].gt(AS_OF).all():
        raise ValueError("Forward feature ledger contains a decision date at/before the frozen cutoff")
    if "research_cutoff" in forward.columns:
        cutoffs = pd.to_datetime(forward["research_cutoff"], errors="raise").dt.normalize()
        if not cutoffs.eq(AS_OF).all():
            raise ValueError("Forward feature ledger research cutoff drift")
    if "feature_set_version" in forward.columns and not forward["feature_set_version"].eq(feature_version).all():
        raise ValueError("Forward feature ledger feature version drift")
    if "feature_count" in forward.columns and not pd.to_numeric(forward["feature_count"], errors="raise").eq(len(features)).all():
        raise ValueError("Forward feature ledger feature count drift")
    validate_feature_columns(forward, features)

    long_selected, _ = select_stable_features(train, features)
    short_selected, _ = select_short_memory_features(train, features)
    consensus = build_consensus(long_selected, short_selected)
    representatives, _ = redundancy_clusters(train, consensus)
    if len(representatives) < MIN_REPRESENTATIVES:
        raise ValueError("V3 shadow scoring lacks STAB-004 structural support")

    historical_scores = historical[["decision_date", *features]].copy()
    forward_scores = forward[["decision_date", *features]].copy()
    scoring_frame = (
        pd.concat([historical_scores, forward_scores], ignore_index=True)
        .sort_values("decision_date")
        .drop_duplicates(subset="decision_date", keep="last")
        .reset_index(drop=True)
    )
    raw_scores = consensus_score(train, scoring_frame, representatives)
    target_dates = set(forward["decision_date"].tolist())
    rolling = rolling_score_percentiles(scoring_frame["decision_date"], raw_scores, target_dates)
    latest = rolling.sort_values("decision_date").iloc[-1]

    percentile = _plain(latest.get("rolling_percentile"))
    call_state = str(latest.get("call_state", "ABSTAIN"))
    if call_state == "STRONG_FAVORABLE":
        interpretation = "Strong favorable opportunity rank"
    elif call_state == "STRONG_UNFAVORABLE":
        interpretation = "Strong unfavorable opportunity rank"
    else:
        interpretation = "Middle-rank / abstain"

    viability = evaluation.get("viability", {})
    return {
        "mode": "RESEARCH_ONLY",
        "production_effect": "NONE",
        "method_id": "STAB-004",
        "method_status": evaluation.get("status"),
        "development_evidence_only": True,
        "latest_decision_date": _plain(latest["decision_date"]),
        "raw_score": _plain(latest.get("raw_score")),
        "rolling_percentile": percentile,
        "reference_count": int(latest.get("reference_count", 0)),
        "call_state": call_state,
        "interpretation": interpretation,
        "representative_feature_count": len(representatives),
        "representative_features": [str(item["feature"]) for item in representatives],
        "frozen_validation": {
            "mean_roc_auc": _plain(viability.get("mean_roc_auc")),
            "minimum_fold_roc_auc": _plain(viability.get("minimum_fold_roc_auc")),
            "support_folds": _plain(viability.get("support_folds")),
            "viability_gate_pass": bool(viability.get("viability_gate_pass", False)),
            "reason_not_promoted": "Frozen STAB-004 viability gate failed; 2025 coverage was 0.552 versus the pre-registered 0.55 maximum.",
        },
        "guardrails": {
            "evid001_outcomes_opened": False,
            "champion_selected": False,
            "v3_019_eligible": False,
            "sizing_multiplier": 1.0,
            "production_action_changed": False,
        },
    }


def render_panel(snapshot: dict[str, Any]) -> str:
    percentile = snapshot.get("rolling_percentile")
    percentile_text = "—" if percentile is None else f"{float(percentile) * 100:.1f}%"
    auc = snapshot.get("frozen_validation", {}).get("mean_roc_auc")
    auc_text = "—" if auc is None else f"{float(auc):.3f}"
    feature_names = snapshot.get("representative_features", [])
    features_text = ", ".join(feature_names[:6]) if feature_names else "No supported representatives"
    if len(feature_names) > 6:
        features_text += f" +{len(feature_names) - 6} more"

    return f"""
<section id="v3-research-challenger" style="margin:18px 0;padding:18px;border:1px solid rgba(130,160,210,.38);border-radius:14px;background:rgba(55,85,130,.10);">
  <div style="display:flex;justify-content:space-between;gap:16px;align-items:flex-start;flex-wrap:wrap;">
    <div>
      <div style="font-size:.72rem;font-weight:800;letter-spacing:.09em;color:#8fb8ff;">V3 RESEARCH CHALLENGER · SHADOW MODE</div>
      <h2 style="margin:.3rem 0 .35rem;">Opportunity Rank: {html.escape(percentile_text)}</h2>
      <div style="font-weight:750;">{html.escape(str(snapshot.get('interpretation', 'Unavailable')))}</div>
    </div>
    <div style="padding:7px 10px;border-radius:999px;border:1px solid rgba(238,197,90,.45);color:#e3c55a;font-size:.72rem;font-weight:850;">RESEARCH ONLY · NO PRODUCTION EFFECT</div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px;margin-top:14px;">
    <div><small style="opacity:.65;">Decision date</small><div style="font-weight:800;">{html.escape(str(snapshot.get('latest_decision_date') or '—'))}</div></div>
    <div><small style="opacity:.65;">STAB state</small><div style="font-weight:800;">{html.escape(str(snapshot.get('call_state', '—')))}</div></div>
    <div><small style="opacity:.65;">Prior-score references</small><div style="font-weight:800;">{int(snapshot.get('reference_count', 0))}</div></div>
    <div><small style="opacity:.65;">Frozen mean AUC</small><div style="font-weight:800;">{html.escape(auc_text)}</div></div>
    <div><small style="opacity:.65;">Production sizing</small><div style="font-weight:800;">1.00x unchanged</div></div>
  </div>
  <p style="margin:14px 0 6px;line-height:1.5;font-size:.82rem;opacity:.82;">The challenger uses only frozen pre-2026-08-19 outcome relationships plus post-cutoff point-in-time feature rows. It does not read forward realized outcomes and cannot change BUY / WAIT / HOLD or sizing.</p>
  <p style="margin:6px 0 0;line-height:1.5;font-size:.76rem;opacity:.67;"><strong>Current representative signals:</strong> {html.escape(features_text)}. STAB-004 remains rejected for promotion because its frozen viability gate failed, despite strong ranking evidence.</p>
</section>
""".strip()


def inject_panel(index_html: str, snapshot: dict[str, Any]) -> str:
    if PANEL_MARKER in index_html:
        raise ValueError("V3 research challenger panel already exists in dashboard HTML")
    panel = render_panel(snapshot)
    if "</main>" in index_html:
        return index_html.replace("</main>", panel + "\n</main>", 1)
    if "</body>" in index_html:
        return index_html.replace("</body>", panel + "\n</body>", 1)
    raise ValueError("Dashboard HTML has no </main> or </body> insertion point")


def write_shadow_outputs(site_dir: Path, snapshot: dict[str, Any]) -> None:
    index_path = site_dir / "index.html"
    if not index_path.exists():
        raise FileNotFoundError(f"Dashboard index not found: {index_path}")
    original = index_path.read_text(encoding="utf-8")
    index_path.write_text(inject_panel(original, snapshot), encoding="utf-8")
    (site_dir / OUTPUT_NAME).write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_SITE)
    args = parser.parse_args()
    snapshot = build_shadow_snapshot()
    write_shadow_outputs(args.site_dir, snapshot)
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
