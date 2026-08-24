#!/usr/bin/env python3
"""Render the committed research-only V3 challenger lane in the static dashboard.

The dashboard never trains or mutates the challenger. It reads the immutable
shadow prediction ledger, compares those recorded predictions with the existing
production decision history, and adds a clearly isolated research panel.
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

from v3.evaluation.stab003_consensus_abstention import evid001_outcomes_are_sealed
from v3.evidence.shadow_predictions import PREDICTION_LEDGER, read_prediction_ledger

FORWARD_LEDGER = ROOT / "v3" / "evidence" / "forward_feature_ledger.csv"
STAB004_EVALUATION = ROOT / "v3" / "reports" / "stab004_evaluation.json"
DEFAULT_SITE = ROOT / "site"
OUTPUT_NAME = "v3_challenger.json"
HISTORY_OUTPUT_NAME = "v3_challenger_history.csv"
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


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", "NA"):
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _latest_forward_date() -> str | None:
    if not FORWARD_LEDGER.exists():
        return None
    frame = pd.read_csv(FORWARD_LEDGER, usecols=["decision_date"])
    if frame.empty:
        return None
    dates = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
    return dates.max().strftime("%Y-%m-%d")


def _row_to_history(row: dict[str, str]) -> dict[str, Any]:
    return {
        "decision_date": row["decision_date"],
        "rolling_percentile": _float_or_none(row.get("rolling_percentile")),
        "raw_score": _float_or_none(row.get("raw_score")),
        "reference_count": int(row.get("reference_count", 0)),
        "call_state": row.get("call_state", "ABSTAIN"),
        "prediction_row_sha256": row.get("row_sha256"),
        "forward_feature_row_sha256": row.get("forward_feature_row_sha256"),
    }


def build_shadow_snapshot() -> dict[str, Any]:
    """Load the latest already-recorded shadow prediction without recomputing it."""
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

    rows = read_prediction_ledger(PREDICTION_LEDGER)
    latest_forward = _latest_forward_date()
    viability = evaluation.get("viability", {})
    base = {
        "mode": "RESEARCH_ONLY",
        "production_effect": "NONE",
        "method_id": "STAB-004",
        "method_status": evaluation.get("status"),
        "development_evidence_only": True,
        "available_forward_date": latest_forward,
        "prediction_ledger_rows": len(rows),
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

    if not rows:
        base.update(
            {
                "latest_decision_date": None,
                "call_state": "PREDICTION_LEDGER_PENDING",
                "rolling_percentile": None,
                "raw_score": None,
                "reference_count": 0,
                "interpretation": "Shadow prediction ledger is initializing",
                "representative_feature_count": 0,
                "representative_features": [],
                "prediction_ledger_current": False,
                "recent_history": [],
            }
        )
        return base

    latest = rows[-1]
    if latest.get("production_effect") != "NONE" or latest.get("sizing_multiplier") != "1":
        raise ValueError("Committed shadow prediction violated production isolation")
    percentile = _float_or_none(latest.get("rolling_percentile"))
    call_state = str(latest.get("call_state", "ABSTAIN"))
    if call_state == "STRONG_FAVORABLE":
        interpretation = "Strong favorable opportunity rank"
    elif call_state == "STRONG_UNFAVORABLE":
        interpretation = "Strong unfavorable opportunity rank"
    else:
        interpretation = "Middle-rank / abstain"
    features = [item for item in str(latest.get("representative_features", "")).split("|") if item]
    latest_date = latest["decision_date"]
    base.update(
        {
            "latest_decision_date": latest_date,
            "raw_score": _float_or_none(latest.get("raw_score")),
            "rolling_percentile": percentile,
            "reference_count": int(latest.get("reference_count", 0)),
            "call_state": call_state,
            "interpretation": interpretation,
            "representative_feature_count": int(latest.get("representative_feature_count", 0)),
            "representative_features": features,
            "prediction_row_sha256": latest.get("row_sha256"),
            "forward_feature_row_sha256": latest.get("forward_feature_row_sha256"),
            "prediction_ledger_current": latest_forward is None or latest_date == latest_forward,
            "recent_history": [_row_to_history(row) for row in rows[-20:]],
        }
    )
    return base


def build_comparison_history(site_dir: Path) -> pd.DataFrame:
    rows = read_prediction_ledger(PREDICTION_LEDGER)
    if not rows:
        return pd.DataFrame(
            columns=[
                "decision_date",
                "v3_opportunity_percentile",
                "v3_call_state",
                "production_action",
                "production_confidence",
                "production_timing_action",
            ]
        )

    shadow = pd.DataFrame(
        {
            "decision_date": [row["decision_date"] for row in rows],
            "v3_opportunity_percentile": [_float_or_none(row["rolling_percentile"]) for row in rows],
            "v3_call_state": [row["call_state"] for row in rows],
            "v3_reference_count": [int(row["reference_count"]) for row in rows],
            "v3_prediction_row_sha256": [row["row_sha256"] for row in rows],
        }
    )

    production_path = site_dir / "historical_decisions.csv"
    if not production_path.exists():
        shadow["production_action"] = ""
        shadow["production_confidence"] = ""
        shadow["production_timing_action"] = ""
        return shadow

    production = pd.read_csv(production_path)
    required = {"decision_date", "action"}
    if not required.issubset(production.columns):
        shadow["production_action"] = ""
        shadow["production_confidence"] = ""
        shadow["production_timing_action"] = ""
        return shadow

    keep = ["decision_date", "action"]
    for optional in ("confidence", "timing_action"):
        if optional in production.columns:
            keep.append(optional)
    production = production[keep].copy()
    production["decision_date"] = pd.to_datetime(production["decision_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    production = production.dropna(subset=["decision_date"]).drop_duplicates("decision_date", keep="last")
    production = production.rename(
        columns={
            "action": "production_action",
            "confidence": "production_confidence",
            "timing_action": "production_timing_action",
        }
    )
    merged = shadow.merge(production, on="decision_date", how="left", validate="one_to_one")
    for column in ("production_action", "production_confidence", "production_timing_action"):
        if column not in merged.columns:
            merged[column] = ""
        merged[column] = merged[column].fillna("")
    return merged


def _history_table(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""
    rows: list[str] = []
    for item in reversed(history[-8:]):
        percentile = item.get("v3_opportunity_percentile")
        percentile_text = "—" if percentile is None or pd.isna(percentile) else f"{float(percentile) * 100:.1f}%"
        rows.append(
            "<tr>"
            f"<td style='padding:6px 8px;text-align:left;'>{html.escape(str(item.get('decision_date', '—')))}</td>"
            f"<td style='padding:6px 8px;text-align:right;font-weight:800;'>{html.escape(percentile_text)}</td>"
            f"<td style='padding:6px 8px;text-align:left;'>{html.escape(str(item.get('v3_call_state', '—')))}</td>"
            f"<td style='padding:6px 8px;text-align:left;'>{html.escape(str(item.get('production_action') or '—'))}</td>"
            "</tr>"
        )
    return (
        "<div style='overflow:auto;margin-top:14px;border:1px solid rgba(130,160,210,.18);border-radius:10px;'>"
        "<table style='width:100%;border-collapse:collapse;font-size:.74rem;'>"
        "<thead><tr style='opacity:.65;'>"
        "<th style='padding:6px 8px;text-align:left;'>Date</th>"
        "<th style='padding:6px 8px;text-align:right;'>V3 rank</th>"
        "<th style='padding:6px 8px;text-align:left;'>V3 state</th>"
        "<th style='padding:6px 8px;text-align:left;'>Production</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def render_panel(snapshot: dict[str, Any]) -> str:
    percentile = snapshot.get("rolling_percentile")
    percentile_text = "—" if percentile is None else f"{float(percentile) * 100:.1f}%"
    auc = snapshot.get("frozen_validation", {}).get("mean_roc_auc")
    auc_text = "—" if auc is None else f"{float(auc):.3f}"
    feature_names = snapshot.get("representative_features", [])
    features_text = ", ".join(feature_names[:6]) if feature_names else "No recorded representatives yet"
    if len(feature_names) > 6:
        features_text += f" +{len(feature_names) - 6} more"
    current = bool(snapshot.get("prediction_ledger_current", False))
    freshness = "ledger current" if current else "latest forward row awaiting immutable prediction"
    history_html = _history_table(snapshot.get("comparison_history", []))

    return f"""
<section id="v3-research-challenger" style="margin:18px 0;padding:18px;border:1px solid rgba(130,160,210,.38);border-radius:14px;background:rgba(55,85,130,.10);">
  <div style="display:flex;justify-content:space-between;gap:16px;align-items:flex-start;flex-wrap:wrap;">
    <div>
      <div style="font-size:.72rem;font-weight:800;letter-spacing:.09em;color:#8fb8ff;">V3 RESEARCH CHALLENGER · IMMUTABLE SHADOW MODE</div>
      <h2 style="margin:.3rem 0 .35rem;">Opportunity Rank: {html.escape(percentile_text)}</h2>
      <div style="font-weight:750;">{html.escape(str(snapshot.get('interpretation', 'Unavailable')))}</div>
    </div>
    <div style="padding:7px 10px;border-radius:999px;border:1px solid rgba(238,197,90,.45);color:#e3c55a;font-size:.72rem;font-weight:850;">RESEARCH ONLY · NO PRODUCTION EFFECT</div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px;margin-top:14px;">
    <div><small style="opacity:.65;">Recorded decision date</small><div style="font-weight:800;">{html.escape(str(snapshot.get('latest_decision_date') or '—'))}</div></div>
    <div><small style="opacity:.65;">STAB state</small><div style="font-weight:800;">{html.escape(str(snapshot.get('call_state', '—')))}</div></div>
    <div><small style="opacity:.65;">Prior-score references</small><div style="font-weight:800;">{int(snapshot.get('reference_count', 0))}</div></div>
    <div><small style="opacity:.65;">Frozen mean AUC</small><div style="font-weight:800;">{html.escape(auc_text)}</div></div>
    <div><small style="opacity:.65;">Shadow history</small><div style="font-weight:800;">{int(snapshot.get('prediction_ledger_rows', 0))} immutable rows</div></div>
    <div><small style="opacity:.65;">Production sizing</small><div style="font-weight:800;">1.00x unchanged</div></div>
  </div>
  <p style="margin:14px 0 6px;line-height:1.5;font-size:.82rem;opacity:.82;">This panel reads committed point-in-time V3 predictions rather than recalculating history after the fact. EVID-001 outcomes remain sealed, and the challenger cannot change BUY / WAIT / HOLD or sizing. Status: {html.escape(freshness)}.</p>
  <p style="margin:6px 0 0;line-height:1.5;font-size:.76rem;opacity:.67;"><strong>Recorded representative signals:</strong> {html.escape(features_text)}. STAB-004 remains rejected for promotion because its frozen viability gate failed, despite strong ranking evidence.</p>
  {history_html}
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

    comparison = build_comparison_history(site_dir)
    history_path = site_dir / HISTORY_OUTPUT_NAME
    comparison.to_csv(history_path, index=False)
    enriched = dict(snapshot)
    enriched["comparison_history"] = comparison.tail(20).replace({np.nan: None}).to_dict(orient="records")

    original = index_path.read_text(encoding="utf-8")
    index_path.write_text(inject_panel(original, enriched), encoding="utf-8")
    (site_dir / OUTPUT_NAME).write_text(json.dumps(enriched, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_SITE)
    args = parser.parse_args()
    snapshot = build_shadow_snapshot()
    write_shadow_outputs(args.site_dir, snapshot)
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
