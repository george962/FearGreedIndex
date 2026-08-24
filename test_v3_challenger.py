import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_v3_challenger import inject_panel, write_shadow_outputs


class V3ShadowDashboardTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = {
            "mode": "RESEARCH_ONLY",
            "production_effect": "NONE",
            "method_id": "STAB-004",
            "method_status": "complete_reject",
            "latest_decision_date": "2026-08-21",
            "rolling_percentile": 0.91,
            "reference_count": 252,
            "call_state": "STRONG_FAVORABLE",
            "interpretation": "Strong favorable opportunity rank",
            "prediction_ledger_rows": 3,
            "prediction_ledger_current": True,
            "representative_features": ["spx_realized_vol_20", "spx_distance_ma_200"],
            "frozen_validation": {"mean_roc_auc": 0.671686, "viability_gate_pass": False},
            "guardrails": {
                "evid001_outcomes_opened": False,
                "champion_selected": False,
                "v3_019_eligible": False,
                "sizing_multiplier": 1.0,
                "production_action_changed": False,
            },
            "comparison_history": [
                {
                    "decision_date": "2026-08-21",
                    "v3_opportunity_percentile": 0.91,
                    "v3_call_state": "STRONG_FAVORABLE",
                    "production_action": "HOLD / NO EXTRA BUYING",
                }
            ],
        }

    def test_inject_panel_is_explicitly_research_only(self):
        output = inject_panel("<html><body><main><h1>Production</h1></main></body></html>", self.snapshot)
        self.assertIn('id="v3-research-challenger"', output)
        self.assertIn("RESEARCH ONLY · NO PRODUCTION EFFECT", output)
        self.assertIn("IMMUTABLE SHADOW MODE", output)
        self.assertIn("Opportunity Rank: 91.0%", output)
        self.assertIn("STRONG_FAVORABLE", output)
        self.assertIn("HOLD / NO EXTRA BUYING", output)

    def test_shadow_write_does_not_touch_production_analysis_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory)
            (site / "index.html").write_text("<html><body><main>Production</main></body></html>", encoding="utf-8")
            production = {"action": "HOLD / NO EXTRA BUYING", "sizing_multiplier": 1.0}
            analysis_path = site / "analysis.json"
            analysis_path.write_text(json.dumps(production, sort_keys=True), encoding="utf-8")
            before = analysis_path.read_bytes()

            write_shadow_outputs(site, self.snapshot)

            self.assertEqual(before, analysis_path.read_bytes())
            self.assertEqual(production, json.loads(analysis_path.read_text(encoding="utf-8")))
            shadow = json.loads((site / "v3_challenger.json").read_text(encoding="utf-8"))
            self.assertEqual(shadow["production_effect"], "NONE")
            self.assertFalse(shadow["guardrails"]["production_action_changed"])
            self.assertTrue((site / "v3_challenger_history.csv").exists())

    def test_duplicate_panel_fails_closed(self):
        with self.assertRaises(ValueError):
            inject_panel('<html><main><section id="v3-research-challenger"></section></main></html>', self.snapshot)


if __name__ == "__main__":
    unittest.main()
