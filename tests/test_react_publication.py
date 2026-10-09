"""Isolated publication contract tests; does not require the research engine."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from scripts.publish_react_dashboard import publish
from scripts.validate_react_publication import verify


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.site = self.root / 'site'
        self.dist = self.root / 'dist'
        self.site.mkdir()
        (self.dist / 'assets').mkdir(parents=True)
        (self.dist/'assets'/'app-a1.js').write_text('console.log("hello");')
        (self.dist/'index.html').write_text('<html><head><meta name="robots" content="noindex,nofollow" /></head><body><div id="root"></div><script src="./assets/app-a1.js"></script></body></html>')
        (self.site/'index.html').write_text('<html><body>legacy</body></html>')
        (self.site/'decision_history.html').write_text('<html><body>history</body></html>')
        (self.site/'analysis.json').write_text(json.dumps({'latest':{'signal_date':'2026-09-01','fear_greed':22},'warnings':[],'verdict':{'action':'WAIT'}}))
        (self.site/'historical_decisions.json').write_text(json.dumps({'decisions':[{'decision_date':'2026-09-01'}]}))
        (self.site/'version.json').write_text(json.dumps({'build_id':'abc','generated_at':'2026-09-01'}))
        (self.site/'v3_challenger.json').write_text(json.dumps({'mode':'RESEARCH_ONLY','production_effect':'NONE','guardrails':{'production_action_changed':False,'champion_selected':False,'v3_019_eligible':False,'evid001_outcomes_opened':False,'sizing_multiplier':1.0}}))
        (self.site/'v3_challenger_history.csv').write_text('decision_date\n2026-09-01\n')

    def test_publish_preserves_json_and_provides_fallback(self):
        previous = hashlib.sha256((self.site/'analysis.json').read_bytes()).hexdigest()
        result = publish(self.site,self.dist)
        self.assertEqual(result['data_artifact_sha256']['analysis.json'],previous)
        self.assertIn('legacy',(self.site/'legacy-dashboard.html').read_text())
        self.assertEqual(verify(self.site)['status'],'PASS')

    def test_no_new_research_permissions(self):
        shadow=json.loads((self.site/'v3_challenger.json').read_text())
        shadow['guardrails']['champion_selected']=True
        (self.site/'v3_challenger.json').write_text(json.dumps(shadow))
        with self.assertRaisesRegex(RuntimeError,'isolation'):
            publish(self.site,self.dist)

    def test_missing_required_data_fails_closed(self):
        (self.site/'historical_decisions.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'missing generated'):
            publish(self.site,self.dist)

    def test_does_not_overwrite_legacy_fallback(self):
        publish(self.site,self.dist)
        publish(self.site,self.dist)
        self.assertIn('legacy',(self.site/'legacy-dashboard.html').read_text())

    def test_exact_existing_pages_target(self):
        result = publish(self.site,self.dist)
        self.assertEqual(result['deployment_target'], 'https://george962.github.io/FearGreedIndex/index.html')
        self.assertEqual(result['deployment_path'], '/FearGreedIndex/index.html')
        self.assertEqual(verify(self.site)['deployment_target'], result['deployment_target'])

    def test_root_relative_assets_are_rejected(self):
        (self.dist / 'index.html').write_text('<html><head><meta name="robots" content="noindex,nofollow" /></head><body><div id="root"></div><script src="/assets/app-a1.js"></script></body></html>')
        with self.assertRaisesRegex(RuntimeError, 'relative'):
            publish(self.site,self.dist)

    def test_tampered_data_fails_manifest_verification(self):
        publish(self.site,self.dist)
        (self.site/'analysis.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'changed'):
            verify(self.site)

if __name__ == '__main__':unittest.main()
