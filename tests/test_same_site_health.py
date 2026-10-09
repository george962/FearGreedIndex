"""Health-check contracts; network-free tests for the exact public Pages path."""
import json
import unittest
from urllib.parse import urlparse
from scripts.check_live_pages import verify_publication, SITE_URL


class SameSiteHealthTests(unittest.TestCase):
    def setUp(self):
        self.files = {
            '/FearGreedIndex/index.html': b'<html><head><meta name="robots" content="noindex,nofollow"></head><body><div id="root"></div><script src="./assets/index-a1.js"></script></body></html>',
            '/FearGreedIndex/assets/index-a1.js': b'console.log(1)',
            '/FearGreedIndex/publication_manifest.json': json.dumps({'deployment_target':SITE_URL,'build_id':'abc','assets':['assets/index-a1.js']}).encode(),
            '/FearGreedIndex/version.json': b'{"build_id":"abc"}',
            '/FearGreedIndex/analysis.json': b'{"latest":{"signal_date":"2026-10-09"},"warnings":[]}',
            '/FearGreedIndex/v3_challenger.json': json.dumps({'mode':'RESEARCH_ONLY','production_effect':'NONE','guardrails':{'champion_selected':False,'production_action_changed':False,'v3_019_eligible':False,'evid001_outcomes_opened':False,'sizing_multiplier':1.0}}).encode(),
            '/FearGreedIndex/historical_decisions.json': b'{"decisions":[]}',
            '/FearGreedIndex/legacy-dashboard.html': b'legacy',
        }

    def fetch(self,url):
        return self.files[urlparse(url).path]

    def test_existing_url_passes(self):
        self.assertEqual(verify_publication(self.fetch)['status'],'PASS')

    def test_other_public_path_rejected(self):
        with self.assertRaisesRegex(ValueError,'existing'):
            verify_publication(self.fetch,'https://george962.github.io/other/index.html')

    def test_promoted_shadow_rejected(self):
        value=json.loads(self.files['/FearGreedIndex/v3_challenger.json'])
        value['guardrails']['champion_selected']=True
        self.files['/FearGreedIndex/v3_challenger.json']=json.dumps(value).encode()
        with self.assertRaisesRegex(ValueError,'Unsafe'):
            verify_publication(self.fetch)

    def test_broken_assets_rejected(self):
        self.files['/FearGreedIndex/index.html']=self.files['/FearGreedIndex/index.html'].replace(b'./assets/',b'/assets/')
        with self.assertRaisesRegex(ValueError,'relative'):
            verify_publication(self.fetch)

if __name__=='__main__':unittest.main()
