"""FGI-001 contract smoke tests for generated public presentation.

These tests complement (do not replace) the existing runtime/backtest suites.
"""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from jinja2 import Environment, meta

ROOT = Path(__file__).resolve().parent
DASHBOARD = ROOT / "scripts" / "build_dashboard.py"


class PublicFoundationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = DASHBOARD.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(DASHBOARD))
        cls.dashboard_template = None
        for statement in cls.tree.body:
            if not isinstance(statement, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == 'PAGE_TEMPLATE' for target in statement.targets):
                continue
            if isinstance(statement.value, ast.Call) and statement.value.args and isinstance(statement.value.args[0], ast.Constant):
                cls.dashboard_template = statement.value.args[0].value
        if cls.dashboard_template is None:
            raise AssertionError('Cannot find static PAGE_TEMPLATE: revisit test contract rather than silently passing')

    def test_noindex_remains_explicit_on_both_pages(self):
        self.assertEqual(self.source.count('content="noindex,nofollow"'), 2)

    def test_dashboard_template_compiles_and_has_freshness_input(self):
        parsed = Environment().parse(self.dashboard_template)
        variables = meta.find_undeclared_variables(parsed)
        self.assertIn('observation_age_days', variables)
        self.assertIn('generated', variables)
        self.assertIn('metrics', variables)

    def test_onboarding_navigation_and_readable_sections(self):
        t = self.dashboard_template
        for target in ['main-content', 'market-context', 'historical-evidence', 'research-methodology']:
            self.assertIn(f'id="{target}"', t)
        self.assertIn('href="#main-content"', t)
        self.assertIn('research-only', t)
        self.assertIn('not a live quote', t)
        self.assertIn('Stale observation', t)
        self.assertIn('Recent observation', t)
        for link in re.findall(r'href="#([a-z-]+)"', t):
            self.assertIn(f'id="{link}"', t, msg=f'Unresolved in-page anchor #{link}')

    def test_public_source_is_escaped_in_generated_main_page(self):
        self.assertIn('source=html_lib.escape(source_label, quote=True)', self.source)
        self.assertIn('observation_age_days=signal_age_days', self.source)

    def test_history_page_has_navigation_and_retains_original_replay(self):
        self.assertIn('href="#history-main"', self.source)
        self.assertIn('id="history-main"', self.source)
        self.assertIn('decision_history.html', self.dashboard_template)
        self.assertIn('render_history_page(', self.source)

    def test_public_docs_and_agent_workflow_exist(self):
        for relative in ['AGENTS.md', '.agent/WORKFLOW.md', 'docs/USER_GUIDE.md', 'docs/PUBLICATION_PRECHECK.md']:
            self.assertTrue((ROOT / relative).is_file(), relative)
        agent_doc = (ROOT / 'AGENTS.md').read_text(encoding='utf-8')
        self.assertIn('.agent/WORKFLOW.md', agent_doc)
        self.assertNotIn('Atomic workflow', agent_doc)


if __name__ == '__main__':
    unittest.main()
