#!/usr/bin/env python3
"""Apply the FGI-001 presentation-only changes to an existing repo checkout.

One-time local helper. It is deliberately kept outside scripts/ so it does not
become a production runtime dependency. Run from the repository root:

  python extras/apply_fgi001.py --check
  python extras/apply_fgi001.py

Changes only scripts/build_dashboard.py. Preserve a clean Git worktree first.
"""
from __future__ import annotations

import argparse
import ast
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "build_dashboard.py"

INTRO = '''    <main class="workspace" id="main-content" tabindex="-1">
      <section class="panel onboarding-panel" aria-labelledby="intro-title">
        <p class="eyebrow">MARKET SENTIMENT RESEARCH</p>
        <h2 id="intro-title">What can today's fear and greed tell us?</h2>
        <p>Compare the latest recorded sentiment with historically similar S&amp;P 500 market conditions.
           Explore subsequent returns, downside outcomes, and the amount of supporting evidence.</p>
        <div class="publication-facts" aria-label="Observation and report dates">
          <p><strong>Sentiment observed:</strong> {{ metrics[0].note }}</p>
          <p><strong>Data status:</strong> {{ 'Stale observation — ' ~ observation_age_days ~ ' calendar days old' if observation_age_days > 4 else 'Recent observation — ' ~ observation_age_days ~ ' calendar days old' }}</p>
          <p><strong>Report generated:</strong> {{ generated }}</p>
        </div>
        <p class="research-disclaimer">This is a periodically generated research snapshot, not a live quote or investment recommendation.
           Historical analogs do not establish predictive performance. Experimental v3 models remain research-only.</p>
        <div class="intro-links">
          <a href="#historical-evidence">Explore historical outcomes</a>
          <a href="#research-methodology">Understand the methodology</a>
        </div>
      </section>
      {% if warnings %}'''

ADDED_CSS = '''a { color: var(--accent); }
a:focus-visible { outline: 3px solid #e2c76e; outline-offset: 3px; }
.skip-link { position: fixed; left: 12px; top: -100px; z-index: 9999; padding: 10px 14px; border-radius: 8px; background: #fff; color: #101827; }
.skip-link:focus { top: 12px; }
.primary-nav { display: flex; flex-wrap: wrap; gap: 7px; }
.primary-nav a { padding: 7px 10px; border: 1px solid var(--border); border-radius: 8px; font-size: .78rem; text-decoration: none; }
.primary-nav a:hover, .primary-nav a:focus-visible { border-color: var(--accent); }
.onboarding-panel h2 { font-size: 1.25rem; margin: 2px 0 8px; }
.onboarding-panel > p:not(.eyebrow) { color: var(--muted); font-size: .88rem; line-height: 1.55; }
.publication-facts { margin: 14px 0; display: flex; flex-wrap: wrap; gap: 9px 18px; }
.publication-facts p { color: var(--text); font-size: .78rem; }
.publication-facts strong { color: #c8d4e8; }
.intro-links { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px; font-size: .83rem; font-weight: 700; }
.intro-links a { text-underline-offset: 3px; }
.shell {'''

REPLACEMENTS = [
    (
        '  <title>Fear &amp; Greed Market Dashboard</title>',
        '  <title>Fear &amp; Greed | Evidence-first market research</title>\n'
        '  <meta name="description" content="Explore market sentiment, historical analogs, and their limitations. Research only; not live trading advice.">',
    ),
    (
        '<body data-build-id="{{ build_id }}" data-refresh-seconds="{{ refresh_seconds }}">',
        '<body data-build-id="{{ build_id }}" data-refresh-seconds="{{ refresh_seconds }}">\n'
        '  <a class="skip-link" href="#main-content">Skip to market research</a>',
    ),
    (
        '      <div class="gauge-card">\n        {{ gauge | safe }}',
        '      <nav class="primary-nav" aria-label="Research sections">\n'
        '        <a href="#market-context">Market context</a>\n'
        '        <a href="#historical-evidence">Historical evidence</a>\n'
        '        <a href="#research-methodology">How to interpret</a>\n'
        '        <a href="decision_history.html">Decision history</a>\n'
        '      </nav>\n\n'
        '      <div class="gauge-card">\n        {{ gauge | safe }}',
    ),
    ('    <main class="workspace">\n      {% if warnings %}', INTRO),
    ('      <section class="panel chart-panel">\n        <div class="panel-heading">',
     '      <section class="panel chart-panel" id="market-context">\n        <div class="panel-heading">'),
    ('      <section class="panel history-panel">',
     '      <section class="panel history-panel" id="historical-evidence">'),
    ('      <section class="panel note-panel">',
     '      <section class="panel note-panel" id="research-methodology">'),
    ('a { color: var(--accent); }\n.shell {', ADDED_CSS),
    ('        source=source_label,\n        generated=generated.strftime("%Y-%m-%d %H:%M UTC"),',
     '        source=html_lib.escape(source_label, quote=True),\n'
     '        generated=generated.strftime("%Y-%m-%d %H:%M UTC"),\n'
     '        observation_age_days=signal_age_days,'),
    ('  <title>Fear &amp; Greed Decision History</title>',
     '  <title>Fear &amp; Greed | Historical decision research</title>\n'
     '  <meta name="description" content="Point-in-time replay of historical Fear and Greed research decisions, including uncertainty and limits.">'),
    ('<body>\n  <main class="history-shell">',
     '<body>\n  <a class="skip-link" href="#history-main">Skip to historical decisions</a>\n'
     '  <main class="history-shell" id="history-main" tabindex="-1">'),
]


def apply(text: str) -> str:
    """Fail closed on ambiguous or previously modified templates."""
    if 'id="intro-title"' in text:
        raise ValueError("Onboarding already appears to be applied. Do not run this tool twice.")
    for before, after in REPLACEMENTS:
        count = text.count(before)
        if count != 1:
            raise ValueError(f"Expected one matching anchor, got {count}: {before[:100]!r}")
        text = text.replace(before, after, 1)
    ast.parse(text, filename=str(SOURCE))
    if text.count('noindex,nofollow') != 2:
        raise ValueError("Indexing guard has changed; refusing to continue")
    if not all(token in text for token in ('observation_age_days=signal_age_days', 'id="research-methodology"', 'source=html_lib.escape')):
        raise ValueError("Generated output is missing expected public presentation fields")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate all anchors without writing')
    args = parser.parse_args()
    original = SOURCE.read_text(encoding='utf-8')
    changed = apply(original)
    if args.check:
        print('PASS: compatible source and safe FGI-001 presentation changes; nothing written')
        return
    fd, temp_name = tempfile.mkstemp(prefix='.build_dashboard.', suffix='.py', dir=SOURCE.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(changed)
        os.chmod(temp_name, SOURCE.stat().st_mode)
        os.replace(temp_name, SOURCE)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    print('Updated scripts/build_dashboard.py locally. Review git diff and run tests; no GitHub operation performed.')


if __name__ == '__main__':
    main()
