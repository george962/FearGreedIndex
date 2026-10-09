#!/usr/bin/env python3
"""Publish a separate React presentation without modifying analytics or v3 evidence.

Run AFTER the legacy production builder and the shadow append step. This script
only replaces the landing HTML and adds hashed frontend assets + an integrity
manifest. It preserves original HTML as a fallback and proves that every
produced data artifact remains byte-identical before/after frontend publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SITE_URL = 'https://george962.github.io/FearGreedIndex/index.html'
PUBLIC_SITE_PATH = '/FearGreedIndex/index.html'
DATA_ARTIFACTS = (
    'analysis.json', 'historical_decisions.json', 'historical_decisions.csv',
    'event_study.csv', 'analogs.csv', 'full_analysis.csv',
    'timing_evaluation.csv', 'decision_changes.csv',
    'timing_decision_changes.csv', 'version.json',
    'v3_challenger.json', 'v3_challenger_history.csv',
)
REQUIRED = ('analysis.json', 'historical_decisions.json', 'version.json', 'v3_challenger.json', 'v3_challenger_history.csv')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_contract(site: Path) -> dict[str, str]:
    missing = [x for x in REQUIRED if not (site / x).is_file()]
    if missing:
        raise RuntimeError(f'React publication refuses missing generated artifacts: {missing}')
    analysis = json.loads((site / 'analysis.json').read_text(encoding='utf-8'))
    history = json.loads((site / 'historical_decisions.json').read_text(encoding='utf-8'))
    shadow = json.loads((site / 'v3_challenger.json').read_text(encoding='utf-8'))
    version = json.loads((site / 'version.json').read_text(encoding='utf-8'))
    if not isinstance(analysis.get('warnings'), list) or not analysis.get('latest', {}).get('signal_date'):
        raise RuntimeError('Production analysis is missing required data fields')
    if not isinstance(history.get('decisions'), list):
        raise RuntimeError('Historical payload is missing decisions')
    if not version.get('build_id'):
        raise RuntimeError('version.json is missing build_id')
    guards = shadow.get('guardrails', {})
    if not (
        shadow.get('mode') == 'RESEARCH_ONLY'
        and shadow.get('production_effect') == 'NONE'
        and guards.get('production_action_changed') is False
        and guards.get('champion_selected') is False
        and guards.get('v3_019_eligible') is False
        and guards.get('evid001_outcomes_opened') is False
        and guards.get('sizing_multiplier') == 1.0
    ):
        raise RuntimeError('Shadow research-only isolation contract failed')
    return {name: digest(site / name) for name in DATA_ARTIFACTS if (site / name).is_file()}


def publish(site: Path, dist: Path) -> dict[str, object]:
    site = site.resolve()
    dist = dist.resolve()
    if not site.is_dir() or not dist.is_dir():
        raise RuntimeError('Both the generated site/ and Vite dist/ directories are required')
    index = dist / 'index.html'
    assets = dist / 'assets'
    if not index.is_file() or not assets.is_dir():
        raise RuntimeError('Vite must produce index.html and assets/. Run npm run build first.')
    html = index.read_text(encoding='utf-8')
    if '<div id="root"></div>' not in html or '<meta name="robots" content="noindex,nofollow"' not in html:
        raise RuntimeError('React landing HTML is missing root element or public discovery guardrail')
    if './assets/' not in html or 'src="/assets/' in html or 'href="/assets/' in html:
        raise RuntimeError('Vite landing HTML must reference relative ./assets/ for /FearGreedIndex/index.html')
    if '<base href=' in html:
        raise RuntimeError('A base URL can break GitHub Pages project paths')
    before = ensure_contract(site)
    old_index = site / 'index.html'
    if not old_index.is_file():
        raise RuntimeError('Legacy dashboard must be built before React publication')
    legacy = site / 'legacy-dashboard.html'
    # A fresh legacy build may replace site/index.html on each release.
    # On repeated frontend-only publish calls, never back up React as legacy.
    if '<div id="root"></div>' not in old_index.read_text(encoding='utf-8'):
        shutil.copyfile(old_index, legacy)
    elif not legacy.exists():
        raise RuntimeError('Missing legacy fallback while React index is already published')
    previous_assets = site / 'assets'
    if previous_assets.exists():
        if previous_assets.is_symlink() or not previous_assets.is_dir():
            raise RuntimeError('Unsafe destination: site/assets must be a real directory')
        shutil.rmtree(previous_assets)
    shutil.copytree(assets, previous_assets, symlinks=False)
    shutil.copyfile(index, old_index)
    (site / '.nojekyll').touch()
    after = ensure_contract(site)
    if before != after:
        raise AssertionError('Frontend publication mutated Python-generated evidence')
    manifest = {
        'schema_version': 1,
        'frontend': 'react-vite',
        'deployment_target': PUBLIC_SITE_URL,
        'deployment_path': PUBLIC_SITE_PATH,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'research_mode': 'RESEARCH_ONLY',
        'production_effect': 'NONE',
        'default_search_indexing': 'noindex,nofollow',
        'build_id': json.loads((site / 'version.json').read_text(encoding='utf-8'))['build_id'],
        'data_artifact_sha256': after,
        'assets': sorted(path.relative_to(site).as_posix() for path in previous_assets.rglob('*') if path.is_file()),
        'legacy_url': './legacy-dashboard.html',
    }
    (site / 'publication_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--site-dir', type=Path, default=ROOT/'site')
    parser.add_argument('--dist', type=Path, default=ROOT/'frontend'/'dist')
    args = parser.parse_args()
    result = publish(args.site_dir, args.dist)
    print(json.dumps({k: v for k, v in result.items() if k != 'data_artifact_sha256'}, indent=2))
    print(f'Publication contract validated for {len(result["data_artifact_sha256"])} immutable-by-frontend data artifacts.')
    print(f'GitHub Pages target (on owner-approved main deployment): {PUBLIC_SITE_URL}')


if __name__ == '__main__':
    main()
