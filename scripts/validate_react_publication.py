#!/usr/bin/env python3
"""Validate a generated Pages artifact before deployment; no network required."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
PUBLIC_SITE_URL = 'https://george962.github.io/FearGreedIndex/index.html'
PUBLIC_SITE_PATH = '/FearGreedIndex/index.html'
from pathlib import Path


def verify(site: Path) -> dict:
    manifest = json.loads((site/'publication_manifest.json').read_text(encoding='utf-8'))
    html = (site/'index.html').read_text(encoding='utf-8')
    if '<div id="root"></div>' not in html or '<meta name="robots" content="noindex,nofollow"' not in html:
        raise ValueError('React landing HTML does not satisfy public presentation contract')
    if not (site/'legacy-dashboard.html').is_file() or not (site/'decision_history.html').is_file():
        raise ValueError('Legacy fallback or historical HTML is missing')
    if manifest.get('deployment_target') != PUBLIC_SITE_URL or manifest.get('deployment_path') != PUBLIC_SITE_PATH:
        raise ValueError('Wrong Pages deployment target; expected existing /FearGreedIndex/index.html')
    if './assets/' not in html or 'src="/assets/' in html or 'href="/assets/' in html:
        raise ValueError('Bundled HTML would not load under the existing GitHub Pages project path')
    if not manifest.get('assets'):
        raise ValueError('React asset manifest is empty')
    for asset in manifest['assets']:
        path = Path(asset)
        if path.is_absolute() or '..' in path.parts or not (site/path).is_file():
            raise ValueError(f'Missing or unsafe React asset {asset}')
        if not re.search(r'\.(?:js|css|svg|woff2?)$', asset):
            raise ValueError(f'Unexpected file type in bundled asset {asset}')
    for name, expected in manifest['data_artifact_sha256'].items():
        actual = hashlib.sha256((site/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Published data changed after frontend build: {name}')
    shadow = json.loads((site/'v3_challenger.json').read_text(encoding='utf-8'))
    if shadow.get('mode') != 'RESEARCH_ONLY' or shadow.get('production_effect') != 'NONE':
        raise ValueError('V3 research has not been isolated')
    if manifest['research_mode'] != 'RESEARCH_ONLY' or manifest['production_effect'] != 'NONE':
        raise ValueError('Publication manifest attempted an unapproved research promotion')
    return {'status':'PASS','data_artifacts':len(manifest['data_artifact_sha256']),'assets':len(manifest['assets']),'historical_fallback':True,'search_indexing':'noindex','deployment_target':PUBLIC_SITE_URL}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--site-dir', type=Path, default=Path(__file__).resolve().parents[1]/'site')
    args = p.parse_args()
    print(json.dumps(verify(args.site_dir), indent=2))


if __name__ == '__main__':
    main()
