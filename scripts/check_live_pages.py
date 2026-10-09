#!/usr/bin/env python3
"""Read-only post-deployment health check for the EXISTING GitHub Pages URL.

Does not create, merge, deploy, or mutate repo files. Checks the public CDN
and fails if a legacy page, missing bundle, inconsistent evidence or unexpected
research promotion is published. Reruns briefly to tolerate Pages propagation.
"""
from __future__ import annotations

import argparse
import json
import time
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

SITE_URL = 'https://george962.github.io/FearGreedIndex/index.html'


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets: list[str] = []
        self.robots: list[str] = []
        self.root_seen = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'div' and attrs.get('id') == 'root':
            self.root_seen = True
        if tag == 'meta' and attrs.get('name') == 'robots':
            self.robots.append(attrs.get('content', ''))
        if tag == 'script' and attrs.get('src'):
            self.assets.append(attrs['src'])
        if tag == 'link' and attrs.get('rel') == 'stylesheet' and attrs.get('href'):
            self.assets.append(attrs['href'])


def http_get(url: str, *, max_bytes=4_000_000, timeout=12):
    with urlopen(Request(url, headers={'User-Agent': 'FearGreedIndex-ReleaseCheck/1.0', 'Cache-Control':'no-cache'}), timeout=timeout) as response:
        if response.status != 200:
            raise ValueError(f'{url}: HTTP {response.status}')
        final = urlparse(response.url)
        if final.scheme != 'https' or final.netloc != 'george962.github.io':
            raise ValueError(f'Unexpected redirect outside Pages host: {response.url}')
        payload = response.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ValueError(f'Response too large: {url}')
        return payload


def verify_publication(fetch=http_get, site_url=SITE_URL) -> dict:
    page = urlparse(site_url)
    if page.scheme != 'https' or page.netloc != 'george962.github.io' or page.path != '/FearGreedIndex/index.html':
        raise ValueError('Release must target existing FearGreedIndex/index.html path')
    html = fetch(site_url).decode('utf-8')
    parser = Assets()
    parser.feed(html)
    if not parser.root_seen or 'noindex,nofollow' not in parser.robots:
        raise ValueError('React root or public indexing guardrail missing')
    if not parser.assets or any(not x.startswith('./assets/') or '/..' in x for x in parser.assets):
        raise ValueError('React script/styles are missing or not relative to the existing Pages path')
    manifest = json.loads(fetch(urljoin(site_url, './publication_manifest.json')))
    version = json.loads(fetch(urljoin(site_url, './version.json')))
    analysis = json.loads(fetch(urljoin(site_url, './analysis.json')))
    shadow = json.loads(fetch(urljoin(site_url, './v3_challenger.json')))
    if manifest.get('deployment_target') != SITE_URL or manifest.get('build_id') != version.get('build_id'):
        raise ValueError('Publication target or version/build identifier inconsistent')
    if not analysis.get('latest', {}).get('signal_date') or not isinstance(analysis.get('warnings'),list):
        raise ValueError('Missing current published analysis or warnings')
    guards = shadow.get('guardrails',{})
    if not (shadow.get('mode') == 'RESEARCH_ONLY'
            and shadow.get('production_effect') == 'NONE'
            and guards.get('champion_selected') is False
            and guards.get('production_action_changed') is False
            and guards.get('v3_019_eligible') is False
            and guards.get('evid001_outcomes_opened') is False
            and guards.get('sizing_multiplier') == 1.0):
        raise ValueError('Unsafe published shadow state')
    for asset in parser.assets:
        if asset[2:] not in manifest.get('assets',[]):
            raise ValueError(f'Asset absent from manifest: {asset}')
        fetch(urljoin(site_url, asset))
    for name in ('./historical_decisions.json','./legacy-dashboard.html'):
        fetch(urljoin(site_url,name))
    return {'status':'PASS','site_url':site_url,'build_id':version['build_id'],'assets_checked':len(parser.assets),'research_only':True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempts',type=int,default=6)
    parser.add_argument('--delay',type=float,default=10)
    args=parser.parse_args()
    if args.attempts < 1 or args.delay < 0:
        raise SystemExit('Invalid attempts or delay')
    for i in range(args.attempts):
        try:
            print(json.dumps(verify_publication(),indent=2))
            return
        except Exception as err:
            print(f'Attempt {i+1}/{args.attempts}: {err}',flush=True)
            if i+1 == args.attempts:
                raise SystemExit('FAIL: Existing GitHub Pages release not yet healthy') from err
            time.sleep(args.delay)

if __name__=='__main__':main()
