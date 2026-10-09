# Fear & Greed Atlas — product architecture

## Non-negotiable boundaries

`data/* → scripts/build_dashboard.py → site/analysis.json + history` remains the **only** source of operational signal and historical replay decisions. `v3/evidence/*` and the frozen methodologies remain sealed and research-only; `scripts/build_v3_challenger.py` exports observational V3 JSON and CSV after immutable ledger verification. React reads those public exports. It never invokes Python models, backfills predictions, trains models, alters sizing or communicates with a broker.

## Layer separation / FGI-002

1. **Ingestion and research** — existing Python/Scheduled GitHub Actions; protected v2.1 and V3 research definitions.
2. **Publication / API contract** — existing JSON and CSV in `site/`; version.json carries build ID and generated timestamp. `scripts/publish_react_dashboard.py` validates provenance and all research-only guards, fingerprints produced data bytes, then publishes Vite static assets. No network request to an external API is introduced at browser runtime.
3. **UI presentation** — `frontend/src/data.js` parses/validates normalized published contract; `frontend/src/main.jsx` React routes/components; `frontend/src/styles.css` responsive design. No external authentication, account, server, or user telemetry.
4. **Transport** — GitHub Pages static assets; read-only HTTP fetches to same origin. Hash-based routing works under the `/FearGreedIndex/` subpath and with direct deep links.

## Dataset contract

Required: `site/analysis.json`, `site/historical_decisions.json`, `site/version.json`, `site/v3_challenger.json`, `site/v3_challenger_history.csv`. Existing Python generator also outputs CSV diagnostics. Optional shadow JSON is shown only when complete promotion/isolation guardrails are valid. JSON fields used: `analysis.latest.signal_date`, `analysis.latest.fear_greed`, `analysis.verdict`, `analysis.fast_timing`, `analysis.warnings`, `history.decisions[*].decision_date`, and published forward columns. No unpublished private data are added.

## Error behavior

- Missing essential JSON or incorrect schema: show a visible blocking error and retry; **no synthetic market data**.
- Missing/invalid V3 guardrails: hide model summary and show unverified research state.
- Sentiment age >4 calendar days: visibly warn about stale data. Market closures/weekends can increase age; use recorded source date rather than implying real-time status.
- Missing/not-yet-matured outcome: `—`, excluded from mature sample denominator; not interpreted as zero return.
- Version change: offer explicit reload rather than silently changing the dashboard during analysis.
- All interactions are read-only client-side sorting/filtering/export, never changing data files or evidence.

## Accessible interaction and performance

Mobile sidebar, labeled form controls, keyboard focus outlines, semantic tables, per-chart descriptive context, no animation when reduced motion is set. History plots downsample to at most about 450 points without altering underlying records; filtered tables paginate 20 rows. Data contract loads once per page session with AbortController. Playwright smoke tests cover desktop and mobile, navigation, filtering and V3 isolation. Automated Lighthouse and screen-reader/manual accessibility review remain release gates.

## Security and privacy

No secrets/client tokens. All third-party navigation uses `rel="noopener noreferrer"`. No HTML injection from model outputs (`React` escapes interpolated text). Exports guard spreadsheet-formula strings. Github Actions publish token scope remains as currently configured. Leave `noindex,nofollow` until owner approves data rights. No cookies or analytics SDK added. No external font service is loaded; system fonts provide a privacy-preserving default. No third-party telemetry is requested.

## CDN / client cache

Hashed Vite assets are cacheable. Required data fetches use `cache:'no-store'` and the existing version.json check prevents silent long-lived stale research. The deployment publishes all assets and data as one Pages artifact. Rolling back should use a known-good commit, **not** rebasing or rewriting evidence.
