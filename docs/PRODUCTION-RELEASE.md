# Production launch and operating runbook

## Release gates (must all pass before claiming production ready)

1. [ ] Repo and branch contain only intended product-source changes and dependency lockfile. `git diff --check` passes; no data ledger/evidence/checkpoint changes are staged.
2. [ ] Dependency audit: Node >=22, `npm ci`, `npm audit --omit=dev`, pinned lockfile, Python environment and `python -m pip check`; review advisories, licenses and updates.
3. [ ] Tests: frontend Node contracts, isolated Python publisher checks, existing v2.1 and V3 unit suites, sealed forward evidence and shadow verification, historical replay parity.
4. [ ] Publication: baseline generator, research append, React build and publisher, manifest digest validation; no changes to `analysis.json` or V3 JSON between before/after UI publication.
5. [ ] Playwright desktop and mobile browser smoke passes; navigate all four tabs, reload a hash route directly, test empty filters, stale warning, offline error state, keyboard-only and accessible label audit.
6. [ ] Security/privacy: validate third-party CDN/font policy, rights to reproduce the sentiment/market price feed, privacy/security/legal disclaimers, CSP and domain settings, GitHub Pages privacy model, absence of secrets in generated files.
7. [ ] Data integrity: no v3 research promotion, outcomes remain sealed, tactical sizing remains unchanged, no service-side API called by React.
8. [ ] Hosted PR checks pass; owner approves merge. After merge, both Build and Publish GitHub Pages jobs succeed. Verify deployed `site/publication_manifest.json` and that new React front page loads the same immutable contract.
9. [ ] Official branding / optional custom domain, metadata and search indexing decision documented. Noindex remains default until explicit review.
10. [ ] Post-deploy smoke and rollback rehearsal documented; monitor future scheduled ingestion, Pages deploy, stale observations and failed build alerts.

## Local commands

```bash
cd frontend
npm ci && npm run test && npm run build
cd ..
python scripts/build_dashboard.py --skip-yahoo-fallback
python -m v3.features.build_features && python -m v3.labels.build_labels && python -m v3.evaluation.validate_dataset
python -m v3.ci.run_treasury_feature_stage
python -m v3.evidence.verify_forward_lane
python -m v3.evidence.shadow_predictions verify
python scripts/build_v3_challenger.py
python scripts/publish_react_dashboard.py
python scripts/validate_react_publication.py
cd frontend
npx playwright install chromium
npm run test:e2e
```

## Observe

- CI: GitHub `Actions → Deploy Fear & Greed Dashboard` build/publish jobs, and V3/evidence integrity workflows.
- Pages: `https://george962.github.io/FearGreedIndex/index.html`, `publication_manifest.json`, `analysis.json`, `version.json`.
- Data freshness: source `analysis.latest.signal_date`, `analysis.generated_at`, shadow `latest_decision_date` and `available_forward_date`.
- Warning/error states must never silently display invented values.

## Failure / rollback

If React asset loading or schema validation fails, preserve the last successful Pages artifact. Use `legacy-dashboard.html` as an emergency viewing fallback; revert the frontend commit (owner authorization) to restore old landing page, and rerun deployment. If research verification fails, do not publish newly recomputed frozen evidence, bypass guards, rebase immutable ledgers or change method acceptance criteria. Root-cause and isolate historical source drift separately.

## Exact public site validation after deployment

The React page must replace the original `site/index.html` in the **existing** Pages workflow. The deploy job runs `python3 scripts/check_live_pages.py --attempts 10 --delay 12` after the official GitHub Pages deployment step. A PR preview does not deploy, and an old HTML page does not count as a passed release. The checker fails if manifest/version/build, bundled relative assets or research-only safety fail. This health check is not evidence of human keyboard/a11y/license approval.
