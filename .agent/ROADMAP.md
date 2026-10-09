# FGI-001–004 roadmap — one existing FearGreedIndex Pages URL

Updated 2026-10-09. This is an implementation/release plan, not proof of a production rollout. **Target**: `https://george962.github.io/FearGreedIndex/index.html` — no new site, domain, or server.

| ID | User-visible deliverable | Code phase | Acceptance required |
| --- | --- | --- | --- |
| FGI-001 | Credible public overview, purpose, sources, freshness, caveats and discoverability policy | Partial — existing PR #81 + React overview | Reconcile PR #81; privacy/data-rights/indexing review; CI pass |
| FGI-002 | Clean presentation separate from frozen analytics | Implemented in delivered React bundle and Python publisher | Real published artifact parity, full Python unit suite, compiled build and review |
| FGI-003 | Interactive historical explorer, evidence and research pages | Implemented in delivered React bundle | Actual-data Playwright/mobile/keyboard/accessibility/empty/stale/export tests |
| FGI-004 | Release reliability, contributor/review process, rollback, audit and monitoring | PR and release workflows/source implemented | npm lock, security review, hosted checks, owner approval, same-URL Pages deployment and live health check |

## Step-by-step final acceptance checklist

- [x] Establish production URL (`/FearGreedIndex/index.html`) and project-path-compatible relative assets.
- [x] Develop React frontend that consumes only current published Python JSON/CSV and keeps existing engine untouched.
- [x] Preserve original `site/analysis.json`, v3 shadow data, immutable evidence and fallback legacy HTML; add parity tests.
- [x] Implement historical filters, charts, descriptive matured-outcome summaries and CSV export without redefining historical model outcomes.
- [x] Add isolated contract tests (13 Python and 5 JS pass in source-package test environment).
- [x] Author PR build + browser QA, existing Pages deployment patch, and postdeploy exact-URL validator. Not yet hosted.
- [ ] Integrate against latest main, including the previous CI repairs and PR #81; resolve conflicts without changing research evidence.
- [ ] Generate and commit reviewed `frontend/package-lock.json`; pass `npm ci`, Vite build, Python pipeline and Playwright on Mac/CI.
- [ ] Inspect real site across devices, a11y keyboard/contrast, error/empty/stale states, dependency vulnerabilities and data redistribution rights.
- [ ] Review final source diff and hosted PR Actions; obtain owner approval and merge.
- [ ] Verify new React site at the **exact same GitHub Pages URL**; record build SHA, CI links, live-site test, rollback practice and accessibility acceptance.
- [ ] Mark milestone states COMPLETE only once all relevant real-world acceptance evidence is attached to `.agent/STATUS.md`.

## Immutable boundaries

Keep v2.1 frozen, V3 STAB-004 shadow only, EVID-001 sealed, `sizing_multiplier=1.0`, and public indexing set to `noindex,nofollow` until approved. Do not retune old evidence or claim trading profitability to make a UI showcase look more impressive. Research and product roadmaps are independent.
