# FGI-004 — Production-quality frontend delivery and rollout

## Outcome
A deterministic, audited, static build with a safe rollback and no changes to v2.1 / V3 core calculations, including no accidental promotion of STAB-004.

## Local code delivered
- [x] Pinned direct npm dependency versions and Vite production configuration; no long-running frontend server in Pages.
- [x] Python publication fingerprint manifest and checks proving byte-level JSON/CSV parity.
- [x] Dev and source error states fail visibly, with no fabricated observations.
- [x] Immutable research guardrails are rechecked both server-side and client-side.
- [x] GitHub Pages workflow patch adds Node build, static publish and desktop/mobile Playwright checks.
- [x] Tests cover malformed data, CSV injection, negative outcomes, filters and research-only enforcement.
- [x] Legacy fallback maintained; deployment/rollback and operating runbook documented.

## Must still complete before marking Done
- [ ] Generate `frontend/package-lock.json` with actual npm registry (not forged); `npm ci` reproducibility and `npm audit` review.
- [ ] All Python, Node, Playwright and integration tests pass locally and in GitHub Actions; Git diffs clean.
- [ ] Real browser audit across desktop and mobile; contrast, keyboard and screen reader review.
- [ ] Review data licensing, external font policy, privacy and security disclosures, logo/domain, exact search indexing choice.
- [ ] Owner-approved merge after CI passes and smoke test of actually deployed React Pages URL.
- [ ] Document production release SHA, build manifest and rollback test in `.agent/STATUS.md`.

## Risks and constraints
Untrusted PRs cannot get repository secrets or write tokens from product CI. Host deployment only after owner approval on protected `main` branches. Preserve current GitHub Actions read-only research isolation. Prevent stale feeds being labeled 'live'. No user financial information or telemetry in Pages.

## Exact next step
Run `tools/install.py --check` on a fresh checkout, then follow README source validation, npm lockfile and release tests. A packaged implementation is not a completed production deployment.

## Same-site release checkpoint (2026-10-09)

A read-only pull-request CI workflow validates frontend before merge. After owner-approved Pages deployment, `scripts/check_live_pages.py` verifies the exact existing `/FearGreedIndex/index.html` URL, asset paths, evidence contracts and shadow safety. Do not call this milestone complete until hosted tests and live checks pass.
