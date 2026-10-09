# FGI-002 — Isolate production presentation from the research engine

## Big picture
Deliver a modern maintained frontend without touching historically frozen signal calculations. React/Vite is a separate presentation layer, fed from versioned public JSON/CSV.

## Implemented locally
- [x] `frontend/src/data.js` centralizes parsers, guards and data contracts.
- [x] `frontend/src/main.jsx` and `styles.css` implement independent source-controlled UI.
- [x] `vite.config.js` supports the GitHub Pages subpath, dev data serving and production assets.
- [x] `scripts/publish_react_dashboard.py` saves `legacy-dashboard.html`, swaps new landing HTML and copies versioned assets.
- [x] Same script checks SHA-256 of each generated JSON/CSV before and after UI publication.
- [x] `scripts/validate_react_publication.py` verifies artifact references, manifest and research status.

## Still to accept
- [ ] Build real `site/` from frozen Python runtime; SHA parity tests pass.
- [ ] `npm ci`, Node tests, Vite production build, browser smoke checks and all Python suites pass.
- [ ] Hosted PR checks and Pages deployment pass, owner validates no unintended model impact.

## Non-goals
Do not move or refactor legacy Python research functions in this deliverable. Preserve legacy HTML and scripts for fallback; this is presentation decoupling, not an engine rewrite.

## Rollback
Revert React publication/deploy workflow changes; restore legacy build. Never regenerate immutable research evidence.

## Next command
`cd frontend && npm install && npm run check` followed by the repository release runner.

## Same-site release checkpoint (2026-10-09)

React/Vite now owns the presentation and publishes `site/index.html` at the same existing Pages URL; Python retains read-only model contracts. Full artifact parity and npm build remain acceptance gates.
