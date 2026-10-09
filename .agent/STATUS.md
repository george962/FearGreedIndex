# FearGreedIndex product engineering — same-site checkpoint

Updated: 2026-10-09 UTC, local package delivery. This does not represent a GitHub commit, PR merge or deployment.

**Confirmed final public URL:** `https://george962.github.io/FearGreedIndex/index.html`. React must replace the `site/index.html` artifact in the existing GitHub Pages deployment. The previous separate worktree suggestion was solely for Git isolation, not for a separate public website. Normal feature-branch installation in the existing checkout is now documented.

| Milestone | Local source readiness | Acceptance / next step |
| --- | --- | --- |
| FGI-001 Public foundation | PARTIAL — PR #81 still open when last checked | Reconcile it against current main and confirm data-publication rights, metadata and noindex; do not claim merged |
| FGI-002 Modularization/presentation | SOURCE IMPLEMENTED — original Python signal engine unchanged; React/Vite bundles separately; original JSON/CSV contract preserved | Full npm build + Python parity + review of generated `site/index.html` |
| FGI-003 Interactive explorer | SOURCE IMPLEMENTED — chart periods, historical filters, mature return denominators, exports, read-only V3 research | Desktop/mobile Playwright plus keyboard, accessibility and actual-data checks |
| FGI-004 Reliable release | INTEGRATION CHECKS BUILT — read-only PR job, post-deploy exact-URL health check, rollback and audit docs | Generate lockfile, run complete release tests, hosted CI and owner-approved merge; verify live Pages after deployment |

### Evidence available from this delivered package

- Local isolated Python unit tests: 13 PASS (publication contracts, page target, installer, postdeploy checker).
- Local JavaScript Node data-contract tests: 5 PASS.
- Syntax: Python scripts, Node JS modules and known workflow configuration checked locally.
- **Not run here:** Vite dependency installation/bundle, full Python engine, Playwright Chromium, GitHub Actions on new PR, fresh Pages deployment, and independent live React/browser/a11y/performance acceptance.

### Boundaries and blockers

Research methods and evidence remain unchanged. V3 remains RESEARCH_ONLY, no champion promotion or tactical sizing. Source rights, branding, indexing, dependency security, browser acceptance and owner-approved deployment remain release gates. Previous `main` Pages workflow successfully deployed the legacy site, not this React package.

### Exact next action

Read `README.md`, install in a clean branch of the **existing repository**, generate `frontend/package-lock.json`, run `tools/run_release_checks.sh`, and report output. Reconcile PR #81 before owner-approved release. Do not treat an unmerged PR or successful source tests as a live release.
