# FGI-004 — Make publication, releases, and contributions reproducible

This is a living ExecPlan governed by `.agent/PLANS.md` and `AGENTS.md`.

## Purpose / Big Picture

A contributor should be able to reproduce the software; a maintainer should know whether a deployment is valid, which data version it contains, and how to roll back safely when ingestion or presentation fails.

## Scope and Non-goals

In scope: CI and end-to-end smoke checks, artifact provenance and freshness validation, startup/setup documentation, contributor and security process, release notes policy, Pages failure recovery, dependency and data rights review. Out of scope: unapproved automatic merges/deployments, vendor service purchases, trading activity, and modification of research evidence.

## Progress

- [ ] Inventory and document current scheduled ingestion → ledger → build → Pages pipeline and precise failure modes.
- [ ] Establish deterministic build and schema checks; fail publishing if mandatory contracts are invalid, preserve last successful output for data outages.
- [ ] Add minimal browser smoke tests and publication/artifact integrity checks with clear logs.
- [ ] Document local development, reproduction, contribution/testing, security reporting, releases/rollback, and public data-source limitations.
- [ ] Verify independent reviewer findings are resolved and permissions cannot write sensitive data from untrusted PRs.
- [ ] Run actual CI on a feature PR (if authorized), record results, obtain explicit owner approval before merge/deploy.

## Surprises & Discoveries

- None yet.

## Decision Log

- 2026-10-09, starter-kit author: use GitHub Pages and GitHub Actions as existing operational infrastructure; avoid introducing a production server without need.

## Context and Orientation

- Existing workflows include `.github/workflows/deploy-pages.yml`, `market_data.yml`, `forward_evidence_collect.yml`, and many targeted research validations. Preserve chain/order/permissions rather than consolidating blindly.
- Repo has `requirements.txt`, `README-dashboard.md`, and `test_*.py` plus `v3/tests/`.
- Frozen v2.1 operational and v3 immutable-evidence constraints apply to workflow jobs and caches.

## Plan of Work

Document existing triggers, permissions and artifacts; classify trusted versus untrusted execution paths. Add publication validation for schemas/required assets/immutable read-only research rendering, browser smoke checks, stale-data behavior, and recovery on failure. Prepare contributor-facing setup and security guidance; clarify credentials, source licensing, testing, how to propose a PR, and what is not supported. Validate the exact workflow edits using YAML parse and applicable workflow lint where available, and actual hosted CI rather than a theoretical green label.

## Concrete Steps

1. Inspect all relevant `.github/workflows/*.yml` triggers, credentials and race/concurrency guards.
2. Run `python -m unittest -v test_feargreed test_fear_greed_market_data test_dashboard test_signal_ledger test_strategy_validation test_v3_challenger` and `python -m unittest discover -s v3/tests -p 'test_*.py'`.
3. Build site with `python scripts/build_dashboard.py`, run new publication and browser smoke checks against generated content.
4. Review YAML syntax, least-privilege GITHUB_TOKEN permissions, and PR trust boundaries before changing CI.
5. If authorized, test a feature PR and inspect actual Actions checks and artifact before owner-approved merge/deploy.

## Validation and Acceptance

- NOT RUN: unit/research test commands; zero new failures.
- NOT RUN: static publication artifacts and schema integrity; no missing source/freshness status.
- NOT RUN: browser smoke cases; key pages and error states behave as expected.
- NOT RUN: GitHub Actions validation and hosted PR run; logs and permission boundaries documented.
- NOT RUN: documented rollback rehearsal; no rewriting evidence and no exposing secrets.

## Risks and Stop Conditions

Stop for approval before changing hosted permissions, security settings, paid infrastructure, source licensing, public URL, merging, or deployment. Do not delete immutable evidence to get tests green.

## Outcomes & Retrospective

- Not implemented.
