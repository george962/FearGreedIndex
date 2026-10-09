# FGI-001 — Make the existing FearGreedIndex dashboard publicly understandable

This is a living ExecPlan governed by `.agent/PLANS.md` and `AGENTS.md`.

## Purpose / Big Picture

A first-time visitor should understand what the application analyzes, when its observations were updated, how to interpret historical evidence and its uncertainty, and where to find documentation. A contributor should be able to reproduce a local build. These changes must not modify v2.1 signals, v3 research, or immutable ledgers.

## Scope and Non-goals

In scope: README/overview/navigation, metadata, usability, freshness and sources display, documentation, publication-contract checks, accessibility fixes in the generated presentation, and a reproducible smoke build. Out of scope: new predictive models, retuning thresholds, historical performance claims, backend services, authentication, trading execution, and major frontend rewrite.

## Progress

- [ ] Preflight: inspect deployed Pages configuration and latest actual source state; document what the public URL currently serves. Verify data-license/public visibility requirements; record missing approvals.
- [ ] Establish baseline: capture generated JSON schema, HTML/asset list, output timestamps, and representative production decision from a clean rebuild; never change historical decisions to satisfy parity.
- [ ] Implement public information architecture and onboarding: clear name, purpose, user-oriented documentation, repository homepage link suggestion, mobile and keyboard basics.
- [ ] Add transparent source/freshness and research-status presentation; stale/unavailable data must not masquerade as live; v3 must remain research-only.
- [ ] Add or update targeted tests for publication contracts and readable metadata; verify existing dashboard, ledger, and v3 output invariants.
- [ ] Run and record targeted + relevant regression commands, inspect output, update `.agent/STATUS.md`, write final review and handoff.

## Surprises & Discoveries

- None observed yet. Record findings during preflight.

## Decision Log

- 2026-10-09, starter-kit author: prioritize public usability and explicit evidence instead of introducing a backend. Confirm source rights/public URL before enabling search discovery.

## Context and Orientation

- `README.md`: currently describes frozen v2.1 and research-only v3; preserve links and research rules.
- `README-dashboard.md`: operational explanation of the Pages build, data collection and caveats.
- `scripts/build_dashboard.py`: currently generates static `site/` presentation and output files; large, coupled, and protected against semantic changes.
- `scripts/build_v3_challenger.py`: attaches a research-only shadow presentation; must not affect production `site/analysis.json`.
- `.github/workflows/deploy-pages.yml`: deploys generated Pages artifact on main and downstream workflow completion.
- `site/` files are generated, not manually maintained. Do not hand-edit build outputs as a durable change.
- Both main and history HTML templates were observed to contain `noindex,nofollow`; do not remove until owner/publication approval.
- `v3/STATUS.md` describes a no-champion research state; re-check current state.

## Plan of Work

Start with read-only inspection of code, deployment workflow, public site, documentation, and data provenance. Create a representative baseline by running existing tests and dashboard generation in an isolated checkout. Design a minimal onboarding/freshness presentation using only fields with verified provenance. Prefer small changes to the dashboard presentation and README; if a generated field is needed, add it to a versioned contract without changing decision functions. Review every HTML/JSON change for escaping, source freshness, search visibility, and accessibility. Add tests, build locally, and compare semantic decision and history outputs against baseline. Keep the PR limited to this outcome.

## Concrete Steps

From repository root on a new feature branch, after dependencies are installed:

1. `git status --short && git branch --show-current` — record branch and outstanding changes; don't overwrite existing work.
2. `python -m unittest -v test_feargreed test_fear_greed_market_data test_dashboard test_signal_ledger test_strategy_validation test_v3_challenger` — baseline must be captured, or a real failure documented.
3. `python scripts/build_dashboard.py` — record generated file set and schema (do not commit generated outputs unless repository policy requires it).
4. Inspect `site/analysis.json`, `site/version.json`, and the source files to identify production decision, timestamps, and warnings without guessing field names.
5. Implement smallest changes and add test coverage for the new public contract.
6. Repeat both test/build commands. For v3 presentation touches, follow the existing deploy workflow's verify steps and confirm a stable production JSON hash.
7. Record manual keyboard/mobile review and exact GitHub Pages URL result once actually verified.

## Validation and Acceptance

- NOT RUN: baseline and after-change unit tests above; expected zero failures.
- NOT RUN: static build command; expected generated `site/index.html`, `site/styles.css`, `site/app.js`, `site/analysis.json`, `site/version.json`.
- NOT RUN: publication metadata tests; expected accurate source/freshness/limitations with accessible navigation.
- NOT RUN: semantic parity comparison of production decisions and historical replay; expected no change except explicitly allowed presentation fields.
- NOT RUN: live public URL check and mobile keyboard smoke tests; expected usable site, accurate loading/error states, correct public visibility.
- NOT RUN: licensing/attribution review; public data must not be newly redistributed until cleared.

## Risks and Stop Conditions

Approval required to enable search crawling, change public data scope, use third-party assets/data without known rights, alter Pages secrets/settings, deploy to `main`, or change frozen research contracts. Do not simply remove `noindex` and call the site launch-ready.

## Outcomes & Retrospective

- Not implemented. Fill in links, changes, tests, commits, risks, and next milestone when accepted.
