# FGI-002 — Separate presentation assets from the dashboard analysis generator

This is a living ExecPlan governed by `.agent/PLANS.md` and `AGENTS.md`.

## Purpose / Big Picture

Maintainers should be able to adjust UI markup, styles, and frontend logic without editing an enormous Python generator or risking silent changes to market signals. The public dashboard must continue to display the same production and shadow research facts as before the refactor.

## Scope and Non-goals

In scope: extract static CSS, JS, Jinja2 templates, document a versioned publication schema, and create parity tests while preserving the generator's CLI and imported test functions. Out of scope: redesigning v2.1 rules, moving frozen runtime code under `v2.1/`, changing historical research evidence, inventing a server API, replacing Python analytics, or adding a full frontend framework in this milestone.

## Progress

- [ ] Establish reproducible baseline outputs with fixed fixtures, normalize only nondeterministic timestamps/build IDs, and capture decision/history semantics.
- [ ] Inventory dependencies of `scripts/build_dashboard.py`, `test_dashboard.py`, generated `site/` and `scripts/build_v3_challenger.py`.
- [ ] Extract presentation templates and assets behind stable rendering functions in incremental behavior-preserving commits.
- [ ] Document publication JSON schemas, provenance, and no-leakage boundary; add compatibility/fixture tests.
- [ ] Verify production JSON/decision and v3 research-only separation; run broader tests and a Pages build smoke check.
- [ ] Independent diff review, checkpoint `.agent/STATUS.md`, and handoff.

## Surprises & Discoveries

- None yet.

## Decision Log

- 2026-10-09, starter-kit author: prefer a parity-proven incremental extraction over a simultaneous React rewrite.

## Context and Orientation

- `scripts/build_dashboard.py` generates HTML, CSS, JavaScript, CSV, and JSON with Plotly/Jinja2; its imported public parsing/merge functions must remain compatible.
- `.github/workflows/deploy-pages.yml` runs root tests, builds the dashboard, then adds v3 shadow output. Do not disrupt invocation paths without an atomic, approved deployment migration.
- Current dashboard output includes `site/analysis.json`, `site/index.html`, `site/app.js`, `site/styles.css`, and `site/version.json`.

## Plan of Work

Create fixture-driven semantic snapshots. Extract HTML templates and CSS/JavaScript into source-controlled assets outside the generated `site/` output directory. Wire the existing Python renderer to load these source assets while preserving escaping and generated values. Separate pure computations from write-only publication without changing decision functions. Assert equivalent values for core production JSON, event histories, signal decisions, and the untouched v3 append step. Make each extraction independently testable and revertible.

## Concrete Steps

From root:

1. Run `python -m unittest -v test_dashboard test_feargreed test_fear_greed_market_data test_v3_challenger` and `python scripts/build_dashboard.py` before edits; save local baseline artifacts outside committed history.
2. Inventory module imports, template interpolations, generated schema, and asset paths; choose stable source layout only after inventory.
3. Extract one presentation layer at a time, add compatibility tests, and rerun targeted tests and build after each extraction.
4. Run `python -m unittest discover -s v3/tests -p 'test_*.py'` for boundary regressions; use v3-specific build stages only as provided by existing approved workflow.
5. Review `git diff`, parity report, and output manifest; checkpoint findings and commits.

## Validation and Acceptance

- NOT RUN: targeted tests; expect zero regressions.
- NOT RUN: v3 tests; expect no accidental research change.
- NOT RUN: comparison of representative decisions/historical replay JSON; expect semantic parity, using documented nondeterministic-field normalization only.
- NOT RUN: generated asset paths and static Pages build check; expect no missing files or broken browser console requests.
- NOT RUN: production-v3 separation contract; `analysis.json` must be unchanged by research-only renderer.

## Risks and Stop Conditions

Any changed decision rule, altered data provenance, broken immutable evidence, build/deploy path migration, or unapproved dependency architecture is out of scope; stop and get owner approval rather than broadening this plan.

## Outcomes & Retrospective

- Not implemented.
