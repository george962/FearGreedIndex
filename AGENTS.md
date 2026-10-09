# FearGreedIndex — shared coding-agent instructions

This file applies to Codex, Claude Code (via `CLAUDE.md`), and other repository-aware coding agents working in this repository. Read it before editing. Do not infer permissions to perform external actions from this file.

## Mission

Turn FearGreedIndex into an approachable, dependable, publicly shareable **market-sentiment research application**. The primary users are curious retail investors and technically inclined researchers who want to understand current sentiment, historical analogs, return distributions, limitations, and methodological uncertainty. This is not an execution or personalized-investment-advice service.

## Read before acting

1. `AGENTS.md`, `PRODUCT.md`, `DESIGN.md`.
2. `.agent/ROADMAP.md`, `.agent/STATUS.md`, `.agent/PLANS.md`.
3. The applicable `.agent/execplans/FGI-*.md` file in full.
4. The files targeted by that ExecPlan and their associated tests/workflows.
5. When touching research boundaries: `v2.1/README.md`, `v3/README.md`, `v3/STATUS.md`, and `v3/PLAN.md`.

Do not use old chat history as project state. Read the checked-in state and current code. If the documents differ from code or current tests, record the inconsistency instead of silently choosing an interpretation.

## Non-negotiable research boundaries

- v2.1 is the frozen operational/research benchmark. Root files including `FearGreed.py`, `FearGreedHistory.py`, `FearGreedMarketData.py`, `backtest.py`, `config.json`, `strategy_manifest.json`, and the dashboard implementation currently constitute its operational runtime. Do **not** retune signals, thresholds, feature definitions, exposure sizing, or time conventions as part of product work.
- Do not move v2.1 root runtime files piecemeal. A migration must be atomic, separately scoped, regression-tested, and explicitly approved.
- v3 remains **research-only**. `v3/STATUS.md` is authoritative for its changing status. No production champion currently passes the acceptance gate. Do not change production actions, claim confirmed live edge, enable tactical sizing, or unseal forward labels/evidence. Display v3 outputs only as clearly labeled shadow/research evidence.
- Treat append-only forward ledgers, immutable experiment reports, source snapshots, manifests, and checkpoint histories as protected artifacts. Never rewrite, backfill, or regenerate past recorded predictions for convenience.
- Product UI and generated JSON/CSV contracts may be improved without changing market logic. Preserve point-in-time and next-session-entry semantics.
- Never publish credentials, personal positions/balances, or restricted datasets; confirm third-party data rights and attribution before public redistribution.

## Source and architecture boundaries

- Current dashboard generation: `scripts/build_dashboard.py`; research-only overlay: `scripts/build_v3_challenger.py`.
- Existing scheduled ingestion and Pages deployment live under `.github/workflows/`.
- Existing output contracts include `site/analysis.json`, `site/v3_challenger.json`, `site/index.html`, `site/styles.css`, `site/app.js`, and `site/version.json` after a build.
- Favor a static frontend over a server/database until a specific user-facing feature justifies more infrastructure. Keep analytics and model evaluation in Python. Any frontend rewrite must read stable, versioned, validated publication artifacts and remain separate from research computations.

## How to work on substantial changes

Follow `.agent/WORKFLOW.md` for the project-owned agent delivery lifecycle and approval gates.

1. Identify the next eligible, unblocked milestone from `.agent/STATUS.md` and `.agent/ROADMAP.md`; don't restart completed work when asked to continue.
2. For multi-file features, refactors, workflow changes, or migrations, read `.agent/PLANS.md` and maintain a self-contained ExecPlan under `.agent/execplans/` **before** implementation. Keep its progress, decisions, surprises, and evidence current.
3. Check `git status --short` and current branch. Preserve uncommitted user work. Use a new feature branch when one has not been provided. Don't force-push or reset others' changes.
4. Make the smallest change that satisfies the defined acceptance criteria. Use incremental, comprehensible commits. Avoid unrelated dependency upgrades, aesthetic changes, or opportunistic research retuning.
5. After every material step, update the ExecPlan's `Progress`, `Decision Log`, and `Validation and Acceptance` evidence. Update `.agent/STATUS.md` with milestone state, commit hashes when available, tests run, blockers, and next action. Commit these updates alongside the work.
6. Validate before claiming completion. For a targeted change, run targeted tests; for meaningful pipeline changes, run all relevant regression tests and build the static dashboard. Record exact commands, exit codes, and unrun checks. Don't treat a generated file's presence as proof of semantic parity.
7. Review the diff for data leakage, sensitive-data exposure, accidental research behavior changes, user-visible regression, license problems, reproducibility, CI workflow risks, and maintainability. Fix findings or mark as blocked; do not report success while blocking failures remain.
8. Summarize changed paths, user-visible result, test evidence, commits, known risks, status, and next task. Prepare a pull request only if asked or explicitly approved. **Never merge, deploy, tag a release, change secret settings, or publish a dataset without explicit owner approval.**

### Representative validation commands (from repository root)

- `python -m unittest -v test_feargreed test_fear_greed_market_data test_dashboard test_signal_ledger test_strategy_validation test_v3_challenger`
- `python -m unittest discover -s v3/tests -p 'test_*.py'`
- `python scripts/build_dashboard.py`
- `python scripts/build_v3_challenger.py` only when the required snapshot inputs have been built and verified according to the existing workflow; do not use it to re-form historical evidence.
- For frontend changes, add a build check and browser smoke tests to the ExecPlan once the frontend toolchain exists. Do not invent a `npm` command before a package manifest exists.

These are starting commands, not a promise that any have been executed. Respect local environment dependencies and document failures.

## Stop and ask the owner

Stop rather than guessing if work would (a) alter the frozen v2.1 methodology, (b) unseal or mutate forward evidence, (c) promote/activate v3, (d) change funding/execution/positions, (e) incur recurring cloud expenses, (f) create a public data-licensing exposure, (g) modify credentials or repository permissions, (h) merge or publish, or (i) require destructive data migration. All other routine implementation decisions may be made within the approved ExecPlan and documented.

## Agent orientation

- Codex and other agents should follow this `AGENTS.md` and `.agent/PLANS.md`.
- Claude Code should read `CLAUDE.md`, which delegates to this file.
- Use `.agent/WORKFLOW.md` for the project-owned planning, implementation, review, verification, and checkpoint process. This is a set of repository rules, not a dependency on any external agent runner.
- These files guide active coding sessions; they do not schedule independent work or guarantee platform-specific agent support.

## Repository hygiene

Before deleting, moving, or archiving files, follow `docs/REPOSITORY_CLEANUP.md`. Existing tests, workflow entrypoints, ledgers, baseline reports, and research checkpoints are protected until dependencies and preservation have been verified. Never delete tests to make CI pass.

## Agent roles

For complex tasks apply `.agent/roles/planner.md`, `.agent/roles/implementer.md`, `.agent/roles/reviewer.md`, and `.agent/roles/verifier.md` in sequence. A separate agent/subagent is optional; if unavailable, perform separate review passes with fresh context.
