# ExecPlan specification for FearGreedIndex

An ExecPlan is a self-contained, maintained design-to-implementation contract for a single deliverable. It tells an agent what to change, why, how to test, how to recover, and how to know when it is finished. Do not rely on chat context.

## When required

Use an ExecPlan for any multi-file feature, architecture change, data contract change, CI/workflow change, dependency introduction, migration, or work expected to exceed one hour. Tiny docs/typo fixes can omit one if they don't change behavior.

## Creation workflow

1. Read `AGENTS.md`, `PRODUCT.md`, `DESIGN.md`, `.agent/ROADMAP.md`, `.agent/STATUS.md`, relevant code/tests, and the protected v2.1/v3 docs as appropriate.
2. Confirm actual repository state before promising changes. Name exact files and dependencies; mark unknowns as preflight checks.
3. Write the ExecPlan **before coding**, using the section template below. Define user-observable success and explicit non-goals.
4. Implement milestone by milestone. Each milestone needs its own completed test/evidence and a durable status update; don't simply check off an untested step.
5. Keep `Progress`, `Surprises & Discoveries`, `Decision Log`, `Validation and Acceptance`, and `Outcomes & Retrospective` current throughout. On every interruption, write the exact next runnable action.
6. When the plan completes, update `.agent/STATUS.md` and preserve the plan as a record; don't delete it to conceal failures.

## Required sections and semantics

- `Purpose / Big Picture`: what a user can now do that was impossible before, and how to observe it.
- `Scope and Non-goals`: allowed modifications, exclusions, upstream/downstream boundaries.
- `Progress`: timestamped `[ ]`, `[x]`, or partial checkboxes, each tied to an observable deliverable.
- `Surprises & Discoveries`: unexpected findings, evidence, and impact.
- `Decision Log`: decision, alternatives, rationale, date, author.
- `Context and Orientation`: repository paths, business and technical vocabulary, current state, constraints.
- `Plan of Work`: an ordered narrative of file changes and integration points, not hand-wavy phases.
- `Concrete Steps`: commands from repository root, expected observable output, migration/recovery procedures.
- `Validation and Acceptance`: exact tests and manual checks, before/after parity checks where appropriate, failure/rollback behavior, and explicit PASS/FAIL/NOT RUN results.
- `Risks and Stop Conditions`: how to handle ambiguous behavior, data rights, security, deploy and research boundaries.
- `Outcomes & Retrospective`: final behavior, changed files, commit SHA(s), evidence, unresolved issues, follow-up.

## Validation rules

- A documented checklist is not proof of passing tests. Record actual output and failures; do not fabricate SHA, CI state, Pages URL, data licenses, or tests.
- Validate data contracts and reproducibility when touching publication artifacts; don't redefine research semantics in product refactors.
- Code review must be separate from initial implementation; fresh-context review is preferable, but does not replace automated assertions.
- Keep non-reversible operations behind human approval. A completed plan does not authorize merging or publishing.

## Minimal reusable skeleton

# FGI-XXX — Verb-focused title

This is a living ExecPlan governed by `.agent/PLANS.md` and `AGENTS.md`.

## Purpose / Big Picture

## Scope and Non-goals

## Progress
- [ ] (not started) Preflight inventory and baseline evidence.
- [ ] Implement and verify milestone 1.
- [ ] Implement and verify milestone 2.
- [ ] Independent review, status update, and handoff.

## Surprises & Discoveries
- None observed yet.

## Decision Log
- No implementation decisions yet.

## Context and Orientation

## Plan of Work

## Concrete Steps

## Validation and Acceptance
- NOT RUN: [command]; expected result [description].

## Risks and Stop Conditions

## Outcomes & Retrospective
- Not completed.
