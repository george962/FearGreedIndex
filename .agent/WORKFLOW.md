# FearGreedIndex engineering workflow

**Owner:** FearGreedIndex. **Execution tools:** any coding agent capable of reading and editing the repository, running commands, and writing Git commits. This is the project's own workflow; no third-party orchestration runtime is required.

## Source of truth

1. `AGENTS.md`: permanent guardrails and permission boundaries.
2. `PRODUCT.md` and `DESIGN.md`: user experience and product intent.
3. `.agent/ROADMAP.md`: ordered product milestones.
4. `.agent/STATUS.md`: one current task, durable checkpoint and blockers.
5. `.agent/PLANS.md` and `.agent/execplans/FGI-*.md`: how to plan and perform a bounded change.
6. Implementation code and actual tests always outrank stale descriptions; update documentation when discrepancies are discovered.

This workflow governs the public-facing product. `v3/STATUS.md` is a **separate** research system checkpoint; it must never be silently combined with or overwritten by product work.

## Execution stages and agents

| Stage | Agent role | Mandatory output | Gate |
| --- | --- | --- | --- |
| A. Orient | Planner | Current branch, clean/dirty state, active milestone, existing contracts, dependencies, approvals | Stop if scope or permissions conflict |
| B. Baseline | Verifier | Test/build results, outputs and comparison criteria before changing code | Note unrun checks explicitly |
| C. Implement | Implementer | Minimal code/documentation changes, targeted tests, small commits | Frozen research/data boundaries remain intact |
| D. Independently review | Reviewer (separate review pass or context if available) | Specific findings ranked by severity: bugs, regressions, data provenance, accessibility, security, maintainability | Blocking findings must be fixed or task marked BLOCKED |
| E. Verify | Verifier | Exact commands and exit codes, before/after invariants, any browser and licensing checks | No unsupported PASS claims |
| F. Checkpoint | Planner or implementer | Updated ExecPlan, `.agent/STATUS.md`, branch, implementation SHA, checkpoint SHA when available, exact next step | Git commit; no merge/deploy without owner approval |

The roles are **responsibilities**, not separate required AI services. One coding agent can perform them serially, but it must perform a fresh review pass instead of assuming its implementation is correct. A second reviewer is preferred for high-risk changes.

## Session boot procedure

1. `git status --short`, `git branch --show-current`, and `git log -1 --oneline`; preserve any unrelated work.
2. Read the sources listed above, including the entire active ExecPlan and relevant research contract documentation.
3. Choose the **one** earliest unblocked, approved task. Do not invent new priorities during a continuation run.
4. Record intended files, non-goals, validation commands, stop conditions, and rollback strategy in the ExecPlan before editing.
5. For each material change: implement, run targeted checks, review, fix, update plan and checkpoint, and commit.
6. Prefer evidence over assurances. Never change research outputs to make an assertion pass.
7. End with status, changed paths, test evidence, commit(s), unrun checks, blockers, and one next executable task.

## Resumption and failure handling

- When asked to "continue," resume the next unchecked item in `.agent/STATUS.md`; inspect Git history first because the checkpoint may lag a committed implementation.
- If interrupted, write a `PARTIAL` checkpoint stating the exact command or file to inspect next.
- If a test fails, reproduce and fix its cause; do not weaken, delete, skip, or blindly regenerate the test.
- If blocked by licensing, Pages access, approvals, or missing environment capabilities, do the unrelated safe work and mark the blocked acceptance item `NOT VERIFIED`.
- Prefer a branch such as `feat/fgi-001-public-foundation`. Never push directly to `main`.

## Owner approval conditions

Obtain explicit owner approval **before** merging, deploying, enabling search indexing, changing Pages/repository settings or secrets, introducing ongoing costs, exposing new source datasets, altering v2.1 decisions, promoting v3, changing position/funding execution, or rewriting an immutable ledger. Tests passing does not waive approval.

## Release checklist

An FGI milestone can only be marked `COMPLETE` when its user-facing acceptance, relevant tests, reproducibility, review, and any legally required source/provenance checks are evidenced. Otherwise mark `PARTIAL` or `BLOCKED` and document next steps. A completed development branch is not the same thing as a released site.

## One-line instructions for a coding agent

> Read `AGENTS.md`, `.agent/WORKFLOW.md`, `.agent/STATUS.md`, and the active ExecPlan. Continue its next approved unblocked task using the planner → implementer → reviewer → verifier → checkpoint stages. Run and record real tests, commit to a feature branch, and stop at owner approval gates. Do not merge or deploy.
