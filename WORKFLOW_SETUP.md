# Agent-driven development: no additional runtime required

This package contains **only Markdown instructions and plans**. No Atomic installation, plugins, or workflow `.ts` files are needed.

## Installation

Create a feature branch in `george962/FearGreedIndex`. Copy this archive's folders/files into the **repository root**, preserving hidden `.agent/`. Inspect the additions and commit. Avoid overwriting files that appeared since the inventory.

## What the product does

The public-facing product is a read-only market sentiment research application. It shows current source-dated Fear & Greed conditions, market context, historical analog studies, returns/drawdowns, backtests and clearly labeled v3 research evidence. It is not a brokerage, automated trader or personalized financial advice service. The initial public delivery remains a static GitHub Pages site with Python-generated evidence and gradually improved interaction.

## Work session

Open the project in Claude Code or Codex with filesystem and terminal access. Tell the agent:

> Read AGENTS.md, .agent/STATUS.md and the active ExecPlan. Work on the next approved unblocked task. Follow the planner, implementer, reviewer, verifier roles, update durable checkpoints and commit your work on a feature branch. Stop at owner approval boundaries; do not merge or deploy.

Future sessions: `Resume from .agent/STATUS.md; verify current branch/worktree and complete the next unblocked ExecPlan step.`

These files provide durable instructions, not a background scheduler. An active coding session must run them. If the agent has no repository write/terminal permissions it can only advise.

## Build order

- FGI-001: public onboarding, site visibility, evidence freshness, safe release foundation.
- FGI-002: split template/presentation concerns from the monolithic Python dashboard builder while preserving decisions.
- FGI-003: historical filtering and research exploration.
- FGI-004: stronger test/release/contributor documentation.

## Cleanup

Read `docs/REPOSITORY_CLEANUP.md` before deleting anything. No tests should be removed just for tidiness.

## Protect the live system

Frozen v2.1 signals and v3 research evidence are protected. Do not unseal forward evidence, promote an unapproved challenger, change execution sizing, rewrite historical ledgers, or auto-deploy.
