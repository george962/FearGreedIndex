# FearGreedIndex — agent kit setup

This starter kit was prepared against the public repository shape observed on 2026-10-09. It has **not** been committed to GitHub, and no implementation tests have been run. Review and adapt before use.

## Install into a local checkout

1. Start from a clean feature branch of `george962/FearGreedIndex`.
2. Unzip this kit **at repository root** so `AGENTS.md`, `CLAUDE.md`, `PRODUCT.md`, `DESIGN.md`, `.agent/`, and `.atomic/` appear beside `README.md`. Avoid blindly overwriting an existing file if one was introduced since this kit was generated.
3. Read `.agent/ROADMAP.md`, `.agent/STATUS.md` and `FGI-001`.
4. Commit the starter-kit files on the feature branch (do not merge into main until reviewed).

## Use from Codex or Claude Code

Open the checkout in a coding agent that reads repository files. Ask once:

> Read `AGENTS.md`, `.agent/PLANS.md`, `.agent/ROADMAP.md`, and `.agent/STATUS.md`. Work on the next unblocked ExecPlan, starting with FGI-001. Update the living plan and status after every checkpoint; implement, test and independently review. Stop on the owner-approval boundaries in AGENTS.md. Do not merge or deploy.

On later sessions, ask: `Resume the next unfinished step from .agent/STATUS.md under AGENTS.md and the active ExecPlan.`

Ordinary ChatGPT conversation does not automatically execute local commands just because these files exist. Use a coding agent/environment with file and terminal access, or Work/connected development tools appropriate for the task.

## Use actual Atomic

1. Install Atomic separately following the current docs at `https://github.com/bastani-inc/atomic`. npm-based install at time of drafting: `npm install -g @bastani/atomic` (requires supported Node.js). Run `atomic` **from the FearGreedIndex repository root**, then `/login` and choose your model/provider.
2. Prefer the built-in `/workflow goal` for the most robust loop and reviewer-gated tracking. Run `/workflow inputs goal` to inspect its current input schema. Supply an objective instructing it to implement exactly the FGI-001 ExecPlan and the acceptance criteria from that file, with PR creation disabled by default. Check status and evidence before accepting.
3. For the included custom workflow: run `/workflow reload`, `/workflow list`, `/workflow inputs feargreed-delivery`, then `/workflow feargreed-delivery plan_path=".agent/execplans/FGI-001-public-foundation.md"`. Confirm preflight and handoff gates manually.
4. This custom workflow is a **starter**, not a machine-verifiable approval reducer. Its reviews are prompted tasks and its final gate is a human confirmation. If you need fail-closed automated reviewer thresholds and bounded repair, build on Atomic's tested built-in `goal` or `adversarial-verification` primitives rather than interpreting free-form review text as a boolean.

## Security

Atomic documents that it does **not** provide a built-in shell sandbox/permission gate. Run autonomous work on a devcontainer/VM or dedicated checkout with least-privilege credentials. Keep brokerage or other personal credentials and restricted datasets out of that environment. Approval inside a workflow is not a substitute for OS/GitHub permissions.

## Next milestone

FGI-001 is the next queued task. It begins with read-only public-site, data-license, and baseline checks. Public search indexing and deployment remain owner decisions, not defaults.
