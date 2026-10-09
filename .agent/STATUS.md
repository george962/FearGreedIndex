# FearGreedIndex productization — durable engineering status

Last initial scaffold preparation: 2026-10-09. **No project code, tests, runtime workflows, GitHub PRs, or deployments were changed or executed when this starter kit was drafted.** These are proposed instructions to add to the repository and validate on a feature branch.

## Milestones

| ID | Status | Evidence and commits | Next action |
| --- | --- | --- | --- |
| FGI-001 | NOT STARTED | None | Read plan; inspect current site and data rights; establish preflight baseline |
| FGI-002 | NOT STARTED | None | Wait for FGI-001 acceptance |
| FGI-003 | NOT STARTED | None | Wait for FGI-002 acceptance |
| FGI-004 | NOT STARTED | None | Wait for relevant preceding milestones |

## Current task

FGI-001: `.agent/execplans/FGI-001-public-foundation.md`.

## Research-boundary snapshot (not a live status guarantee)

Based on repository documentation reviewed 2026-10-09: v2.1 is frozen; v3 has no promotable champion, and its forward evidence remains sealed. On each new session re-read `v3/STATUS.md` rather than treating this summary as authoritative.

## Active blockers / owner decisions

- Confirm desired public URL/hosting and branding.
- Review source licenses and public redistribution permissions for all exposed market data.
- Confirm when discovery (`noindex`) should be enabled for public search engines.

## Agent checkpoint protocol

Every implementation stop must update this file with:
- UTC update timestamp and branch.
- Exact milestone and checklist step completed/partial/blocked.
- Changed files and implementation commit SHA (if committed).
- Test commands/results and links to CI if actually run.
- Status/checkpoint commit SHA where available.
- Risks/blockers and the **single exact next runnable step**.

If a tool can't write to this file, say so in the handoff and don't claim durable checkpointing.
