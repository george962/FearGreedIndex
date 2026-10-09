# FearGreedIndex productization — durable engineering status

**Package checkpoint:** 2026-10-09 UTC. **Prepared offline for user upload; NOT committed, pushed, merged, or deployed by this package.** Exact branch and Git SHAs must be filled in after the user applies it to their checkout. No repository credentials or settings are needed to use these documents.

## Milestones

| ID | Status | Evidence | Next action |
| --- | --- | --- | --- |
| FGI-001 | PARTIAL — local delivery package | Read-only source/deployment/research inspection; template modification helper syntax-checked and applied to a synthetic fixture; live tests NOT RUN | Apply package on a feature branch; baseline and regression tests; review source rights |
| FGI-002 | NOT STARTED | Depends on FGI-001 acceptance | Do not start yet |
| FGI-003 | NOT STARTED | Depends on FGI-002 | Do not start yet |
| FGI-004 | NOT STARTED | Depends on earlier milestones | Do not start yet |

## Current task

FGI-001: `.agent/execplans/FGI-001-public-foundation.md`.

## Latest implementation handoff

- Included files: project-owned `.agent/WORKFLOW.md`; revised instructions; revised README; `docs/USER_GUIDE.md`; `docs/PUBLICATION_PRECHECK.md`; `test_public_foundation.py`; a **local one-time helper** `extras/apply_fgi001.py` that changes only the dashboard presentation in `scripts/build_dashboard.py` when run.
- Validation performed: helper and new test module compiled with Python; dry-run and application against a synthetic source fixture PASS; `python -m unittest -v test_public_foundation` PASS (6 tests) **on that synthetic fixture only**; AST syntax and preservation of `noindex,nofollow` verified in that fixture.
- Validation **not performed**: complete local repository's root and v3 tests, a real `python scripts/build_dashboard.py` build, generated semantic decision parity, live Pages visit, mobile/keyboard browser validation, third-party data licensing review, CI checks. These results MUST remain unverified until actually executed.
- Implementation commit SHA: **NONE (package only)**. Checkpoint commit SHA: **NONE (package only)**. Development branch: **user choice**.
- Research protections: v2.1 is frozen, v3 shadow is research-only, no evidence ledger was modified. Recheck `v3/STATUS.md` each session.

## Blocking owner decisions

1. Verify the canonical Pages URL/desired public branding and whether it is accessible to visitors. Do not modify repository Pages settings without approval.
2. Confirm licensing/attribution and public redistribution rights for sentiment and SPX/SPY inputs and exported derivative datasets.
3. Decide when search indexing is allowed; both templates deliberately keep `noindex,nofollow`.
4. Explicitly approve deployment/merge only after successful local/CI checks and acceptance review.

## Single next executable step

On the user's local checkout: create or switch to a clean FGI-001 feature branch, extract this bundle, run `python extras/apply_fgi001.py --check`, then apply the helper and run the precise test/build sequence in `FGI-001-public-foundation.md`. Record actual baseline/evidence before labeling any item COMPLETE.

## Checkpoint requirements for all future work

Record UTC timestamp, branch, task state, changed files, implementation SHA, checkpoint SHA when available, commands/exit codes, blockers and one next runnable step. If files are not committed, do not claim durability or a commit hash.
