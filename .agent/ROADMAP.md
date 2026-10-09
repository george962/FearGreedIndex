# FearGreedIndex product engineering roadmap

This roadmap concerns **productization**, not predictive research. Research status belongs to `v3/STATUS.md`. Implement only one bounded ExecPlan at a time unless the owner explicitly changes the scope. All four plans begin as NOT STARTED in this starter kit; they are proposals, not records of completed work.

| ID | Title | Depends on | Product acceptance | State |
| --- | --- | --- | --- | --- |
| FGI-001 | Public-ready foundation | Existing dashboard | First-time user onboarding, demo discovery, safe public metadata, source freshness disclosure, reproducible local checks | NOT STARTED |
| FGI-002 | Decouple dashboard generator | FGI-001 | Templates/assets separated from analysis with preserved semantic JSON and historical outputs | NOT STARTED |
| FGI-003 | Interactive market Explorer | FGI-002 | Filterable history and comparison from versioned, read-only published data; mobile/keyboard UX | NOT STARTED |
| FGI-004 | Reliable release and contribution workflow | FGI-001..003 | CI/browser smoke tests, deployment/rollback procedures, dependency/security/documentation baseline | NOT STARTED |

## Constraints across all milestones

- No changes to frozen v2.1 decision methodology or research conclusions.
- v3 is visible as **shadow/research-only**, never approved automatically.
- Do not modify `v3/evidence/` or `data/signal_ledger.csv` for product work.
- Don't silently turn `noindex` off until the public visibility and data-rights decision is documented.
- No database/server/account system until clearly justified by product use cases.

## Promotion from roadmap to work

1. Pick the first NOT STARTED or PARTIAL unblocked milestone.
2. Open its `.agent/execplans/FGI-*.md` and complete preflight and scope evidence.
3. Execute its verifiable steps. Update its progress and `.agent/STATUS.md` after each milestone.
4. Obtain owner approval for PR or public deployment. Never infer approval from a passing test.
5. Mark COMPLETED only when evidence supports all acceptance criteria; otherwise BLOCKED or PARTIAL.
