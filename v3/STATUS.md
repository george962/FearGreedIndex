# FearGreedIndex v3 Status

`PLAN.md` remains the authoritative roadmap. This file records execution status so the roadmap does not have to be inferred from branches or old chat history.

## Completed

| Task | Status | Evidence |
| --- | --- | --- |
| V3-001 Freeze v2.1 baseline | COMPLETE | PR #6, frozen package in `reports/baseline_v2_1/` |
| V3-002 Point-in-time feature dataset | COMPLETE | PR #6, `v3-features-001`, 41 features |
| V3-003 Leakage/data-quality validation | COMPLETE | PR #6, CI leakage gate |
| V3-004 Multi-horizon executable-entry labels | COMPLETE | PR #6, `v3-labels-001` |
| V3-005 Logistic classification baseline | COMPLETE | PR #12, `EXP-001` |
| V3-006 Regularized return regression | COMPLETE | PR #14, `EXP-002` |
| V3-007 Gradient-boosting candidate | COMPLETE | PR #16, `EXP-003` |
| V3-008 Random-forest benchmark | COMPLETE | PR #18, `EXP-004` |

## Active

- **V3-009 — Common walk-forward evaluator** (issue #19)

## Next

1. V3-010 — model tournament scoreboard.
2. V3-011/V3-012 — add VIX and run a controlled ablation.
3. Continue feature families only after the common evaluator and tournament are fixed.

## Current evidence summary

- All initial models use the same `v3-features-001` and `v3-labels-001` point-in-time contracts.
- All initial experiments use the same 2024, 2025, and 2026 YTD chronological folds.
- Random forest currently looks strongest in aggregate predictive metrics, but **no champion has been selected**. Formal ranking belongs to V3-010.
- Trading/action evaluation remains intentionally separate until the decision-policy stage; V3-009 provides generic backtest metrics without inventing a policy early.
