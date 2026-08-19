# FearGreedIndex v3 Status

`PLAN.md` remains the authoritative roadmap. This file records execution status so the roadmap does not have to be inferred from branches or old chat history.

## Completed

| Task | Status | Evidence |
| --- | --- | --- |
| V3-001 Freeze v2.1 baseline | COMPLETE | PR #6; reproducibility contract hardened in PR #24; frozen package in `reports/baseline_v2_1/` |
| V3-002 Point-in-time feature dataset | COMPLETE | PR #6, `v3-features-001`, 41 features |
| V3-003 Leakage/data-quality validation | COMPLETE | PR #6, CI leakage gate |
| V3-004 Multi-horizon executable-entry labels | COMPLETE | PR #6, `v3-labels-001` |
| V3-005 Logistic classification baseline | COMPLETE | PR #12, `EXP-001` |
| V3-006 Regularized return regression | COMPLETE | PR #14, `EXP-002` |
| V3-007 Gradient-boosting candidate | COMPLETE | PR #16, `EXP-003` |
| V3-008 Random-forest benchmark | COMPLETE | PR #18, `EXP-004` |
| V3-009 Common walk-forward evaluator | COMPLETE | PR #20, `v3-evaluator-001` |
| V3-010 Model tournament scoreboard | COMPLETE | PR #24, `v3-tournament-001`; no promotion-ready model |

## Active

- **V3-011 — Add VIX/volatility features** is in progress.
- The baseline `v3-features-001` remains frozen; VIX is being added as separate `v3-features-002-vix` for a clean ablation.

## Next

1. Finish V3-011 — first-party VIX snapshot, point-in-time VIX features, leakage/data-quality gate.
2. V3-012 — controlled baseline-vs-VIX ablation with identical models, labels, folds, seeds, and evaluation dates.
3. V3-013 — QQQ/SPY relative-strength family only after the VIX ablation result is recorded.

## Current evidence summary

- All initial models use the same `v3-features-001` and `v3-labels-001` point-in-time contracts.
- All initial experiments use the same 2024, 2025, and 2026 YTD chronological folds.
- `v3-evaluator-001` verified identical realized-date hashes within each comparable model lane.
- `v3-tournament-001` ranks random forest (`EXP-004`) first among full candidates, but rank is separate from promotion readiness.
- **No champion has been selected.** No experiment currently passes the absolute promotion gates.
- Random forest still has negative mean relative Brier improvement versus the fold base-rate benchmark and negative mean return Spearman correlation; its drawdown signal is not robust across enough folds.
- Trading/action evaluation remains intentionally separate until the decision-policy stage.
- The v2.1 frozen baseline now uses an immutable `2026-08-18` cutoff and canonical parsed-input fingerprints, so later live data appends cannot silently change frozen outcomes.
