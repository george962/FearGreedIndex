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
| V3-011 VIX/volatility feature family | COMPLETE | PR #27, `v3-features-002-vix`; first-party Cboe snapshot, 8 VIX features, zero future joins/missing VIX features in the v3 sample |
| V3-012 Controlled VIX ablation | COMPLETE | PR #28, `v3-vix-ablation-001`; frozen as-of 2026-08-18; VIX rejected by pre-registered retention gate |

## Active

- No V3-011/V3-012 work remains.
- **V3-013 — Add QQQ/SPY relative-strength features** is the next implementation task.

## Next

1. V3-013 — add a separately versioned QQQ/SPY relative-strength feature family with point-in-time provenance and leakage tests.
2. Evaluate the new family with the same one-family-at-a-time discipline used for VIX before allowing it into later research.
3. V3-014 — market breadth only after the V3-013 evidence is recorded.

## Current evidence summary

- All initial models use the same `v3-features-001` and `v3-labels-001` point-in-time contracts.
- All initial experiments use the same 2024, 2025, and 2026 YTD chronological folds.
- `v3-evaluator-001` verifies identical realized-date hashes within each comparable model lane.
- `v3-tournament-001` ranks random forest (`EXP-004`) first among the original full candidates, but rank remains separate from promotion readiness.
- **No champion has been selected.** No experiment currently passes the absolute promotion gates.
- V3-011 added 8 VIX features as a separate `v3-features-002-vix` lane and preserved every baseline feature value exactly.
- V3-012 was frozen as-of `2026-08-18`; later-maturing outcomes are censored from both comparison lanes.
- V3-012 compared 60 identical fold/target cells and verified matching realized-date hashes between baseline and +VIX.
- VIX did **not** pass the pre-registered retention rule: `0/3` prediction lanes were robust in both `EXP-003` and `EXP-004`.
- In the ablation tournament, baseline random forest (`BASE-EXP-004`) remained the best-ranked full candidate; all +VIX full candidates ranked worse and none passed absolute promotion gates.
- The VIX feature family is therefore not retained for the main research feature set. Its code/data/evidence remain available as a documented negative experiment.
- Trading/action evaluation remains intentionally separate until the decision-policy stage.
- The v2.1 frozen baseline uses an immutable `2026-08-18` cutoff and canonical parsed-input fingerprints, so later live data appends cannot silently change frozen outcomes.
