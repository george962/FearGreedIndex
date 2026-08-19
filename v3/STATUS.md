# FearGreedIndex v3 Status

`PLAN.md` remains the authoritative roadmap. This file records execution status so the roadmap does not have to be inferred from branches, stale PRs, or old chat history.

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
| V3-011 VIX/volatility feature family | COMPLETE | PR #27, `v3-features-002-vix`, 8 VIX features |
| V3-012 Controlled VIX ablation | COMPLETE — REJECT | PR #28; frozen as-of 2026-08-18; `0/3` robust lanes in both full models |
| V3-013 QQQ/SPY relative strength | COMPLETE — REJECT | repaired in PR #41; manifest-matching frozen rerun; `1/3` robust lanes |
| V3-015A Treasury rates/yield curve | COMPLETE — KEEP | repaired/current-code rerun in PR #41; `2/3` robust lanes; `UST-EXP-004` best-ranked full candidate in this ablation |
| V3-015B Broad U.S. dollar | COMPLETE — REJECT | repaired/current-code rerun in PR #41; `1/3` robust lanes |
| V3-015C 76-feature combined stack | COMPLETE — REJECT | repaired/current-code rerun in PR #41; `1/3` robust lanes |
| V3-016 Prediction-to-action policy | COMPLETE | `v3-decision-policy-001`; research-only, deterministic, no sizing/sell semantics |
| V3-017 Minimal sizing layer | IMPLEMENTED / ACTIVATION BLOCKED | `v3-sizing-policy-001`; current multiplier remains `1.00x`; `1.10x` requires promotion-ready prediction |
| V3-018 Champion acceptance gates | COMPLETE — CURRENT CANDIDATE NOT PROMOTION READY | PR #43, `v3-champion-gates-001`; fail-closed; `UST-EXP-004` fails prediction prerequisite; V3-019 ineligible |
| EXP-005 Chronologically calibrated ExtraTrees | COMPLETE — REJECT | PR #45; calibration improved materially but absolute prediction gate still fails; no post-result tuning |

## Blocked / deferred

- **V3-014 market breadth:** `DATA_SOURCE_BLOCKED`. Do not backfill today's constituent universe into history. Issue #31 records point-in-time source requirements.
- **V3-015 credit spreads:** source/license gated. Do not redistribute or silently depend on restricted ICE/Moody's history without a compliant source.
- **V3-019 champion manifest / production promotion:** blocked because zero candidates pass V3-018.
- **V3-017 1.10x empirical sizing:** blocked because zero candidates pass the prediction/champion prerequisite.

## Current evidence summary

- PR #41 repaired historical V3 repository state and reran V3-013, V3-015A/B/C, V3-016, and V3-017 under current code using the frozen `2026-08-18` research cutoff.
- The repair rerun passed the full V3 test suite, matched realized-date samples, reproduced the frozen v2.1 benchmark, and removed leaked temporary write-enabled finalizers.
- Corrected feature-family decisions are:
  - VIX: **REJECT** (`0/3` robust lanes).
  - QQQ/SPY relative strength: **REJECT** (`1/3`).
  - Treasury: **KEEP** (`2/3`).
  - Broad dollar: **REJECT** (`1/3`).
  - 76-feature combined stack: **REJECT** (`1/3`).
- Treasury remains the only later feature family retained.
- PR #43 completed V3-018 using immutable `v3-champion-gates-001`, pre-registered before the repaired candidate was evaluated. `UST-EXP-004` is **NOT_PROMOTION_READY** and V3-019 remains blocked.
- EXP-005 tested a fixed, pre-registered `extra_trees_calibrated_v1` model on the same 53-feature Treasury lane and frozen samples.
- EXP-005 materially improved probability calibration versus `UST-EXP-004`: mean Brier improved from about `0.2249` to `0.1996`, and mean ECE from about `0.1805` to `0.1096`.
- That improvement is not enough: mean relative Brier improvement remains negative (`-0.0985`), with zero positive relative-Brier folds; mean return Spearman remains negative (`-0.0782`); mean drawdown Spearman remains negative (`-0.0354`).
- EXP-005 therefore fails the unchanged absolute prediction prerequisite even though it ranks first in its comparison tournament. Rank remains separate from promotion readiness.
- EXP-005 is frozen as a negative experiment; its registered parameters must not be tuned post hoc under the same experiment/version.
- V3-017 remains hard-gated at **1.00x**. No champion has been selected; `v3_019_eligible = false`.
- Immutable repaired evidence is summarized in `v3/reports/integrity_rebuild_summary.json`; V3-018 evidence is preserved in the `champion_*` reports; EXP-005 evidence is preserved in `exp005_*` reports and `v3/experiments/EXP-005/manifest.json`.

## Next

1. Do **not** proceed to V3-019; zero candidates pass V3-018.
2. Any next prediction experiment must use a new experiment/version and a genuinely different formulation rather than tuning EXP-005 after seeing its result.
3. Prioritize the remaining absolute prediction failures: classification must beat the base rate robustly, while return and drawdown rank correlations must become positive and robust across chronological folds.
4. V3-014 breadth may resume only when a point-in-time historical source satisfies issue #31.
5. V3-015 credit spreads may resume only with a compliant point-in-time source/license.
6. Only when a future candidate passes the existing prediction prerequisite should the previously blocked V3-018 portfolio/cost/perturbation evidence be generated and evaluated.
