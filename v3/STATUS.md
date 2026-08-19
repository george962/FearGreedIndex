# FearGreedIndex v3 Status

`PLAN.md` remains the authoritative original roadmap. This file is the current source of truth so research status does not have to be inferred from branches, stale PRs, or old chat history.

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
| V3-012 Controlled VIX ablation | COMPLETE — REJECT | PR #28; frozen as-of 2026-08-18; `0/3` robust lanes |
| V3-013 QQQ/SPY relative strength | COMPLETE — REJECT | repaired in PR #41; `1/3` robust lanes |
| V3-015A Treasury rates/yield curve | COMPLETE — KEEP | repaired/current-code rerun in PR #41; `2/3` robust lanes |
| V3-015B Broad U.S. dollar | COMPLETE — REJECT | repaired/current-code rerun in PR #41; `1/3` robust lanes |
| V3-015C 76-feature combined stack | COMPLETE — REJECT | repaired/current-code rerun in PR #41; `1/3` robust lanes |
| V3-016 Prediction-to-action policy | COMPLETE | `v3-decision-policy-001`; deterministic research-only policy |
| V3-017 Minimal sizing layer | IMPLEMENTED / ACTIVATION BLOCKED | `v3-sizing-policy-001`; current multiplier remains `1.00x` |
| V3-018 Champion acceptance gates | COMPLETE — NOT PROMOTION READY | PR #43, `v3-champion-gates-001`; V3-019 ineligible |
| EXP-005 Chronologically calibrated ExtraTrees | COMPLETE — REJECT | PR #45; calibration improved but absolute prediction gate still fails |
| EXP-006 Opportunity-state target reformulation | COMPLETE — REJECT | issue #46 / PR #47; strong cross-year relationship reversal |
| EXP-007 Fixed 20-session 10Y rate regime | COMPLETE — REJECT | issue #48 / PR #49; mean AUC ~0.499; regime ordering flips |
| EXP-008 Fixed Fear & Greed extreme states | COMPLETE — REJECT | issue #50 / PR #51; mean AUC ~0.528, negative mean relative Brier, ordering `0/3` |
| EXP-009 Fixed 504-row recent window | COMPLETE — REJECT | issue #52; mean AUC ~0.463, Brier worse in all 3 folds despite later-fold AUC improvement |

## Blocked / deferred

- **V3-014 market breadth:** `DATA_SOURCE_BLOCKED`. Do not backfill today's constituent universe into history. Issue #31 records point-in-time source requirements.
- **V3-015 credit spreads:** source/license gated. Do not redistribute or silently depend on restricted history without a compliant source.
- **V3-019 champion manifest / production promotion:** blocked because zero candidates pass V3-018.
- **V3-017 1.10x empirical sizing:** blocked because zero candidates pass the prediction/champion prerequisite.

## Current evidence summary

- Historical repository integrity was repaired in PR #41 and remains protected by read-only rebuild CI.
- Treasury remains the only later feature family retained. VIX, QQQ/SPY, broad dollar, and the 76-feature combined stack were rejected under corrected frozen reruns.
- EXP-005 showed calibration can improve substantially without recovering true predictive edge.
- EXP-006 changed the target formulation and still failed; its key diagnostic was RF AUC about `0.629` in 2024, `0.282` in 2025, and `0.157` in 2026 YTD.
- EXP-007 rejected a simple rising-vs-falling 10Y explanation: mean AUC ~`0.499` and regime favorable-entry ordering changes over time.
- EXP-008 rejected a fixed sentiment-extremes explanation: mean relative Brier `-0.0187`, mean AUC `0.5285`, minimum AUC `0.4332`, and hypothesized `EXTREME_FEAR > NEUTRAL > EXTREME_GREED` ordering in `0/3` folds.
- EXP-009 tested time adaptation directly with exactly 504 recent legally mature observations and the unchanged EXP-006 RF. It improves AUC versus full history in `2/3` later folds, but mean AUC is only `0.4632`, positive-AUC folds are `1/3`, mean relative Brier is `-0.1726`, and Brier is worse than full history in `0/3` folds.
- The combined evidence no longer points to a single obvious static regime or a simple training-window problem. The next step should be **diagnosis of concept/covariate drift and feature-target sign changes before another predictive model is trained**.
- V3-017 remains hard-gated at **1.00x**. No champion has been selected; `v3_019_eligible = false`.

## Repository organization

- `PLAN.md` — original roadmap/methodology rules.
- `STATUS.md` — current truth.
- `experiments/EXP-XXX/` — pre-registration and immutable experiment contract.
- `checkpoints/` — human-readable conclusions and historical repair records.
- `reports/` — compact machine-readable evidence.
- generated Parquet outputs remain rebuildable and normally stay out of Git.

See `v3/experiments/README.md` for the experiment lifecycle.

## Next

1. Do **not** proceed to V3-019; zero candidates pass V3-018.
2. Do not retune EXP-005 through EXP-009 after seeing their results.
3. Run a diagnostic-only drift study on the frozen 53-feature Treasury dataset and EXP-006 target before training another model.
4. Quantify year/fold target-prevalence drift, feature-distribution shift, and per-feature feature-target association/sign changes across 2024, 2025, and 2026 YTD.
5. Preserve the diagnostic output as a new versioned research checkpoint; do not promote features merely because an in-sample association looks strong.
6. Use the drift diagnosis to choose the next model formulation under a new pre-registration rather than guessing another window, threshold, or model family.
7. V3-014 breadth may resume only with a valid point-in-time source; credit spreads remain license/source gated.
