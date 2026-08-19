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
| EXP-009 Fixed 504-row recent window | COMPLETE — REJECT | issue #52 / PR #54; mean AUC ~0.463, Brier worse in all 3 folds despite later-fold AUC improvement |
| DIAG-001 Target/covariate/relationship drift | COMPLETE — DIAGNOSTIC | issue #55 / PR #56; all 53 features preserved; broad relationship drift confirmed |
| EVID-001 Untouched post-DIAG forward evidence | COMPLETE — COLLECTION ACTIVE | issue #60 / PR #61; append-only point-in-time feature/Treasury ledgers beginning after `2026-08-18`; no outcomes open automatically |

## Blocked / deferred

- **V3-014 market breadth:** `DATA_SOURCE_BLOCKED`. Do not backfill today's constituent universe into history. Issue #31 records point-in-time source requirements.
- **V3-015 credit spreads:** source/license gated. Do not redistribute or silently depend on restricted history without a compliant source.
- **V3-019 champion manifest / production promotion:** blocked because zero candidates pass V3-018.
- **V3-017 1.10x empirical sizing:** blocked because zero candidates pass the prediction/champion prerequisite.

## Current evidence summary

- Historical repository integrity was repaired in PR #41 and remains protected by read-only rebuild CI.
- Treasury remains the only later feature family retained. VIX, QQQ/SPY, broad dollar, and the 76-feature combined stack were rejected under corrected frozen reruns.
- EXP-005 through EXP-009 tested calibration, target reformulation, a simple rate regime, sentiment extremes, and a fixed recent training window. None recovered robust absolute predictive edge.
- DIAG-001 explains why another broad model tweak is unlikely to solve the problem: **42/53 features (79%)** reverse training-to-test Spearman sign at least once, and **34/53 (64%)** change test association sign across adjacent research years.
- The Fear & Greed family is especially unstable: **17/17** Fear & Greed features have at least one training-to-test sign reversal, and **16/17** change test sign across years.
- The favorable-entry base rate also drifts materially: 2024 test prevalence is about **+13.8 percentage points** above its training history, 2025 about **-4.3 pp**, and 2026 YTD about **-10.3 pp**.
- Covariate shift is material in macro/market context. Maximum absolute standardized mean differences include roughly `1.45` for `treasury_10y_level`, `1.42` for `treasury_10y_percentile_252`, `1.06` for `treasury_2y_level`, and `1.01` for `spx_distance_ma_200`.
- A smaller set of market-stress/position features has stable-looking directional relationships across all three exposed folds, notably `spx_distance_ma_200`, `spx_realized_vol_5`, `spx_realized_vol_20`, `spx_realized_vol_60`, and `treasury_slope_change_20`. These are **hypothesis-generation observations only**, not promotion evidence.
- DIAG-001 inspected the 2024, 2025, and 2026 YTD outcomes. Therefore those periods are now **research-exposed** for any post-DIAG feature/model formulation. They may be used for development diagnostics, but not presented as fresh final promotion evidence for a model designed from DIAG-001.
- EVID-001 is complete and its collection lane is active. The frozen V3-015 Treasury research snapshot remains immutable; new DGS2/DGS10 observations are captured separately with capture-date provenance so later releases cannot be treated as historically known.
- EVID-001 never hindsight-backfills a missed decision date, never rewrites a sealed same-day row, and never generates labels/outcomes. A forward date becomes research-exposed only through an explicit checkpoint-opening action.
- The EVID-001 merge head passed all six exact-head workflows, including the dedicated forward-evidence checks and full historical repository-integrity rebuild. Frozen v2.1 reproduced with zero diff.
- V3-017 remains hard-gated at **1.00x**. No champion has been selected; `v3_019_eligible = false`.

## Repository organization

- `PLAN.md` — original roadmap/methodology rules.
- `STATUS.md` — current truth.
- `experiments/EXP-XXX/` — pre-registration and immutable predictive-experiment contracts.
- `diagnostics/` — diagnostic-only research; no model/feature selection without a later pre-registered experiment.
- `evidence/` — sealed untouched-forward feature/source evidence and explicit unseal checkpoints.
- `checkpoints/` — human-readable conclusions and historical repair records.
- `reports/` — compact machine-readable evidence.
- generated Parquet outputs remain rebuildable and normally stay out of Git.

See `v3/experiments/README.md`, `v3/diagnostics/README.md`, and `v3/evidence/README.md` for lifecycle rules.

## Next

1. Do **not** proceed to V3-019; zero candidates pass V3-018.
2. Do not retune EXP-005 through EXP-009 after seeing their results.
3. Pre-register a **past-only stability-selection methodology** before training another DIAG-informed model. The selection rule must use only pre-fold training information; the exposed 2024–2026 folds are development evidence only.
4. Keep collecting EVID-001 forward snapshots without opening outcomes. Do not use the forward lane for feature/model selection until an explicit checkpoint is deliberately unsealed for a pre-registered experiment.
5. Any feature/model rule motivated by DIAG-001 requires a new pre-registered experiment ID. Do not directly hard-code the stable-looking DIAG-001 features into production or call their exposed-fold performance unseen.
6. Final champion promotion must eventually rely on a deliberately opened, genuinely untouched evidence checkpoint or separately acquired never-used historical data.
7. V3-014 breadth may resume only with a valid point-in-time source; credit spreads remain license/source gated.
