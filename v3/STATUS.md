# FearGreedIndex v3 Status

`PLAN.md` remains the authoritative original roadmap. This file is the current source of truth so research status does not have to be inferred from branches, stale PRs, or old chat history.

## Current operating state

- **Production:** frozen v2.1 dashboard/action engine. Tactical sizing remains disabled at `1.00x`.
- **Research challenger:** frozen `STAB-004` ranking methodology, visible only in immutable shadow mode. It cannot change production BUY / WAIT / HOLD or sizing.
- **Untouched evidence:** EVID-001 feature/source evidence and matching shadow predictions are collected after `2026-08-18`; forward outcomes remain sealed.
- **Champion state:** no V3 champion; `v3_019_eligible = false`.

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
| STAB-001 Past-only relationship stability selection | COMPLETE — REJECT / RANKING PROMISING | issue #62 / PR #63; mean AUC ~0.573 and AUC >0.50 in 3/3 folds, but relative Brier <0 in 3/3 folds |
| STAB-002 Nested causal calibration | COMPLETE — REJECT | issue #64 / PR #65; ECE improved materially, but positive calibration slope only 1/3 folds and relative Brier remained negative |
| STAB-003 Long/short consensus + abstention | COMPLETE — REJECT / ADAPTIVE RANKING CLUE | issue #70 / PR #71; 2024 supported AUC ~0.656, but support only 1/3 folds and fixed training score thresholds shifted out of distribution |
| STAB-004 Causal rolling normalization + redundancy control | COMPLETE — REJECT / STRONG RANKING EVIDENCE | issue #72 / PR #73; AUC `0.621 / 0.662 / 0.731`, mean ~`0.672`; frozen gate failed because 2025 coverage `0.552` exceeded `0.55`; no retuning |
| V3 shadow dashboard visibility | COMPLETE | issue #74 / PR #75; STAB-004 challenger visible as `RESEARCH ONLY / NO PRODUCTION EFFECT`; production engine unchanged |
| V3-021 Immutable shadow prediction history | COMPLETE | issue #76 / PR #77; hash-chained point-in-time prediction ledger, production-vs-V3 dashboard history, atomic forward collection; all final-head checks passed before merge |

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
- STAB-001 showed that causal ordering information was more stable than absolute probability level: mean AUC ~**0.573**, AUC >0.50 in all three exposed folds, but poor relative Brier.
- STAB-002 confirmed that a simple causal calibration layer is not enough. ECE improved, but calibration slope orientation remained unstable and relative Brier stayed negative.
- STAB-003 directly tested relationship adaptation. Its supported 2024 consensus reached AUC ~**0.656**, but fixed training score quantiles transferred badly. Ranking survived while absolute score location shifted.
- STAB-004 addressed that score-location drift with a fixed 252-score causal reference window and replaced semantic-family counting with training-only redundancy clustering. All three folds retained structural support and raw-score AUCs were approximately **0.621 / 0.662 / 0.731** (mean **0.672**).
- STAB-004 remains formally rejected. The pre-registered maximum supported-fold non-abstain coverage was `0.55`; 2025 produced `0.552`. The threshold was **not** relaxed after seeing the result. Tail separation was also asymmetric in 2024 and 2025. This is the strongest exposed-fold ranking evidence so far, not a promotion result.
- DIAG-001 inspected the 2024, 2025, and 2026 YTD outcomes. Therefore those periods are **research-exposed** for any post-DIAG formulation. They may be used for development diagnostics, but not presented as fresh final promotion evidence for a model designed from them.
- EVID-001 is active. The first sealed feature row is `2026-08-19`; outcomes remain absent/sealed. The frozen V3-015 Treasury research snapshot remains immutable, while new Treasury observations are captured separately with capture-date provenance.
- EVID-001 never hindsight-backfills a missed decision date, never rewrites a sealed same-day row, and never generates labels/outcomes. A forward date becomes research-exposed only through an explicit checkpoint-opening action.
- V3-021 adds a separate immutable prediction chain bound to those sealed feature rows. Initial recorded shadow states are: `2026-08-19` **79.4% / ABSTAIN**, `2026-08-20` **81.7% / STRONG_FAVORABLE**, and `2026-08-21` **84.9% / STRONG_FAVORABLE**. All use 252 prior score references.
- The production v2.1 action was `HOLD / NO EXTRA BUYING` on all three initial shadow dates, while its fast timing layer was `NEUTRAL / NO TIMING EDGE`. That disagreement is now preserved for later evaluation; it does not justify changing production or retuning V3.
- The initial V3 shadow prediction chain head is `dd29baf39a69a93ec86040949141e993da99c0aa152855a76c17377cb8775779`. Existing prediction rows are semantically replayed and cannot be rewritten if later code changes.
- Feature evidence and its matching shadow prediction are collected atomically by the guarded forward-evidence workflow. The dashboard reads committed predictions rather than reconstructing old calls after the fact.
- The V3 dashboard renderer remains downstream of production. CI hashes `site/analysis.json` before V3 rendering and verifies the hash is unchanged afterward.
- V3-017 remains hard-gated at **1.00x**. No champion has been selected; `v3_019_eligible = false`.

## Repository organization

- `PLAN.md` — original roadmap/methodology rules.
- `STATUS.md` — current truth, including the production/research boundary.
- `experiments/EXP-XXX/` — pre-registration and immutable predictive-experiment contracts.
- `methodology/STAB-XXX/` — pre-registered relationship/stability methodology experiments that may justify later predictive experiments but cannot themselves promote a champion.
- `diagnostics/` — diagnostic-only research; no model/feature selection without a later pre-registered experiment.
- `evidence/forward_feature_ledger.csv` — sealed post-cutoff point-in-time feature evidence.
- `evidence/shadow_prediction_ledger.csv` — immutable point-in-time challenger predictions generated before outcomes are opened.
- `evidence/forward_checkpoints.json` — explicit untouched-evidence unseal registry.
- `checkpoints/` — human-readable conclusions and infrastructure checkpoints.
- `reports/` — compact machine-readable evidence.
- generated Parquet outputs remain rebuildable and normally stay out of Git.

See `v3/experiments/README.md`, `v3/diagnostics/README.md`, and `v3/evidence/README.md` for lifecycle rules.

## Next

1. Keep the merged V3-021 pipeline unchanged while it accumulates immutable forward shadow predictions; do not change STAB/ADAPT methodology inside this evidence lane.
2. Keep collecting EVID-001 feature evidence **and the matching frozen STAB-004 shadow prediction** without opening forward outcomes.
3. Do **not** relax STAB-004 thresholds or reinterpret its failed viability gate. Treat it as the frozen benchmark challenger for shadow observation.
4. Do **not** proceed to V3-019 or empirical sizing; production remains v2.1 at `1.00x`.
5. Prioritize additional never-used regime evidence (the DATA-001 / long-history core-market work) before another adaptive methodology iteration. The purpose is to test whether the ranking relationship survives substantially different market regimes rather than repeatedly optimizing 2024–2026.
6. If a future STAB/ADAPT successor is pursued, give it a new pre-registered methodology ID and keep the STAB-004 shadow ledger immutable. Never rewrite historical STAB-004 calls with the successor.
7. Keep EVID-001 outcomes sealed until a deliberately pre-registered checkpoint-opening decision. A later opening evaluates what the frozen challenger actually predicted; it must not be used retroactively to redesign those predictions.
8. V3-014 breadth may resume only with a valid point-in-time source; credit spreads remain source/license gated.
