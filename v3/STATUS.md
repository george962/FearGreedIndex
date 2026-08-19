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
| V3-012 Controlled VIX ablation | COMPLETE — REJECT | PR #28; frozen as-of 2026-08-18; `0/3` robust lanes in both full models |
| V3-013 QQQ/SPY relative strength | COMPLETE — REJECT | repaired in PR #41; manifest-matching frozen rerun; `1/3` robust lanes |
| V3-015A Treasury rates/yield curve | COMPLETE — KEEP | repaired/current-code rerun in PR #41; `2/3` robust lanes |
| V3-015B Broad U.S. dollar | COMPLETE — REJECT | repaired/current-code rerun in PR #41; `1/3` robust lanes |
| V3-015C 76-feature combined stack | COMPLETE — REJECT | repaired/current-code rerun in PR #41; `1/3` robust lanes |
| V3-016 Prediction-to-action policy | COMPLETE | `v3-decision-policy-001`; research-only, deterministic, no sizing/sell semantics |
| V3-017 Minimal sizing layer | IMPLEMENTED / ACTIVATION BLOCKED | `v3-sizing-policy-001`; current multiplier remains `1.00x`; `1.10x` requires promotion-ready prediction |
| V3-018 Champion acceptance gates | COMPLETE — CURRENT CANDIDATE NOT PROMOTION READY | PR #43, `v3-champion-gates-001`; fail-closed; V3-019 ineligible |
| EXP-005 Chronologically calibrated ExtraTrees | COMPLETE — REJECT | PR #45; calibration improved materially but absolute prediction gate still fails |
| EXP-006 Opportunity-state target reformulation | COMPLETE — REJECT | issue #46 / PR #47; stationary 20d favorable-entry target fails; strong cross-year relationship reversal |

## Blocked / deferred

- **V3-014 market breadth:** `DATA_SOURCE_BLOCKED`. Do not backfill today's constituent universe into history. Issue #31 records point-in-time source requirements.
- **V3-015 credit spreads:** source/license gated. Do not redistribute or silently depend on restricted ICE/Moody's history without a compliant source.
- **V3-019 champion manifest / production promotion:** blocked because zero candidates pass V3-018.
- **V3-017 1.10x empirical sizing:** blocked because zero candidates pass the prediction/champion prerequisite.

## Current evidence summary

- PR #41 repaired historical V3 repository state and reran affected feature-family experiments under current code using the frozen `2026-08-18` research cutoff.
- Corrected feature-family decisions are:
  - VIX: **REJECT** (`0/3` robust lanes).
  - QQQ/SPY relative strength: **REJECT** (`1/3`).
  - Treasury: **KEEP** (`2/3`).
  - Broad dollar: **REJECT** (`1/3`).
  - 76-feature combined stack: **REJECT** (`1/3`).
- Treasury remains the only later feature family retained.
- PR #43 completed fail-closed V3-018 champion gates. No current candidate passes the absolute prediction prerequisite.
- EXP-005 improved probability calibration but still had negative relative Brier improvement and negative return/drawdown rank correlations.
- EXP-006 changed the prediction problem instead of tuning EXP-005. It tested a pre-registered 20-session favorable-entry target using the same 53-feature Treasury lane.
- EXP-006 also fails: logistic mean relative Brier improvement `-0.669`, mean ROC AUC `0.352`; random forest mean relative Brier improvement `-0.148`, mean ROC AUC `0.356`.
- The important EXP-006 finding is **cross-year instability** rather than merely weak average performance: random-forest AUC is about `0.629` in 2024, then `0.282` in 2025 and `0.157` in 2026 YTD.
- That pattern argues against another stationary-model tweak and supports testing a genuinely regime-aware formulation under a new experiment ID.
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
2. Do not retune EXP-005 or EXP-006 after seeing their results.
3. The next prediction experiment should use a new ID and explicitly test **regime-aware / regime-conditioned relationships**, motivated by the 2024→2025→2026 reversal seen in EXP-006.
4. Pre-register regime definitions using information available on the decision date; do not define regimes from future returns or from test-period performance.
5. Keep the same frozen cutoff, Treasury feature lane, chronological folds, and base-rate comparisons so any improvement is attributable to the regime formulation.
6. V3-014 breadth may resume only when a point-in-time historical source satisfies issue #31; credit spreads remain license/source gated.
