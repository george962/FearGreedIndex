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
| V3-018 Champion acceptance gates | COMPLETE — CURRENT CANDIDATE NOT PROMOTION READY | `v3-champion-gates-001`; fail-closed; `UST-EXP-004` fails prediction prerequisite; V3-019 ineligible |

## Blocked / deferred

- **V3-014 market breadth:** `DATA_SOURCE_BLOCKED`. Do not backfill today's constituent universe into history. Issue #31 records point-in-time source requirements.
- **V3-015 credit spreads:** source/license gated. Do not redistribute or silently depend on restricted ICE/Moody's history without a compliant source.
- **V3-019 champion manifest / production promotion:** blocked because zero candidates pass V3-018.
- **V3-017 1.10x empirical sizing:** blocked because zero candidates pass the prediction/champion prerequisite.

## Current evidence summary

- PR #41 repaired historical V3 repository state and reran V3-013, V3-015A/B/C, V3-016, and V3-017 under current code using the frozen `2026-08-18` research cutoff.
- The rerun passed the full V3 test suite, matched realized-date samples, reproduced the frozen v2.1 benchmark, and removed leaked temporary write-enabled finalizers.
- Corrected feature-family decisions are:
  - VIX: **REJECT** (`0/3` robust lanes).
  - QQQ/SPY relative strength: **REJECT** (`1/3`).
  - Treasury: **KEEP** (`2/3`).
  - Broad dollar: **REJECT** (`1/3`).
  - 76-feature combined stack: **REJECT** (`1/3`).
- Treasury is the only later feature family retained. `UST-EXP-004` is the strongest retained full research candidate in its direct ablation tournament.
- V3-018 uses immutable `v3-champion-gates-001`, pre-registered in issue #42 before the repaired candidate was evaluated.
- `UST-EXP-004` is tied to `v3-features-004-treasury`, `random_forest_v1`, immutable training protocol `EXP-004`, and `v3-labels-001`.
- `UST-EXP-004` fails the pre-existing absolute prediction-readiness prerequisite, so V3-018 records **`NOT_PROMOTION_READY`**.
- Because prediction readiness fails, sizing-dependent champion evidence is not fabricated: calibration/portfolio/cross-year/risk/cost/perturbation promotion gates remain BLOCKED for this candidate.
- V3-017 remains hard-gated at **1.00x**.
- No champion has been selected; `v3_019_eligible = false`.
- Immutable repaired evidence is summarized in `v3/reports/integrity_rebuild_summary.json`; V3-018 evidence is preserved in the `champion_*` reports and checkpoint.

## Next

1. Do **not** proceed to V3-019 with the current candidate; zero candidates pass V3-018.
2. Resume research at the prediction layer using only legitimate, independently testable additions or model improvements; do not weaken `v3-champion-gates-001`.
3. V3-014 breadth may resume only when a point-in-time historical source satisfies issue #31.
4. V3-015 credit spreads may resume only with a compliant point-in-time source/license.
5. When a future candidate passes the existing prediction prerequisite, generate the previously blocked V3-018 calibration/portfolio/cost/perturbation evidence under the frozen gate version and reevaluate it.
