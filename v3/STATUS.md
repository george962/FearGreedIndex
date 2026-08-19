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

## Blocked / deferred

- **V3-014 market breadth:** `DATA_SOURCE_BLOCKED`. Do not backfill today's constituent universe into history. Issue #31 records point-in-time source requirements.
- **V3-015 credit spreads:** source/license gated. Do not redistribute or silently depend on restricted ICE/Moody's history without a compliant source.

## Active

- **V3-018 — Champion acceptance gates** is active in issue #42.
- Gate version is pre-registered as `v3-champion-gates-001` before evaluating the repaired current candidate.
- The gate is fail-closed: missing evidence, failed prediction readiness, leakage/data-quality failure, weak calibration, weak after-cost robustness, or failed perturbation checks means `NOT_PROMOTION_READY`.

## Current evidence summary

- The repository-integrity repair reran V3-013, V3-015A, V3-015B, V3-015C, V3-016, and V3-017 under current code using the frozen `2026-08-18` research cutoff.
- The rerun passed the full V3 test suite, matched realized-date samples, reproduced the frozen v2.1 benchmark, and confirmed no temporary write-enabled finalizer workflows remain.
- Corrected feature-family decisions are:
  - VIX: **REJECT** (`0/3` robust lanes, from V3-012).
  - QQQ/SPY relative strength: **REJECT** (`1/3`).
  - Treasury: **KEEP** (`2/3`).
  - Broad dollar: **REJECT** (`1/3`).
  - 76-feature combined stack: **REJECT** (`1/3`).
- The old V3-013 KEEP report is superseded because its candidate dataset/source lineage did not match the final frozen QQQ/SPY manifest. The manifest-matching rerun is authoritative.
- Treasury is the only later feature family retained. `UST-EXP-004` is the strongest retained full research candidate by its direct ablation tournament, but **it is not promotion-ready**.
- No current experiment passes the absolute prediction-promotion gate. No champion has been selected.
- V3-016 remains model-agnostic and research-only.
- V3-017 remains hard-gated at **1.00x**. The 1.10x research sizing path cannot activate before champion/prediction acceptance.
- Immutable repaired evidence is summarized in `v3/reports/integrity_rebuild_summary.json` and guarded by `v3/ci/check_repository_integrity.py`.

## Next

1. Complete V3-018 fail-closed champion gate engine and tests using the already pre-registered thresholds in issue #42.
2. Evaluate the strongest retained candidate without relaxing any gate after seeing the result.
3. Only if exactly one candidate passes every V3-018 requirement may V3-019 create a champion manifest and unlock the V3-017 1.10x research sizing experiment.
