# V3-018 Champion Acceptance Gates Checkpoint

## Status

**COMPLETE — CURRENT CANDIDATE NOT PROMOTION READY**

V3-018 implements the deterministic, fail-closed acceptance layer that separates “best-ranked research candidate” from “eligible for champion selection.”

## Frozen gate artifact

- Gate version: `v3-champion-gates-001`
- Pre-registered before current-candidate evaluation: issue #42
- Config: `v3/evaluation/champion_gates_v1.json`
- Engine: `v3/evaluation/champion_gates.py`
- Manifest: `v3/reports/champion_gate_manifest.json`
- Current evidence: `v3/reports/champion_candidate_evidence.json`
- Current assessment: `v3/reports/champion_gate_assessment.json`

Changing thresholds requires a new gate version. The current version may not be relaxed after seeing candidate results.

## Required gates

Every gate must PASS:

1. immutable evidence/lineage completeness;
2. pre-existing absolute prediction-readiness prerequisite;
3. calibration — mean ECE <= 0.10, no fold/horizon ECE > 0.20, positive mean relative Brier improvement;
4. base 2 bps after-cost benchmark-relative portfolio edge;
5. cross-year robustness — at least 2/3 positive folds and no positive fold contributing >70% of positive excess return;
6. Sharpe at least benchmark and frozen v2.1;
7. maximum-drawdown hard/conditional limits;
8. cost robustness at 5 bps and 10 bps;
9. 3/4 predeclared parameter perturbations positive with no hard drawdown-ceiling violation;
10. point-in-time/leakage/sample-hash/frozen-v2.1 data-quality checks.

Missing, NaN, stale, mismatched, failed, or blocked evidence cannot pass.

## Current candidate

The repaired repository retains Treasury as the only later feature family. Its best-ranked full candidate is:

- Candidate: `UST-EXP-004`
- Feature version: `v3-features-004-treasury`
- Model protocol: `random_forest_v1`
- Training protocol: immutable `EXP-004`
- Label version: `v3-labels-001`
- Policy: `v3-decision-policy-001`
- Sizing: `v3-sizing-policy-001`

## Current result

**`NOT_PROMOTION_READY`**

`UST-EXP-004` does not pass the pre-existing absolute prediction-readiness gate. Therefore:

- evidence completeness for champion promotion is intentionally false;
- prediction prerequisite: FAIL;
- sizing-dependent calibration/portfolio/cross-year/risk/cost/perturbation gates: BLOCKED;
- data-quality/integrity evidence remains required and passes;
- V3-017 stays at **1.00x**;
- no artificial 1.10x portfolio evidence is generated;
- `v3_019_eligible = false`;
- champion selected: **no**.

## Validation

Tests cover a fully passing synthetic candidate plus fail-closed behavior for missing/NaN evidence, prediction blocking, calibration boundaries, exact 70% cross-year concentration, Sharpe, drawdown exceptions, 5/10 bps costs, parameter perturbations, data quality, immutable gate version, and current-candidate lineage.

## Next

V3-019 is **blocked** because zero candidates currently pass V3-018. Research must improve the prediction layer without weakening `v3-champion-gates-001`; only a future fully passing candidate may become eligible for V3-019 champion-manifest creation.
