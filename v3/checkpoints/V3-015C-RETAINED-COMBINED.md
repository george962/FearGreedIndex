# V3-015C Combined Feature-Stack Checkpoint

## Status

**COMPLETE — DO NOT RETAIN THE 76-FEATURE COMBINED SET**

V3-015C stacked QQQ/SPY relative strength, Treasury, and broad-dollar features with the frozen baseline and evaluated the 76-feature candidate before decision-policy research.

## Integrity correction

The original combined KEEP conclusion depended on feature-family evidence that was not reproducibly preserved on `main`, including a V3-013 candidate built from source lineage that did not match the final frozen QQQ/SPY manifest. The repository-integrity repair reran all component ablations and this combined ablation from the frozen inputs under current code. That clean rerun is authoritative.

## Candidate

- Baseline: 41 features
- QQQ/SPY relative strength: 12
- Treasury: 12
- Broad dollar: 11
- Combined total: **76 model features**
- Feature version: `v3-features-006-retained-combined`

All component point-in-time rules remain intact, including the broad-dollar T+1 availability lag.

## Frozen experiment contract

- Cutoff: `2026-08-18`
- Later-maturing outcomes censored identically
- Identical models, hyperparameters, seeds, folds, labels, and maturity gates
- Identical realized-date sample hashes required before comparison
- Same pre-registered 2-of-3-lanes retention rule used for prior families

## Corrected result

- Combined-set decision: **REJECT**
- Robust lanes in both full-interface models: **1 / 3**
- Classification: not robust in both full models
- Return regression: robust in both full models
- Drawdown regression: not robust in both full models
- Realized-date sample hashes: **matched**
- Best-ranked full candidate in this ablation: `COMBO-EXP-004`
- Promotion-ready experiments: **none**
- Champion selected: **no**
- Trading policy changed: **no**

`COMBO-EXP-004` may rank first inside this direct ablation tournament, but the feature stack fails its pre-registered retention rule and no longer represents the retained research configuration.

## Evidence

- `v3/reports/retained_combined_ablation.json`
- `v3/reports/retained_combined_ablation_comparison.csv`
- `v3/reports/retained_combined_ablation_lane_summary.csv`
- `v3/reports/retained_combined_ablation_tournament.csv`
- `v3/reports/retained_combined_family_context.csv`
- `v3/reports/retained_combined_result_REJECT.txt`

## Roadmap implication

The strongest retained feature family is Treasury. V3-016 remains a model-agnostic research policy interface and must not bind itself to the rejected combined stack. Promotion remains deferred to V3-018/V3-019.
