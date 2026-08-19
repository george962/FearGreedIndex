# V3-015C Combined Retained Features Checkpoint

## Status

**COMPLETE — KEEP COMBINED RETAINED FEATURE SET**

The three independently retained feature families were stacked and evaluated together before starting V3-016 decision-policy research.

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

## Result

- Combined-set decision: **KEEP**
- Robust lanes in both full-interface models: **2 / 3**
- Realized-date sample hashes: **matched**
- Best-ranked full candidate: **combined random forest (`COMBO-EXP-004`)**
- Promotion-ready experiments: **none**
- Champion selected: **no**
- Trading policy changed: **no**

The combined retained set is therefore the strongest current research feature configuration, but it still does not satisfy the absolute promotion gates required for champion status.

## Evidence

- `v3/reports/retained_combined_ablation.json`
- `v3/reports/retained_combined_ablation_comparison.csv`
- `v3/reports/retained_combined_ablation_lane_summary.csv`
- `v3/reports/retained_combined_ablation_tournament.csv`
- `v3/reports/retained_combined_family_context.csv`
- `v3/reports/retained_combined_result_KEEP.txt`

## Roadmap implication

V3-016 may now build a **research-only decision-policy interface** against the combined candidate outputs, but must not treat `COMBO-EXP-004` as a production champion. Promotion remains deferred to V3-018/V3-019.
