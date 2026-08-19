# V3-013 QQQ/SPY Relative-Strength Checkpoint

## Status

**COMPLETE — DO NOT RETAIN RELATIVE-STRENGTH FEATURE FAMILY**

V3-013 added and evaluated a separately versioned QQQ/SPY relative-strength feature family against the frozen 41-feature baseline.

## Integrity correction

The original V3-013 PR (#30) was never merged. More importantly, its preserved KEEP report referenced a candidate dataset/snapshot lineage that does not match the final frozen QQQ/SPY source manifest. The V3 integrity repair therefore reran the frozen experiment from the manifest-matching snapshot under current code. That clean rerun is authoritative.

## Frozen experiment contract

- Cutoff: `2026-08-18`
- Feature version: `v3-features-003-relative-strength`
- Baseline features: 41
- Candidate features added: 12
- Candidate total: 53
- QQQ/SPY frozen snapshot SHA-256: `f37aecaed12de2839e1716180f1c108613963530d6fb139bf177ca9e0f16a4be`
- Baseline frozen dataset SHA-256: `16d6528800ce12ff92b8e6e9f7ec0764450b11165baf7e14d429fde443cd96f1`
- Realized-date hashes must match in every comparable cell.

## Corrected result

- Retention decision: **REJECT**
- Robust lanes in both full-interface models: **1 / 3**
- Classification: FAIL
- Return regression: PASS
- Drawdown regression: FAIL
- Realized-date sample hashes: **matched**
- Best-ranked full candidate in this ablation: `BASE-EXP-004`
- Promotion-ready experiments: **none**
- Champion selected: **no**

The family remains in the repository as a documented negative experiment but is not part of the retained research feature set.

## Evidence

- `v3/reports/relative_strength_ablation.json`
- `v3/reports/relative_strength_ablation_comparison.csv`
- `v3/reports/relative_strength_ablation_lane_summary.csv`
- `v3/reports/relative_strength_ablation_tournament.csv`
- `v3/reports/relative_strength_result_REJECT.txt`
- `v3/data/qqq_spy_daily.csv.gz`
- `v3/data/qqq_spy_source.json`
