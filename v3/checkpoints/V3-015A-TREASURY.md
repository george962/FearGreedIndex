# V3-015A Treasury Checkpoint

## Status

**COMPLETE — KEEP TREASURY FOR LATER RESEARCH**

The Treasury-rate feature family was evaluated as an independent addition to the frozen `v3-features-001` baseline under the same fixed-as-of, same-sample rules used for V3-012 and V3-013.

## Frozen experiment contract

- Experiment cutoff: `2026-08-18`
- Source family: Federal Reserve/FRED `DGS2` and `DGS10`
- Candidate feature version: `v3-features-004-treasury`
- Candidate features: 12
- Baseline features: 41
- Candidate total: 53
- Later-maturing outcomes are censored identically from baseline and candidate lanes.
- Every comparable fold/target cell must use the same realized-date SHA-256.
- No champion promotion or trading-policy change is allowed in V3-015A.

## Pre-registered retention rule

A prediction lane is robust only when:

1. its aggregate primary metric improves, and
2. at least 2 of 3 chronological folds improve.

Treasury is retained only when at least 2 of 3 prediction lanes are robust in **both** full-interface models (`EXP-003` gradient boosting and `EXP-004` random forest).

## Result

- Treasury retention decision: **KEEP**
- Robust lanes in both full-interface models: **2 / 3**
- Realized-date sample hashes: **matched**
- Best-ranked full candidate in the ablation tournament: **baseline random forest (`BASE-EXP-004`)**
- Promotion-ready experiments: **none**
- Champion selected: **no**
- Trading decision policy changed: **no**

The Treasury family therefore qualifies as useful independent research information, but the absolute model evidence is still insufficient for champion promotion.

## Evidence

The immutable measured outputs are stored in:

- `v3/reports/treasury_ablation.json`
- `v3/reports/treasury_ablation_comparison.csv`
- `v3/reports/treasury_ablation_lane_summary.csv`
- `v3/reports/treasury_ablation_tournament.csv`
- `v3/reports/treasury_result_KEEP.txt`
- `v3/data/treasury_daily.csv`
- `v3/data/treasury_source.json`

## Roadmap state

- V3-014 market breadth remains **DATA_SOURCE_BLOCKED** until a reliable point-in-time historical breadth source is available.
- V3-015A Treasury is complete and retained.
- V3-015B broad-dollar features are next.
- The V3-015 credit-spread subfamily remains source/license gated.
