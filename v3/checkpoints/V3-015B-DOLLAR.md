# V3-015B Broad-Dollar Checkpoint

## Status

**COMPLETE — DO NOT RETAIN BROAD-DOLLAR FEATURES**

The Federal Reserve broad U.S. dollar feature family was evaluated independently against the frozen `v3-features-001` baseline.

## Frozen experiment contract

- Experiment cutoff: `2026-08-18`
- Source: Federal Reserve/FRED `DTWEXBGS`
- Candidate feature version: `v3-features-005-dollar`
- Candidate features: 11
- Baseline features: 41
- Candidate total: 52
- Conservative availability lag: observation T is usable no earlier than T+1 calendar day.
- Later-maturing outcomes are censored identically from both lanes.
- Comparable cells must use identical realized-date SHA-256 values.

## Corrected integrity-rerun result

The earlier KEEP statement was not supported by immutable evidence on `main`; the finalizer-generated source/report files were absent. The V3 integrity repair reran the frozen experiment under current code and preserved the result.

- Retention decision: **REJECT**
- Robust lanes in both full-interface models: **1 / 3**
- Classification: not robust in both full models
- Return regression: robust in both full models
- Drawdown regression: not robust in both full models
- Realized-date sample hashes: **matched**
- Best-ranked full candidate in this ablation: `USD-EXP-004`
- Promotion-ready experiments: **none**
- Champion selected: **no**
- Trading policy changed: **no**

The broad-dollar family remains available as a documented negative experiment but is not retained for the main research feature set.

## Evidence

- `v3/reports/dollar_ablation.json`
- `v3/reports/dollar_ablation_comparison.csv`
- `v3/reports/dollar_ablation_lane_summary.csv`
- `v3/reports/dollar_ablation_tournament.csv`
- `v3/reports/dollar_result_REJECT.txt`
- `v3/data/dollar_daily.csv`
- `v3/data/dollar_source.json`

## Roadmap state

- V3-014 market breadth remains **DATA_SOURCE_BLOCKED**.
- V3-015A Treasury: complete and retained.
- V3-015B broad dollar: complete and rejected.
- V3-015 credit spread: still source/license gated.
- No champion or trading-policy change has been authorized.
