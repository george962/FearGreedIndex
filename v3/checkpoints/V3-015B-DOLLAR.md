# V3-015B Broad-Dollar Checkpoint

## Status

**COMPLETE — KEEP BROAD-DOLLAR FEATURES FOR LATER RESEARCH**

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

## Result

- Retention decision: **KEEP**
- Robust lanes in both full-interface models: **2 / 3**
- Realized-date sample hashes: **matched**
- Best-ranked full candidate: **baseline random forest (`BASE-EXP-004`)**
- Promotion-ready experiments: **none**
- Champion selected: **no**
- Trading policy changed: **no**

The broad-dollar family is therefore retained as useful independent research information, while absolute champion/deployment gates remain unmet.

## Evidence

- `v3/reports/dollar_ablation.json`
- `v3/reports/dollar_ablation_comparison.csv`
- `v3/reports/dollar_ablation_lane_summary.csv`
- `v3/reports/dollar_ablation_tournament.csv`
- `v3/reports/dollar_result_KEEP.txt`
- `v3/data/dollar_daily.csv`
- `v3/data/dollar_source.json`

## Roadmap state

- V3-014 market breadth remains **DATA_SOURCE_BLOCKED**.
- V3-015A Treasury: complete and retained.
- V3-015B broad dollar: complete and retained.
- V3-015 credit spread: still source/license gated.
- No champion or trading-policy change has been authorized.
