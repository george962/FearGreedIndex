# FearGreedIndex v3 — Research Workspace

v3 is the next-generation predictive research system. It is intentionally separated from the frozen v2.1 baseline so experimental work cannot silently change the current operational strategy.

## Start here

Read [`PLAN.md`](PLAN.md) for the full staged implementation plan, acceptance gates, experiment rules, and recommended order of work.

## Planned structure

```text
v3/
├── README.md
├── PLAN.md
├── features/
├── labels/
├── models/
├── evaluation/
├── experiments/
├── reports/
└── tests/
```

## Current status

- v2.1 remains the frozen benchmark.
- v3 has not been promoted to production.
- Initial work should focus on V3-001 through V3-004 only: baseline preservation, feature dataset, leakage tests, and multi-horizon labels.
- No tactical sizing changes should be made until predictive edge is demonstrated out of sample.
