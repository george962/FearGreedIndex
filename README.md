# FearGreedIndex

This repository is organized around two strategy generations:

## [`v2.1/`](v2.1/) — Frozen baseline

The current operational/research baseline. v2.1 remains active for:

- daily Fear & Greed and SPX data collection;
- GitHub Pages dashboard generation;
- point-in-time analog decisions;
- walk-forward/backtest benchmarking;
- immutable forward signal-ledger collection.

**Do not retune v2.1 after seeing later outcomes.** It is the benchmark that v3 must beat.

Start with [`v2.1/README.md`](v2.1/README.md).

## [`v3/`](v3/) — Predictive research system

The next-generation system will replace hand-built analog/timing rules as the primary prediction approach with a disciplined multi-factor predictive research framework.

Planned capabilities include:

- point-in-time feature datasets;
- multi-horizon return and downside labels;
- logistic/regression/tree-model baselines;
- common chronological walk-forward evaluation;
- model tournament and ablation testing;
- VIX, QQQ/SPY, breadth, and macro feature families;
- champion/challenger promotion rules;
- prediction-driven decision policy;
- untouched live v3 signal ledger.

Start with [`v3/PLAN.md`](v3/PLAN.md). That file is the authoritative implementation checklist.

---

## Why are there still files at the repository root?

The existing v2.1 runtime predates the versioned folder layout. Its GitHub Actions workflows, Pages deployment, tests, imports, and scheduled data updates currently depend on root-level paths such as:

- `FearGreed.py`
- `FearGreedHistory.py`
- `FearGreedMarketData.py`
- `backtest.py`
- `config.json`
- `strategy_manifest.json`
- `scripts/`
- `data/`

They are intentionally **not being moved piecemeal**. A partial relocation could break scheduled updates or the dashboard while appearing cosmetically cleaner.

For now, treat those root runtime files as the physical implementation of **v2.1**. The `v2.1/` directory is its version-level documentation/home.

A later dedicated migration can move the v2.1 runtime atomically after updating every workflow, import, test, data path, and Pages build path together.

`.github/` must remain at repository root because GitHub Actions discovers workflows there.

---

## Development rule

New strategy research belongs in `v3/`. The current v2.1 implementation is frozen except for operational reliability fixes that do not change the research methodology.

The immediate v3 sequence is:

1. Freeze permanent v2.1 baseline reports.
2. Build the point-in-time v3 feature dataset.
3. Add leakage/data-quality tests.
4. Build executable-entry multi-horizon labels.
5. Only then begin model comparisons.

See [`v3/PLAN.md`](v3/PLAN.md) for the complete V3-001 through V3-021 roadmap.
