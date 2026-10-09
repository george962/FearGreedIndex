# FearGreedIndex

**Evidence-first market sentiment research.** FearGreedIndex is a read-only dashboard that puts observed Fear & Greed conditions next to historical S&P 500 market conditions, historical analogs, return distributions, downside outcomes and point-in-time decision history. It helps visitors investigate evidence and uncertainty; it does **not** stream live quotes, send brokerage orders or give personalized investment advice.

## What a visitor can do

- See the latest **recorded** Fear & Greed observation and check its date separately from the time the dashboard was generated.
- Investigate the current market regime and price-versus-sentiment context.
- Compare historical analogs, forward outcomes, return distributions, worst outcomes and sample sizes.
- Explore point-in-time replay of prior rule-based decisions and download the published research tables.
- Review limitations, delayed data, small samples and research-only model status.

**New here?** Start with [How to use the dashboard](docs/USER_GUIDE.md) and [dashboard operation/setup](README-dashboard.md). The canonical Pages URL has not been independently verified for this release; after an authorized deployment, obtain the exact link from **Actions → Deploy Fear & Greed Dashboard** instead of assuming a URL is live.

## Local development and reproducibility

Install Python 3.12 and run from the repository root (use an isolated virtual environment):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_feargreed test_fear_greed_market_data test_dashboard test_signal_ledger test_strategy_validation test_v3_challenger test_public_foundation
python scripts/build_dashboard.py
```

Inspect the generated `site/index.html` and `site/decision_history.html` locally. The build uses repository datasets when available and can use the documented market source path; it is **not** a real-time service. See `README-dashboard.md` for deployment, ingestion, source configuration, and research caveats. Changes to Pages settings, publishing, credentials, or indexability require owner approval; the templates intentionally retain `noindex,nofollow` until reviewed.

## Development workflow for Claude Code or Codex

This repository has its **own** agent-development workflow. Start with [`AGENTS.md`](AGENTS.md), [`.agent/WORKFLOW.md`](.agent/WORKFLOW.md), [`.agent/ROADMAP.md`](.agent/ROADMAP.md) and [`.agent/STATUS.md`](.agent/STATUS.md). Implement only the next permitted ExecPlan and record actual test evidence and Git commits. No special agent orchestrator is required. This product roadmap is separate from the v3 research roadmap.

---

## [`v2.1/`](v2.1/) — Frozen operational baseline

The existing operational/research benchmark. It remains active for daily sentiment and SPX data collection, GitHub Pages generation, point-in-time analog decisions, walk-forward/backtest benchmarking, and immutable forward signal-ledger collection.

**Do not retune v2.1 after seeing later outcomes.** It is the benchmark that v3 must beat. Start with [`v2.1/README.md`](v2.1/README.md).

## [`v3/`](v3/) — Predictive research system

The next-generation research framework covers point-in-time features, multi-horizon outcomes, chronological walk-forward evaluation, model candidates, ablations, champion/challenger gates and immutable shadow predictions. See [`v3/README.md`](v3/README.md), [`v3/PLAN.md`](v3/PLAN.md) and, especially, [`v3/STATUS.md`](v3/STATUS.md) for up-to-date status.

As of the reviewed October 9, 2026 repository state, **no v3 champion had passed the promotion gate**. v3 shadow evidence is research-only. The frozen v2.1 system remains operational; tactical sizing remains disabled at a 1.00x baseline.

## Why operational v2.1 files remain at repository root

The root-level runtime predates the version folders. CI, Pages, imports and data-update workflows depend on paths including `FearGreed.py`, `FearGreedHistory.py`, `FearGreedMarketData.py`, `backtest.py`, `config.json`, `strategy_manifest.json`, `scripts/` and `data/`.

These files are intentionally **not** moved piecemeal. A later explicit migration must update the whole import, workflow, data and test boundary atomically while preserving verified behavior. GitHub Actions discovers workflows under the root `.github/` directory.

## Evidence and publication safeguards

Historical comparisons can be biased by incomplete coverage, changed index methods, correlated market episodes and timing conventions. Future outcomes must never enter decisions retroactively. Do not publish restricted source datasets, account information or credentials. Review [publication and source-rights checks](docs/PUBLICATION_PRECHECK.md) before expanding discovery or distribution.
