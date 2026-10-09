# FearGreedIndex: how to use the market research dashboard

FearGreedIndex is a **read-only, periodically built** market-sentiment research application. It compares recorded Fear & Greed observations with historical S&P 500 market conditions. It does not place trades, stream real-time market prices, or offer personalized financial advice.

## A first visit

1. Check the **sentiment observation date** and the **report generation time**. The underlying market observation can be older than the page build. The site's update checker detects a new published build; it does not turn historical data into a live feed.
2. Review the market context chart and the current research label. BUY / WAIT / NEUTRAL / INSUFFICIENT EVIDENCE describe rule-based historical comparisons, **not instructions to execute a trade**.
3. Inspect the historical analog sample size, return distribution and downside observations. A small sample or several observations from one sell-off do not establish reliability.
4. Open **Decision history** to examine what the historical replay would have displayed at different dates. The replay is designed to respect point-in-time availability; a future outcome observed only for evaluation must not be read as available at the decision date.
5. Export a CSV to explore results independently. Return columns are decimals (for example, `0.012` means 1.2%) unless the particular download explicitly says otherwise.

## How to interpret evidence

- The Fear & Greed Index partly reflects market behavior. An association between low sentiment and subsequent returns is not automatically a causal investment edge.
- Data coverage, calendar gaps, changing index methodology, price-source revisions, regime shifts and repeated measurements in one market episode limit comparability.
- SPY and the S&P 500 price index are distinct data series. Check the market proxy/configuration used for each analysis before comparing results.
- Historical win rates, analogs and backtests are research results, not evidence of a guaranteed future gain. Out-of-sample evaluation and untouched forward evidence carry more weight than optimized historical fits.
- Experimental v3 shadow models are **not production trading models**. As documented in `v3/STATUS.md` when this guide was written, no v3 champion was approved.

## If the page appears stale

- Look at the observation date, build time and warnings; do not assume your browser refreshes the upstream data.
- If a new build is not available, the current deployment may still display an older verified snapshot.
- See `README-dashboard.md` for operator instructions and the original data/deployment caveats.

## For contributors

Follow `AGENTS.md`, `.agent/WORKFLOW.md` and the applicable ExecPlan. Do not change the frozen v2.1 research methodology or rewrite the immutable v3 forward evidence merely to improve presentation.
