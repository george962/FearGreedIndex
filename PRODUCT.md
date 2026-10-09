# FearGreedIndex — product definition

## Product promise

An evidence-first, publicly accessible market sentiment research application. A user should be able to understand current Fear & Greed conditions, examine what happened after historically similar conditions, compare historical evidence with a baseline, and see the limits of those conclusions without reading code.

## Users and jobs

- Curious investor: "What does today's sentiment mean, and how uncertain is the historical analogy?"
- Research-oriented analyst: "Which regimes, historical windows, and outcomes are represented, and what was known at the time?"
- Open-source contributor: "Can I reproduce the dashboard, inspect assumptions, and submit a scoped improvement?"

## Public experience

1. Overview: current observed sentiment, source and freshness timestamps, market context, clear research-only label, link to historical evidence.
2. Historical Explorer: date/range controls; historical analogs and forward outcomes; sample sizes and caveats; downloadable nonrestricted results.
3. Strategy Lab: historical rule-based results and benchmark comparisons; exploratory filters must be labeled as exploratory, with no implied out-of-sample guarantees.
4. Research Lab: transparent v3 experiment history, model viability gates, frozen shadow predictions, failure modes, and strictly separated operational/research outputs.

## Evidence standard

No implied profitability or precise forecast from historical analogs. Always show sample sizes, date coverage, pricing/decision convention, and material data-quality warnings. Missing/stale sources must not appear as live. Don't call v3 research models production-ready without gate evidence.

## Product non-goals (initial public release)

No brokerage connection, automated trading, personal accounts, subscription/paywall, multi-tenant API, autonomous strategy optimization, or cloud backend without an approved new ExecPlan.

## Release readiness

A first-time visitor can open a live public URL, understand the product purpose, navigate historical evidence, inspect clear limitations and model status, and use the application from mobile. A new contributor can clone the repository, run documented tests, build the site, and know how to propose changes. Published data rights have been reviewed.
