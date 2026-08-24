# DATA-001 — Long-history core regime dataset

## Purpose

Increase independent market-regime coverage before another model or methodology iteration. This checkpoint is **data infrastructure only**: no model is fit, no CORE-001 result is inspected, and no production/champion/sizing state changes.

## Frozen source contract

- research window: `1990-01-02` through `2026-08-18`;
- S&P 500 daily OHLC: `^GSPC` through the repository's pinned `yfinance==0.2.65` acquisition path;
- VIX daily close: first-party Cboe historical VIX file;
- DGS2/DGS10: Federal Reserve H.15 series via FRED graph CSV endpoints;
- all three sources are frozen as deterministic checked-in gzip CSV snapshots with source/acquisition, normalized, and compressed snapshot hashes;
- downstream research reads the frozen snapshots and does not silently refresh them.

## Frozen feature registry

Feature set: `v3-long-history-core-001` — 47 features.

- 21 S&P 500 return/trend/high-low-position/realized-volatility features;
- 11 VIX level/change/return/percentile/moving-average-distance features;
- 15 Treasury 2Y/10Y/slope/change/percentile features;
- no CNN Fear & Greed fields;
- no cross-family interactions in DATA-001;
- VIX is joined to the exact decision session;
- Treasury is a backward-only as-of join and carries its source observation date for leakage validation.

## Label contract

Reuse the existing V3 executable-entry convention:

- decision on session `T`;
- entry at the **next tradable S&P 500 session open**;
- 5/20/60-session returns and path drawdowns preserve outcome-known dates;
- `favorable_entry_20d` matches EXP-006 exactly: forward return >= `+2%` and max drawdown strictly greater than `-5%`.

## Acceptance gates

DATA-001 is accepted only if CI proves:

1. frozen snapshot hashes reproduce exactly;
2. no source observation date is after its decision date;
3. the feature registry contains exactly 47 unique features and no Fear & Greed/outcome fields;
4. next-session-open labels reproduce the existing V3 convention;
5. at least 100 complete feature rows exist in each of the 1990s, 2000s, 2010s, and 2020s buckets;
6. the dataset remains infrastructure-only: `champion_selected=false`, `v3_019_eligible=false`, sizing `1.00x`, and `core_001_evaluated=false`.

## After DATA-001

Do **not** tune features from DATA-001 outcomes. Once the source snapshots and data validation are frozen, create a separate pre-registration for `CORE-001` with broad chronological market-era folds before running predictive evaluation. Only after CORE-001 is frozen should `FG-INC-001` test the incremental value of audited-era Fear & Greed on identical dates/samples.
