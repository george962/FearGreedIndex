# FearGreedIndex — UX and visual design contract

A product-specific design standard; no external agent runtime or UI dependency. The product should prioritize trustworthy data interpretation over novelty.

## Principles

- Evidence first: dates, units, data source, vintage, sample count, and uncertainty accompany results.
- Honest state: clearly distinguish LIVE/CURRENT OBSERVATION, HISTORICAL REPLAY, RESEARCH-ONLY SHADOW, STALE/UNAVAILABLE, and UNVERIFIED. Do not use color alone to carry meaning.
- Progressive disclosure: overview first; details, caveats, methodologies, and exports accessible without clutter.
- Responsive and accessible: semantic HTML, keyboard usable controls, sufficient contrast, screen-reader labeling, and mobile scrolling for charts/tables.
- Performance: static publication artifacts with predictable caching; avoid loading entire experiment internals by default. Display loading/error/empty states, and preserve the last verified data timestamp.
- Consistent components: typography scale, spacing tokens, tab/navigation patterns, charts, annotation styles, and warning states. Introduce a frontend component system only when a scoped design plan warrants it.

## Page contracts

- Overview: index value and rating, date/source, price-versus-sentiment chart, actionable explanation of historical methodology, and freshness status.
- History: date picker / date-range selection, event outcomes, baseline comparison, small-sample warnings, explanation of lookahead protection.
- Strategy Lab: benchmark, costs/assumptions, in-sample vs walk-forward distinction, risk/drawdown, no implied live investable return.
- Research Lab: experimental model identity, gate status, shadow ledger provenance, source versions, failures, limitations.

## Decisions still open

Visual palette, exact frontend framework, charting library, navigation labels, and branding details are design decisions for an ExecPlan; don't assume they are already approved or implemented. Review a small representative prototype before wholesale changes.
