# ADAPT-001 — Causal long/short consensus with abstention

## Result

**REJECT — `DO_NOT_ADVANCE_LONG_SHORT_CONSENSUS_UNDER_ADAPT_001`**

The pre-registered daily long-memory/short-memory consensus rule did not improve the STAB-001 ranking signal. It remained active on most dates but lost substantial discrimination, especially in 2025.

- mean active coverage: **0.901545**
- minimum fold active coverage: **0.784000**
- mean active ROC AUC: **0.500235**
- frozen STAB-001 mean AUC: **0.573254**
- AUC gain vs STAB-001: **-0.073019**
- positive active-AUC folds: **2/3**
- minimum active fold AUC: **0.430761**
- positive top-quartile-lift folds: **2/3**
- mean top-quartile lift: **0.040636**

Fold active AUCs were approximately **0.510 / 0.431 / 0.560** for 2024 / 2025 / 2026 YTD. Coverage was approximately **92% / 78% / 100%**. The rule therefore did not isolate a reliably stronger subset; it was too often active while the relationship still reversed in 2025.

## Interpretation

Do **not** tune the 126-row window, short-memory `|rho|` threshold, consensus share, or support cutoffs under ADAPT-001. Those values were pre-registered before the result. The next priority is DATA-001: add genuinely independent historical market regimes using a separate long-history SPX/VIX/Treasury core dataset, then learn/adjudicate adaptation rules on much broader history rather than repeatedly optimizing on the already exposed 2024–2026 sample.

2024–2026 YTD remain development evidence only. EVID-001 outcomes were not opened. No champion, V3-019, policy, or sizing state changed.
