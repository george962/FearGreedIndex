# STAB-002 — Nested causal calibration

## Result

**REJECT — `DO_NOT_ADVANCE_CAUSAL_CALIBRATION_UNDER_STAB_002`**

Causal Platt calibration substantially improved the raw STAB-001 probability calibration, but it could not establish a stable forward orientation or beat the legal training base-rate predictor.

- mean ROC AUC: **0.512162**
- positive AUC folds: **1/3**
- positive Platt-slope folds: **1/3**
- mean relative Brier improvement: **-0.040660**
- positive relative-Brier folds: **0/3**
- mean ECE: **0.140331** vs raw STAB-001 **0.263269**
- Brier better than raw STAB-001: **3/3 folds**, aggregate **True**
- exact STAB-001 sample hashes matched: **True**

The fitted slope was negative in 2024 and 2026 YTD. STAB-002 correctly failed closed to the training base rate in those folds rather than reversing the score after the fact. Only 2025 had a positive calibration slope, and even there relative Brier was negative.

## Interpretation

The problem is no longer just probability mapping. The **orientation of the score-to-outcome relationship itself changes**. Do not retune the calibration window, block count, C, or model under STAB-002. The next methodology should test causal drift/consensus gating or acquire more independent historical regimes; it should not force a prediction when long- and short-memory relationships disagree.

2024–2026 YTD remain development evidence only. EVID-001 outcomes were not opened. No champion, V3-019, policy, or sizing state changed.
