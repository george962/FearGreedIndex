# STAB-003 Checkpoint — Long/short consensus with abstention

**Status:** COMPLETE — REJECT  
**Issue:** #70  
**Research cutoff:** 2026-08-18  
**Production effect:** none; no champion, V3-019 remains blocked, sizing remains 1.00x.

## Result

The pre-registered long-memory / short-memory consensus methodology did **not** pass its viability gate.

Aggregate development evidence:

- support in only **1/3** folds;
- mean ROC AUC: **0.5522**;
- only **1/3** folds above AUC 0.52;
- minimum fold AUC under the fail-closed unsupported-fold convention: **0.50**;
- favorable-lift gate: **0** passing folds;
- unfavorable-separation gate: **0** passing folds;
- exact EXP-006 sample hashes matched;
- EVID-001 outcomes remained sealed.

## What worked

The supported 2024 fold produced a strong ranking result: **AUC 0.6565** with seven consensus features spanning all three semantic families. This reinforces the STAB-001 finding that causal past-only relationship selection can recover useful ranking information that the earlier static model families missed.

## What failed

Two distinct failure modes appeared.

1. **Absolute score-location drift.** The fixed training 20th/80th-percentile thresholds did not transfer cleanly to the test distribution. In the supported 2024 fold the method emitted **130 strong-unfavorable calls and zero strong-favorable calls**, even while its full-fold ranking AUC was 0.6565. Ranking survived better than absolute score level.
2. **Semantic-family support collapsed.** The 2025 consensus contained four features but only the SPX/interaction family. The 2026-YTD consensus contained six features but again only the SPX/interaction family. Under the pre-registered support rule, both folds correctly failed closed.

## Interpretation

STAB-003 does **not** justify EXP-010 as registered. Do not loosen the 504-row window, 3-block short selector, 0.05 association threshold, semantic-family requirement, or 20/80 abstention thresholds under STAB-003.

The evidence suggests the next methodology should address two different problems rather than simply retune thresholds:

- normalize the composite score against a **causal recent score history** so confidence states can adapt when the score distribution moves; and
- replace coarse semantic-family diversity with **training-only redundancy/independence control** so a group of highly correlated SPX variables does not masquerade as many independent signals, while genuinely distinct market-stress relationships are not rejected merely because they share a naming family.

A new STAB-004 must be separately pre-registered. The exposed 2024–2026-YTD folds remain development evidence only. Final promotion still requires a genuinely untouched forward checkpoint or separate never-used history.
