# STAB-004 — Causal rolling score normalization with redundancy control

STAB-004 is complete and frozen as **REJECT / STRONG RANKING EVIDENCE**.

## Frozen result

- decision: `DO_NOT_ADVANCE_CAUSAL_ROLLING_NORMALIZATION_UNDER_STAB_004`
- methodology viability: **FAIL**
- support: **3/3 folds**
- mean raw-score ROC AUC: **0.671686**
- minimum fold ROC AUC: **0.621443**
- AUC > 0.52: **3/3 folds**
- favorable enrichment > +5pp: **2/3 folds**
- unfavorable depletion > +5pp: **2/3 folds**
- strong-side minimum call-count gate: **PASS**
- exact EXP-006 sample hashes: **PASS**
- EVID-001 outcomes: **SEALED**
- production state: unchanged; no champion, V3-019 blocked, sizing `1.00x`

## Fold evidence

| Fold | Representative clusters | ROC AUC | Strong favorable | Favorable prevalence | Strong unfavorable | Unfavorable prevalence | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 | 6 | 0.621443 | 34 | 0.764706 | 54 | 0.555556 | 0.349206 |
| 2025 | 4 | 0.662342 | 84 | 0.380952 | 54 | 0.129630 | **0.552000** |
| 2026 YTD | 3 | 0.731273 | 25 | 0.640000 | 21 | 0.047619 | 0.335766 |

The frozen coverage gate requires every supported fold to remain between `0.25` and `0.55`, inclusive. The 2025 fold produced `0.552`, exceeding the maximum by `0.002`, so STAB-004 fails exactly as pre-registered. The threshold must not be relaxed after seeing the result.

## Interpretation

STAB-004 materially strengthens the ranking hypothesis. Training-only redundancy clustering restored structural support in all three folds, and the raw score ranked favorable-entry outcomes above chance in every exposed fold. The causal rolling normalization also avoided STAB-003's severe one-sided call collapse.

However, the methodology does **not** satisfy its complete frozen viability contract. In addition to the formal coverage failure, separation is asymmetric by fold: 2024 has strong favorable enrichment but essentially no unfavorable depletion, while 2025 has strong unfavorable depletion but negative favorable enrichment. 2026 YTD shows strong separation on both sides. This is useful development evidence, but it is not sufficient to declare a validated decision layer.

Do not retune the `0.90` redundancy threshold, `252/126` rolling window, `20/80` call thresholds, or `0.25–0.55` coverage gate under STAB-004. Any next hypothesis must receive a new pre-registered ID and must not use EVID-001 outcomes for selection.

## Next research direction

The result supports continuing to investigate the **ranking signal itself**, because all three fold AUCs are materially positive. A next methodology should be framed separately and should address cross-fold asymmetry in the two tails rather than simply widening the STAB-004 coverage gate. Probability calibration and production sizing remain deferred.
