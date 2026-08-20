# STAB-003 — Long-memory / short-memory consensus with abstention

Issue: #70

## Question

Can changing market relationships be handled more robustly by requiring agreement between a slow full-history stability estimate and a faster recent-history estimate, then abstaining when the resulting score is not extreme?

## Why this follows STAB-001 / STAB-002

STAB-001 recovered useful ranking across the exposed chronological folds but failed probability calibration. STAB-002 improved calibration error but weakened the ranking and still failed to beat the legal training base rate. STAB-003 therefore isolates relationship adaptation and abstention rather than attempting another probability transform.

## Frozen contract

The complete pre-registration is in `manifest.json` and GitHub issue #70. In summary:

- retained Treasury feature version only (`v3-features-004-treasury`, 53 features);
- EXP-006 `favorable_entry_20d` target;
- legal outcome-maturity gate before every outer fold;
- STAB-001 selector unchanged for the long-memory view;
- fixed 504-row, 3-block selector for the short-memory view;
- a feature enters the composite only when both selectors agree on direction;
- no DIAG-001 feature names may be hard-coded;
- 20th/80th percentile abstention thresholds are learned from legal training scores only;
- the raw composite remains a ranking score, not a probability;
- exposed 2024/2025/2026-YTD folds are development evidence only;
- EVID-001 outcomes remain sealed.

## Interpretation

Passing STAB-003 would justify pre-registering EXP-010. It would not select a champion, unlock V3-019, or change the 1.00x sizing multiplier.

Failure is preserved as evidence and the fixed window/block/threshold rules may not be retuned under STAB-003.
