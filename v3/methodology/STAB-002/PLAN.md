# STAB-002 — Nested causal calibration of the STAB-001 ranking score

## Question

STAB-001 recovered a materially better ordering signal (AUC > 0.50 in all three exposed folds) but failed Brier calibration. Can that **unchanged** past-only ranking score be mapped to useful probabilities using only strictly earlier out-of-sample calibration predictions?

## Frozen boundaries

STAB-002 may not change:

- STAB-001 feature-selection thresholds;
- STAB-001 feature direction or weight rules;
- the 53-feature Treasury registry;
- EXP-006 `favorable_entry_20d` target;
- outer 2024 / 2025 / 2026-YTD test samples;
- executable-entry or outcome-maturity conventions.

EVID-001 outcomes remain sealed. The outer folds are development evidence only.

## Nested calibration protocol

For each outer fold independently:

1. Form the legal outer training set exactly as STAB-001.
2. Take the last **240 legal training rows** as calibration history.
3. Split those rows into **three contiguous 80-row blocks**.
4. Before each 80-row block begins, rebuild the frozen STAB-001 selector using only outcomes legally known by the prior calendar day.
5. Score the 80 rows with that prior-only selector. Do not use those 80 outcomes in feature selection or score construction.
6. Pool all three blocks to create exactly 240 chronological out-of-sample `(raw_stability_score, target)` calibration pairs.
7. Fit one fixed one-dimensional Platt model:
   - `LogisticRegression`
   - `C=1.0`
   - `solver=lbfgs`
   - `max_iter=1000`
   - `random_state=42`
8. Calibration slope must be strictly positive. A zero/negative slope fails closed and may not be inverted.
9. Refit the unchanged STAB-001 selector on all legal outer training rows, compute the outer-test raw score, then apply the frozen Platt calibrator.

## Support gate

Every outer fold requires:

- exactly 240 calibration rows;
- exactly 3 x 80 calibration blocks;
- every inner formation set has at least 400 legally mature rows;
- every calibration block contains both target classes;
- every inner formation selects at least 3 features;
- every inner formation contains at least 2 feature families;
- positive Platt slope;
- exact outer sample-hash match to STAB-001/EXP-006.

## Evaluation

Preserve per outer fold:

- calibrated Brier;
- base-rate Brier and relative Brier improvement;
- AUC;
- 10-bin ECE;
- raw STAB-001 CDF Brier/ECE on the same dates;
- Brier improvement versus raw STAB-001;
- Platt slope/intercept;
- all nested formation/support details.

## Viability gate

STAB-002 passes only if all support conditions pass and:

1. mean AUC > `0.52`;
2. AUC > `0.50` in all 3 folds;
3. minimum fold AUC >= `0.50`;
4. mean relative Brier improvement vs legal training base rate > `0`;
5. positive relative Brier improvement in at least 2/3 folds;
6. minimum fold relative Brier improvement >= `-0.05`;
7. calibrated Brier beats raw STAB-001 in at least 2/3 folds and in aggregate;
8. mean calibrated ECE is lower than mean raw STAB-001 ECE;
9. mean calibrated ECE <= `0.15`.

## Interpretation

**PASS:** justify a separately pre-registered `EXP-010` adaptive candidate using the frozen stability selector + nested calibration and a separately specified drift/uncertainty abstention gate.

**FAIL:** preserve STAB-002 and do not tune calibration window, block count, C, or calibrator class under this ID. Revisit data coverage or target structure under a new ID.

STAB-002 cannot select a champion, unlock V3-019, change policy, or increase sizing.
