# ADAPT-001 — Causal long/short relationship consensus with abstention

## Question

Can the system handle changing market relationships by trusting a feature only when its **long-memory stable direction** agrees with its **recent short-memory direction**, and otherwise abstaining rather than forcing a prediction?

## Motivation

STAB-001 recovered useful ranking across all three exposed folds, but STAB-002 showed that the causal score-to-target orientation itself had a negative recent slope in two of three folds. A static mapping is therefore not enough.

ADAPT-001 tests daily causal relationship adaptation without adding another model family.

## Frozen inputs

- research cutoff: `2026-08-18`
- feature set: `v3-features-004-treasury` (53 features)
- target: EXP-006 `favorable_entry_20d`
- exact EXP-006 / STAB-001 outer test periods and sample hashes
- unchanged STAB-001 long-memory selector
- EVID-001 outcomes remain sealed

The outer folds are development evidence only.

## Daily causal rule

For each realized outer-test decision date `D`:

1. legal history contains only rows whose target was known by `D-1` calendar day;
2. require at least 400 legal rows;
3. run the frozen STAB-001 selector on all legal history;
4. take the latest 126 legally mature rows as short memory;
5. compute short-memory Spearman for each long-selected feature;
6. short-memory association is usable only with >=100 valid rows, both target classes, and `|rho| >= 0.03`;
7. a feature is consensus-active only when usable short-memory sign equals long-memory sign;
8. consensus weight share is active long-memory absolute weight divided by all long-selected absolute weight;
9. the date is ACTIVE only with >=3 consensus features, >=2 feature families, and weight share >=0.60;
10. otherwise the date ABSTAINS and receives neutral rank `0.50`;
11. ACTIVE rank uses only consensus features and unchanged long-memory directions/weights;
12. features are transformed with the legal-history empirical percentile and the current raw score is converted to its percentile among legal-history raw scores under the same feature set.

ADAPT-001 claims **rank only**, not calibrated probability.

## Frozen evaluation

Per fold preserve:

- exact test sample hash;
- active rows and coverage;
- active target class support/prevalence;
- active ROC AUC;
- all-row AUC with abstentions at 0.50 (diagnostic only);
- active top/bottom rank-quartile favorable-entry prevalence;
- active top-quartile lift over active prevalence;
- median active consensus weight share;
- median active feature/family counts.

## Viability gate

PASS requires all:

- exact sample hashes match STAB-001;
- mean ACTIVE coverage >=30%;
- every fold coverage >=20%;
- every fold has >=50 ACTIVE rows and both target classes;
- mean ACTIVE AUC >0.57;
- ACTIVE AUC >0.50 in at least 2/3 folds;
- minimum fold ACTIVE AUC >=0.48;
- mean ACTIVE AUC beats frozen STAB-001 mean AUC by >0.01;
- top-quartile lift positive in at least 2/3 folds;
- mean top-quartile lift >=5 percentage points.

## Interpretation

**PASS:** justify a separately pre-registered selective EXP-010 whose outputs are opportunity rank + confidence/abstention. Probability calibration remains a separate proof obligation.

**FAIL:** freeze ADAPT-001. Do not tune 126 rows, `0.03`, `0.60`, or support thresholds under this ID. Prioritize more independent point-in-time regimes and/or target redesign.

ADAPT-001 cannot select a champion, unlock V3-019, change policy, or increase sizing.
