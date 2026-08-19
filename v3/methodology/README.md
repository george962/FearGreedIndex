# V3 Methodology Research

This directory contains pre-registered research about **how** predictive models should be constructed or validated, rather than a new production candidate by itself.

## Rules

- Every methodology change receives a unique immutable ID such as `STAB-001`.
- The hypothesis, data contract, selection rule, and viability gate must be written before the outer-fold result is inspected.
- Methodology may use already research-exposed historical folds for development, but those folds cannot later be described as fresh promotion evidence for a model motivated by the methodology.
- Untouched EVID-001 outcomes remain sealed unless a separate pre-registered checkpoint explicitly opens them.
- A methodology PASS may justify a later predictive experiment ID. It never selects a champion, changes policy, or changes sizing by itself.
- A methodology FAIL is preserved. Thresholds may not be tuned after seeing the result under the same methodology version.

## Current methodology

- `STAB-001/` — past-only feature relationship stability selection before any adaptive model is trained.
- `STAB-002/` — nested causal calibration of the frozen STAB-001 ranking score.
- `ADAPT-001/` — daily long-memory/short-memory relationship consensus with abstention.
