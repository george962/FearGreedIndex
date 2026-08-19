# V3 Repository Integrity Repair

Status: **COMPLETE**  
Primary repair: PR #41

## Problems that were confirmed

1. V3-013 QQQ/SPY relative-strength assets had not been merged even though later combined-feature code imported them.
2. `v3/STATUS.md` and the generic current-stage hook were stale.
3. Temporary write-enabled research finalizer workflows had leaked onto `main`.
4. Several checkpoints claimed immutable report/manifest files that were not actually preserved in the merge commits.
5. Some earlier KEEP/PASS prose conclusions did not reproduce against the final frozen source manifests.

## Repair rule

Prior prose-only KEEP/PASS claims were not trusted. Affected experiments were rebuilt from frozen point-in-time inputs under current code, and the regenerated evidence became authoritative.

## Corrected conclusions

- VIX: REJECT
- QQQ/SPY relative strength: REJECT
- Treasury/yield curve: KEEP for later research
- Broad dollar: REJECT
- 76-feature combined stack: REJECT

The repair also removed leaked temporary write-enabled workflows, restored missing research assets, fixed clean-checkout workflow dependencies, and added repository-integrity validation.

V3-014 remained data-source blocked and was not fabricated with a survivorship-biased universe.

Current conclusions belong in `v3/STATUS.md`; this checkpoint exists only to preserve the historical repair rationale.
