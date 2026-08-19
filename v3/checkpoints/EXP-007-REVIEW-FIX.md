# EXP-007 post-merge integrity hardening

Two review findings were addressed after EXP-007 merged:

1. The dedicated EXP-007 workflow now triggers on upstream inputs that can change its samples, target construction, Treasury features, or EXP-006 lineage.
2. The immutable evidence checker now cross-checks the manifest `result` block against regenerated evaluation evidence in addition to validating report hashes.

These changes do not alter the frozen EXP-007 hypothesis or measured negative result. They strengthen the repository contract so future upstream changes cannot leave EXP-007 evidence silently stale or internally contradictory.
