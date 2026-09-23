# Phase A acceptance record

Scope: roadmap Phase A, steps 1–3 only.

## Deliverables

| Roadmap step | Delivered artifacts |
|---|---|
| 1 — Research questions and claim boundaries | research_protocol.md: primary/co-primary endpoints, hypotheses, comparators, access assumptions, fairness, statistics, and freeze policy. |
| 2 — Project and configuration contract | src layout, artifact folders, explicit JSON configuration/schema, dependency snapshot, CLI, immutable foundation records, environment/source/config hashes, seed namespaces. |
| 3 — Module interfaces and access | Immutable records, trainer/scorer/aggregator Protocols, compatibility/identity checks, access matrix, independent future-pipeline catalogue. |

## Verification

The Phase A unittest suite contains 22 tests and passed during setup. It covers:

- Valid defaults and schema, unknown fields/scope, invalid numerics and split ratios.
- Duplicate JSON keys, non-finite values, unsafe paths, output aliases into the archive.
- Immutable configuration snapshots and formatting-independent configuration fingerprints.
- Unique foundation runs, linked artifact hashes, explicit pending dataset registration.
- No writes from validation and no dataset reads required to initialize a foundation run.
- Source fingerprints excluding archive data while detecting source changes.
- Separate random streams and a fixed global split across matched seeds.
- Trusted-only scoring/aggregation and correct local-training roles.
- Compatibility, stale round/parent, duplicate client and unknown-assessment rejection.
- Feasible candidate contracts and explicit failure handling.
- Future pipelines remaining planned and training commands unavailable.

Run the checks again with scripts/check_phase_a.py after relevant source changes.
Generated command evidence belongs under reports/generated and is not research evidence.

## Scope limit and honest status

No CSV audit, provenance repair, deduplication, split creation, preprocessing fit,
client partitioning, model training, attack logic, quality/risk scoring, reputation
update, NSGA-II implementation, experiment evaluation, or deployment was performed.
Later modules are contracts/catalogue entries only. The archive and original planning
documents were not modified.

Dataset hashes remain pending Phase B. No pretraining PASS or research result is
claimed. No Git repository existed at setup; source fingerprints supplement the
recorded unavailable commit identity. Packaging via editable installation and hardware
beyond the observed Windows/Python environment have not been validated.

The Phase A foundation is ready for a separate Phase B implementation task.
