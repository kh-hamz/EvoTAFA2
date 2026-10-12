# Phase C acceptance evidence

Scope: roadmap steps 10-13 only.

## Current stabilization evidence

This document preserves the original Phase C acceptance results. For the v2 changes, the 89-test result, and the pending real-data recovery checkpoint, see the [2026-10-03 stabilization acceptance report](../reports/generated/phase_bc_stabilization_acceptance_2026-10-03.md).

## Software status

The complete Phase A-C test runner passes 64 tests:

- 22 Phase A foundation and contract regressions;
- 33 Phase B dataset-verification regressions;
- 9 Phase C allocation, local split, preprocessing, leakage, orchestration, and gate tests.

The Phase C tests demonstrate deterministic near-IID, Dirichlet, and specialist
strategies; whole-group ownership; explicit rejection of impossible client minima;
group-isolated local validation; training-only fit rows; base and strict frozen
transformers; constant removal; finite values; byte-identical regeneration; and gate
failure when a local-validation observation is inserted into the fitting manifest.

CLI smoke checks confirm four independent commands and show Phases A-C as available.
Phases D-H remain planned. The `train` command is still unavailable.

## Real-data status

No real Phase C client or preprocessing artifact has been published. The saved Phase B
validation for both the smoke and primary selected datasets is `FAIL`, with zero
feasible folds containing a benign-supported trusted panel. `assign-clients` requires
an accepted Phase B validation and a feasible fold from that exact validation, so the
current artifacts are rejected before client allocation.

This preserves the project gate: Phase C implementation readiness does not convert a
failed dataset scope into training eligibility. No raw-source concatenation, repaired
predictor benchmark, unverified benign substitution, or synthetic real-data manifest
was created.

## Completion boundary

The Phase C software scope is implemented and testable with valid inputs. Real-data
Phase C execution remains blocked by the existing Phase B evidence result. Phase D may
consume only a Phase C `validate-phase-c` completion whose report is `PASS` and whose
completion metadata sets `eligible_for_training` to true.

See [Phase C architecture and commands](phase_c.md) and the existing
[Phase B real-data handoff](../reports/generated/phase_b_handoff.md).
