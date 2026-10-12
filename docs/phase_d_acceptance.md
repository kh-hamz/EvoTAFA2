# Phase D acceptance evidence

Date: 2026-10-05.

## Status

**Software verified on synthetic fixtures. Real-data acceptance remains blocked.**

Phase D implements roadmap Steps 14-17, including FLTrust as required by the readiness
refinement. This report does not establish a successful experiment on the downloaded
Edge-IIoTset archive, model convergence, or readiness for adversarial research phases.

## Implemented scope

- Strict Phase D configuration and versioned immutable artifacts with verified lineage.
- Bounded memory-mapped learning arrays using accepted membership and frozen Phase C preprocessing.
- Common task/feature/seed MLP and logistic initializations.
- Majority-class, logistic, centralized MLP, FedAvg, FedProx, coordinate median, trimmed mean and FLTrust.
- Shared federated engine, explicit failure handling, checkpoint selection, and boundary resume.
- Role-specific predictive diagnostics, update/weight diagnostics, timing and memory.
- Explicit learning reviews and acceptance scoped to one experiment identity and seed.
- Six independently invokable commands; no generic train bypass.

Production Phase A-C algorithms, acceptance rules, contracts and configurations were
preserved. Their catalogue availability assertions were updated to recognize D. E-H remain
planned. No poisoning, assessment-based weighting, reputation or evolutionary search was added.

## Verification

Command: `.venv/Scripts/python.exe scripts/check_phase_d.py`

Result: **107 tests in 245.969 seconds, OK**, exit code 0. This comprises the existing
89 Phase A-C/stabilization tests and 18 Phase D tests. No tests were skipped.

- [Full test log](../reports/generated/phase_d_tests.txt)
- [Machine-readable regression evidence](../reports/generated/phase_d_tests.json)

The test evidence includes:

- Known-answer sample-weighted aggregation, even-count median, trimming and insufficient-client rejection.
- FLTrust direction clipping, root-norm normalization, zero norms and no-positive-trust failure.
- FedProx zero-coefficient equivalence and the actual training routine's proximal gradient.
- Single-client equivalence against an independently written Adam training reference.
- Learnable small MLP/logistic fixtures, deterministic initialization and metric support edge cases.
- All eight methods through signed synthetic B/C lineage and a complete scoped D acceptance workflow.
- Actual foundation/B/C/D stage logic for supported multiclass and unseen-attack binary protocols; only the external Wireshark adapter is mocked in this integration test.
- Changed-source and materialized-array tampering rejection, dependency checks and stale resume rejection.
- Unresolved-review blocking and rejection of attempts to approve failed training.
- CPU federated and CUDA centralized checkpoint/resume comparisons with exact parameter equality in the tested environment.

Additional checks passed: `pip check`, command help registration, tracked diff whitespace,
and foundation CLI/pipeline discovery/Phase C configuration loading with imports of torch,
numpy and psutil explicitly blocked. The latter confirms training packages are not required
by those A-C entrypoints.

## Environment and defaults

Windows, Python 3.12.6, PyTorch 2.10.0+cu128, NumPy 2.5.3 and psutil 7.2.2. The detected
GPU is NVIDIA GeForce RTX 3070 with 8192 MiB VRAM, driver 616.92. CUDA availability and
actual execution were verified. Exact installed packages are in requirements-phase-d.lock.

The default configuration uses strict features and CUDA. Neural comparisons use Adam;
FLTrust explicitly records that optimizer adaptation to the published SGD procedure.
The foundation lock and its configuration defaults remain intact. No cross-platform or
cross-version bitwise reproducibility claim is made.

## Real-data limitation

The existing archive validation evidence consists of ineligible historical Phase B results.
No real Phase C completion exists. Consequently, no real Phase D initialization, training
run or acceptance report was authorized or published. Temporary fixture results are test
evidence and are not a replacement for those real gates.

The pending [capture-recovery review](../reports/generated/capture_recovery_review_2026-10-03/review.md)
and [B/C stabilization acceptance](../reports/generated/phase_bc_stabilization_acceptance_2026-10-03.md)
retain their status. Recovery and real-data regeneration require their separately agreed
process. Phase D commands enforce the existing source-verified gate on entry and resume.

The tests use small fixtures and short runs. Full-size memory, runtime and learning behavior
for the configured 100-round real experiments remain unmeasured. Real learning reviews must
be made from those actual results when eligible data becomes available.

## Usage and scope

See [Phase D pipelines](phase_d.md), [implementation specification](phase_d_implementation_plan.md),
and [readiness decisions](phase_d_readiness_refinement_plan.md). Phase D validation requires
current passing regression evidence plus all eight reviewed methods for its declared scope.
A successful scope does not authorize a different task, fold, scenario, feature variant or seed.

No repository push was performed.
