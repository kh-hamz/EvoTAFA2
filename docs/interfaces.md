# Module and data-access contracts

Active Phase B/C v2 contracts and load_training_ready are described in the
[stabilization guide](phase_bc_stabilization.md). Phase A 1.0 records remain compatible.

Phases B and C are implemented in separate domain and orchestration modules. See
[Phase B architecture, contracts, and commands](phase_b.md) and
[Phase C architecture and commands](phase_c.md). The foundation specification below
records the Phase A baseline; its runtime data contracts now extend through the
pretraining gate.

Contract version: 1.0. Python records live in contracts/records.py; structural
Protocol interfaces live in contracts/interfaces.py. They define boundaries, not
implementations or serialized array formats.

## Record responsibilities

| Record | Required identity and meaning |
|---|---|
| ArtifactRef | Workspace artifact path and verified SHA-256; unresolved sources cannot pretend to be verified refs. |
| DatasetManifest | Dataset identity, source refs, schema/provenance refs, provenance status, contract version. |
| SplitEntry | Stable observation ID, nullable verified capture/session IDs, group, partition, exclusion reason. |
| PartitionRef | Manifest ref plus declared data role; loaders later verify actual contents. |
| ClientManifest | Client ID, group ownership, local train/validation refs, class counts, partition seed. |
| Compatibility | Preprocessing/ordered-features/label-map hashes, model signature, dimensions, contract version. |
| ClientUpdate | Client/round/parent identity, delta ref, compatibility, registered sample count, time and bytes. |
| ClientAssessment | Quality, expertise/support, separate risk components, current risk, prior reputation. |
| CandidateEvaluation | Evaluation ID, client-keyed weights, optional four objectives, feasibility and failure reason. |
| RoundResult | Round/checkpoint, optional scalar weights, diagnostics, reputation transitions, failure status. |
| ExperimentResult | Run/protocol/seed identities, metrics/cost refs, status and version. |

ClassExpertise.score = None means unsupported. Zero support must not be presented
as evidence of zero expertise. Invalid candidates need a failure reason and do not
receive fabricated favorable objective values. Finite feasible objective values lie
in [0,1]; feasible weights are finite, nonnegative, unique by client and sum to one.
Per-client cap policy and repair belong to Phase F, not the record layer.

## Runtime boundaries already implemented

ClientManifest requires local_train and local_validation roles. ScoringRequest
requires trusted data, matching compatibility and a common parent model.
AggregationRequest accepts trusted data only when data are supplied, checks model,
round and client identities, and rejects duplicate/mismatched assessments.
require_compatible compares the full compatibility record, including feature ordering
and label mapping. Dimension equality alone does not establish compatibility.

These checks do not open data files or verify tensor contents. Python dataclasses
are not a hostile-input parser. Untrusted transport decoding, signature verification,
full count/range validation and artifact-content checks belong to their later loaders.
Data-role checks are application safeguards, not process-level filesystem isolation.

## Access matrix for future implementations

| Consumer | Permitted data | Prohibited data |
|---|---|---|
| Preprocessing fitter | Union of actual local-training rows | Local validation, trusted, selection validation, test |
| Client trainer | Assigned local-training partition; local validation only for declared diagnostics | Global trusted/selection/test data and malicious identity registry |
| FLTrust server-reference operation | Permitted trusted resource and compatible current global model | Client-private training partitions, selection validation, final test, malicious identity registry |
| Attack harness | Selected attackable training rows or submitted delta; simulator attack plan | Trusted, selection, final-test modification |
| Client scorer | Submitted models/deltas and trusted panel | Selection validation, final test, malicious identity registry |
| Fitness/aggregator | Compatible updates, assessments, prior reputation, trusted resource if needed | Selection validation, final test, malicious identity registry |
| Checkpoint selector | Selection-validation results | Final-test results |
| Diagnostic evaluator | Frozen outcomes plus simulator truth when computing attack diagnostics | Feeding simulator truth back into defense |
| Final evaluator | Frozen checkpoints and locked test after freeze | Model/hyperparameter selection from test scores |

No generic all_data or experiment_context object is passed into trainers/scorers.
Malicious identity is deliberately absent from defense-facing records and requests.
Final-test refs are not accepted in scoring/aggregation requests.

## Interchangeable implementations

ClientTrainer.train receives a TrainingRequest and returns ClientUpdate.
ClientScorer.assess receives a ScoringRequest and returns assessments.
Aggregator.aggregate receives an AggregationRequest and returns RoundResult.

Coordinate median/trimmed mean may return weights = None; requiring a scalar vector
would incorrectly exclude those baselines. Future algorithms requiring assessments
must ensure one assessment per valid update. Baselines may use no assessments.
Phase D supplies the concrete client trainer, baseline aggregators, server-reference
operation and verified learning-data adapter. The Phase E client scorer and final
evaluator remain unimplemented. See [Phase D pipelines](phase_d.md).

Phase D methods must work without Phase E assessments. Methods needing assessments
enforce their actual requirements when implemented; never substitute fabricated scores.
FLTrust computes its server reference through a separate trusted-data operation and
records its access budget, training procedure, and update normalization. Its normalized
trust weights describe aggregation of normalized updates, not necessarily raw deltas.
FedProx's local optimization belongs to the trainer. These requirements do not change
the existing TrainingRequest, AggregationRequest, or RoundResult Python contracts.

Future real-data experiment entrypoints and resumed runs call load_training_ready with
explicit dataset, task, fold, and scenario expectations. They consume the returned
verified bundle; missing/stale/ineligible or mismatched inputs fail without a raw-CSV
fallback or eligibility override. Synthetic tests cannot replace this entry check.

## Future round diagnostics

Phase D implements these meanings in a versioned artifact referenced by the existing
RoundResult.diagnostics field; E/F extend it when their features exist. Concrete
serialization and its version are defined during D implementation. No foundation-record
fields or versions change in this specification refinement.

| Measurement | Required meaning |
|---|---|
| Identity | Run, method, task, fold, scenario, seed, epoch/round, parent/checkpoint, and input artifact identities |
| Loss | Separate training-objective and evaluation losses; record data role, observation count, and averaging convention. Aggregate observation means using their counts, not an unweighted mean of unequal client means. |
| Macro-F1 | Use the frozen task vocabulary and record per-class support and the declared undefined-class convention consistently across methods. Do not silently change vocabulary between rounds. |
| Benign FPR | Benign observations predicted as any attack divided by benign observations; zero benign support is undefined, with an explicit reason. |
| Update magnitude | L2 norm of each submitted delta and the actual global parameter change using consistent trainable-parameter ordering. Record absolute norm and relative norm against the parent model; a zero parent norm makes the relative value undefined. |
| Weights | Client-keyed normalized weights where meaningful, with method-specific semantics. Coordinate-wise methods report inapplicable; FLTrust records normalization separately. |
| Weight change | For comparable normalized vectors, half the L1 distance over the union of consecutive client identities, assigning zero to absent clients. Record participation changes; the first round or incompatible semantics is inapplicable. |
| Timing | Elapsed round time, cumulative client training time, scoring, aggregation, selection evaluation, and checkpoint I/O. Parallel client totals need not equal elapsed time. Mark nested search timings so they are not added twice. |
| Failure/support | Explicit rejection, unavailable-metric, and unsuccessful-round reasons. Missing is not zero; retaining a model after failed aggregation is not evidence of convergence. |

The trainer supplies training diagnostics, the scorer/aggregator supplies its own
measurements, and orchestration links them with separately evaluated selection metrics.
Keep trusted-panel, training, local-validation, and selection-validation results tagged
by role. Do not feed selection results into fitness. Final-test metrics are absent from
development diagnostics. Report initial and subsequent epoch/round measurements.

Phase F adds candidate requests, cache hits, actual model evaluations, invalid candidates,
construction/inference/evolutionary overhead, total search time, and peak memory. Count
categories explicitly rather than assuming requests equal evaluations. Profile the actual
evaluation path with environment/workload identities and separate warm-up costs.

## Serialization, versions, and evolution

Use explicit JSON writers/readers for metadata, fixed enum strings, and tuple-to-array
encoding. Reject unknown fields/versions at external boundaries. Never deserialize
arbitrary Python objects from untrusted clients. Tensor storage and network transport
will be selected in their implementation phases and referenced through ArtifactRef.

Schema changes require version increments and documented migrations. Preserve older
run snapshots; never reinterpret their labels or feature ordering in place. The
configuration schema currently uses Draft 2020-12 with no network schema resolution.
