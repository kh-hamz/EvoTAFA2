# Module and data-access contracts

Phase B is implemented in separate domain and orchestration modules. See
[Phase B architecture, contracts, and commands](phase_b.md) for the current dataset
pipelines. The foundation specification below records the Phase A baseline;
its planned data boundaries are now implemented for roadmap steps 4-9.

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
No concrete trainer, scorer, aggregator, final evaluator, or dataset loader exists yet.

## Serialization, versions, and evolution

Use explicit JSON writers/readers for metadata, fixed enum strings, and tuple-to-array
encoding. Reject unknown fields/versions at external boundaries. Never deserialize
arbitrary Python objects from untrusted clients. Tensor storage and network transport
will be selected in their implementation phases and referenced through ArtifactRef.

Schema changes require version increments and documented migrations. Preserve older
run snapshots; never reinterpret their labels or feature ordering in place. The
configuration schema currently uses Draft 2020-12 with no network schema resolution.
