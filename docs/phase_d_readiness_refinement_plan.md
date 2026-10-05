# Phase D readiness refinement specification

Date: 2026-10-04. Status: documentation refinement applied; Phase D-H runtime implementation remains planned.

## Agreed decisions

- FLTrust remains required for Phase D completion.
- Weak or unstable learning requires investigation and evidence review, without a universal numerical score threshold.
- Phase D completion does not depend on Phase E assessments or Phase F optimization.
- Every real-data model experiment requires a fresh source-verified Phase C PASS for its dataset, task, fold, and scenario, including at resume.
- This refinement changes documentation only. It neither authorizes capture recovery nor establishes real-data readiness.

## Exact change map

| Document | Targeted changes |
|---|---|
| [Final roadmap](../Edge-IIoTset_FL_Final_Implementation_Roadmap.md) | Dated refinement notice; Step 13 entry gate; Steps 14-17 ownership and clean evidence; Step 23 assessment-based comparisons; Steps 26/29/31 diagnostic counts, budgets and actual profiling; Steps 32/33 integration/scaling prerequisites; mandatory gates and stopping conditions. Existing step numbers and experiment matrix remain. |
| [Research protocol](research_protocol.md) | Comparator ownership and FLTrust access; real-data entry requirement; clean-learning review; role-specific diagnostics; measurement-based development budgets and final freeze. Hypotheses, endpoints and data scope remain. |
| [Architecture](architecture.md) | Future-stage ownership, baseline independence, FLTrust server-reference operation, training-ready entry boundary, diagnostic ownership and profiling timing. |
| [Interfaces](interfaces.md) | Future diagnostic meanings through RoundResult.diagnostics, method-specific requirements, FLTrust access and loader expectations. Python signatures remain unchanged. |
| [Development workflow](development.md) | Synthetic tests versus real-data experiments; separate evidence categories; next-phase ownership, clean review, loader and profiling requirements. |
| [README](../README.md) | One Start here link to this specification. Availability remains A-C only. |
| This specification | Decisions, change map, preserved areas, future implementation obligations and acceptance checklist. |

## Comparison ownership and completion

| Phase | Ownership and evidence |
|---|---|
| D | Majority-class, logistic regression, centralized MLP, FedAvg, FedProx, coordinate median, trimmed mean and FLTrust. Correctness, reproducible clean execution, recorded assumptions and completed clean-learning review. |
| E | Fixed-coefficient adaptive weighting and frozen client weighting using actual quality, risk and prior reputation. Poisoning evaluation follows availability of the attack harness. |
| F | NSGA-II with actual aggregation/fitness evaluation, evolutionary operators and measured search cost. |
| G | Complete matched comparisons and existing experimental matrix after applicable clean evidence and actual search profiling. |

FedProx's client optimization remains separate from aggregation. Coordinate-wise methods
may return no scalar weights. FLTrust's separate server-reference operation uses the
permitted trusted resource, records its procedure/access budget/cost, and does not give
ordinary client trainers access to trusted data. Phase D does not use fabricated assessments.

## Future implementation obligations

### Clean-learning review

Establish centralized behavior before near-IID FedAvg, then non-IID FedAvg. Every condition
advanced downstream needs applicable evidence: artifact identities, feature variant, task,
fold, scenario, configuration, seed, initialization, initial and subsequent measurements,
class support, sanity comparisons and software checks. Investigate weak learning, instability
and unexpectedly perfect performance.

Record acceptable behavior, unresolved behavior, or an explained negative finding with
evidence and limitations. Unresolved behavior blocks advancement. A negative finding cannot
override failed software or data acceptance. Do not impose a universal accuracy threshold
or claim convergence from stable metrics. The full experiment matrix stays in G.

### Diagnostics

The [interface definitions](interfaces.md#future-round-diagnostics) govern consistent
identity, loss, macro-F1, benign FPR, update magnitude, weight change, timing and failure
reporting. Phase D implements them through the existing diagnostics artifact reference;
serialization/version selection belongs to that implementation. E/F extend measurements
when their algorithms exist.

Keep data roles separate, use sample counts when combining observation-average losses,
retain fixed task vocabulary, represent unsupported measurements explicitly, align client
identities for weight comparisons, and distinguish participation changes. A zero parent
norm makes relative update magnitude undefined. Coordinate-wise methods have no required
scalar weights. Parallel cumulative training time differs from elapsed time; nested
optimizer timings must not be counted twice. No final-test metrics enter development.

### Profiling and budgets

D measures ordinary training/evaluation cost. F profiles one representative completed
search through actual candidate aggregation and inference before scaling G. Record
hardware/software/workload, requests/cache hits/actual evaluations/invalid candidates,
construction/inference/evolutionary overhead, total time, peak memory, separate warm-up,
and projected matrix cost with assumptions. Synthetic objectives or sorting-only timings
do not satisfy this requirement. Profiling must not alter experiment checkpoints.

Population 24 and eight offspring generations remain recorded development defaults.
Any budget revision is explicit, justified from development measurements, and recorded
before main comparisons. Budgets do not change silently between methods/seeds and are
never chosen using final-test outcomes.

### Real-data entry and resume

Use the existing load_training_ready loader with explicit dataset/task/fold/scenario
expectations, consuming its verified bundle. Reject absent, stale, incompatible or
ineligible inputs. No warning-only fallback, direct-CSV bypass or manual eligibility
override is permitted. Synthetic component tests cannot authorize real-data experiments.

Capture recovery remains at the [agreed evidence-review checkpoint](../reports/generated/capture_recovery_review_2026-10-03/review.md).
The [B/C acceptance report](../reports/generated/phase_bc_stabilization_acceptance_2026-10-03.md)
records 89 passing tests and the unresolved real-data gate; this refinement does not update
that historical count or claim a real-data PASS.

## Implemented areas preserved

- src/edgefl/pipelines/catalog.py: existing dependency order and planned D-H availability.
- src/edgefl/contracts/interfaces.py and records.py: existing trainer/scorer/aggregator interfaces, optional assessments/trusted resource, diagnostics reference and optional scalar weights.
- src/edgefl/client_data/storage.py and Phase C orchestration: existing source-verified loader and acceptance logic.
- Phase A-C algorithms, tests, configurations, schemas and all current contract versions.
- B/C stabilization plan, recovery review, historical acceptance reports and generated artifacts.
- Older workflow/start documents, already superseded by the final roadmap where they conflict.
- Existing research defaults, hypotheses, endpoints and experiment matrix.

No new workflow engine, training modules, capability registry, record fields, dependencies,
test fixtures, or production commands are introduced. Future source/protocol snapshots
include these document revisions; historical snapshots are not rewritten.

## Verification and acceptance

### Documentation change acceptance

- Step 17 has no E/F prerequisite; FLTrust remains in D with explicit trusted-data use.
- Clean evidence and blocking/review dispositions agree across the documents.
- Diagnostic definitions have one detailed home in interfaces.md and consistent references.
- Actual profiling precedes scaling G without becoming a prerequisite for D.
- Every real-data entry/resume retains the source-verified gate.
- The change set is limited to the seven documents above; previous unrelated work is preserved.
- Local links resolve and implemented availability remains accurate.

Do not add tests that match documentation wording or rerun the full suite for this
documentation-only refinement. Preserve the historical 89-test evidence. Documentation
acceptance does not establish any future runtime behavior or real-data eligibility.

### Tests in the owning implementation phases

- D: baselines without assessments, FLTrust's permitted server-data path, known-answer aggregation and metric calculations, undefined metrics, rejected stale/mismatched inputs and final-test exclusion.
- Diagnostics: unequal client sizes, zero parent norm, changing participants, absent scalar weights, failed rounds and timing without double counting.
- F: actual model inference during profiling, distinct cache/evaluation counts, explicit invalid candidates and no profiling mutation of experiment checkpoints or budgets.
- Experimental acceptance: report regression evidence, real-data acceptance and clean-learning review separately; all applicable requirements must be satisfied.
