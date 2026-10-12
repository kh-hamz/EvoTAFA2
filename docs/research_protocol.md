# Research protocol — edgeiiot-fl-v1

## Data protocol revision: edgeiiot-fl-v2, 2026-10-03

Phase A foundation records remain version 1.0. Active Phase B/C data contracts are v2.
The [stabilization plan](phase_bc_stabilization_plan.md) permits sparse server-scored clients,
requires complete local class-exposure tables, names supported-class closed-set scopes and restricts
unseen-attack Protocol B to binary evaluation. Universal 100/80/20 binary minima are superseded.
Per-client near-IID size deviation is <=0.10 and original-label total variation against the selected
client pool is <=0.05. Exhausted search does not prove infeasibility. Recovery requires evidence review.
Primary hypotheses/endpoints are unchanged; revised data scopes must be reported explicitly.

Status: Phase A research contract established. Numerical defaults remain development
settings until the Phase H experimental freeze. A changed hypothesis or endpoint
requires a new protocol revision and a recorded rationale before final-test access.

## Readiness specification refinement, 2026-10-04

The [Phase D readiness specification](phase_d_readiness_refinement_plan.md) clarifies
phase ownership and development evidence. Foundation record version 1.0, B/C v2 data
contracts, hypotheses, endpoints, and selected dataset scopes are unchanged. These
documentation changes are captured in future protocol/source fingerprints; historical
snapshots are preserved.

## Primary question and endpoints

Does NSGA-II client weighting improve the detection–robustness tradeoff under
synthetic non-IID Edge-IIoTset data and controlled client poisoning?

The co-primary endpoints are task macro-F1 (higher is better) and benign FPR
(lower is better). Report paired differences for both. An improvement in macro-F1
with increased FPR is a tradeoff, not an unconditional win. Report the full result,
including uncertainty, rather than selecting whichever endpoint favors the method.

The primary development setting is ten clients, full participation, alpha 0.5,
20% compromised clients, attacks active from round 11, and five matched seeds.
Binary is the integration task; multiclass is required for attack-class expertise
claims. Each attack is a separate condition. Clean results are always included.

## Hypotheses

| ID | Hypothesis | Required evidence |
|---|---|---|
| H1 | NSGA-II improves the macro-F1/FPR tradeoff under poisoning versus FedAvg and fixed-coefficient weighting. | Matched paired seed results, clean-to-poisoned degradation, both primary endpoints. |
| H2 | Current-risk scoring reduces harmful update influence without systematically suppressing honest specialists. | Risk ablations, malicious aggregate weight, honest-specialist false alarms and class performance. |
| H3 | Historical reputation improves resilience to late or intermittent poisoning. | Remove-reputation ablation, attack-onset/recovery traces, identical attack schedules. |
| H4 | Expertise-informed initialization benefits rare-class behavior or search efficiency. | Remove-expertise initialization ablation, per-class support and performance, equal evaluation budgets. |
| H5 | Benefits depend on trusted-data coverage and search cost. | Trusted-size/missing-class sensitivity, random-search comparison, server compute accounting. |
| H6 | The aggregation method replicates on another dataset. | Independent CICIoT2023 training and evaluation with matched method definitions. |

A null or negative result is a valid outcome. No algorithm must win to pass a
software gate. No significance threshold or minimum effect is invented after test results.

## Comparators and fairness

Use majority/logistic/centralized models for learning sanity checks. FL comparisons
include FedAvg, FedProx, coordinate median, trimmed mean, fixed-coefficient adaptive
weighting, frozen client weighting, NSGA-II, and FLTrust with disclosed trusted-data access.
FedProx changes local optimization; it is not merely an aggregation weight rule.
Coordinate-wise methods need not have one scalar weight per client.

Phase D owns the sanity models, FedAvg, FedProx, median, trimmed mean, and FLTrust.
Its completion requires clean execution, correctness evidence, assumptions, and
clean-learning review for those methods. Phase E owns fixed-coefficient adaptive and
frozen client weighting; Phase F owns NSGA-II; Phase G owns the complete matched matrix.
Phase D completion does not depend on E/F implementations or poisoning evaluation.

FLTrust's server-reference operation uses the permitted trusted resource and records
its training procedure, access budget, and server cost. It does not grant trusted-data
access to ordinary client trainers. Match the trusted-data access budget with other
trusted-data methods and disclose the different use. Baselines receive no fabricated
quality, risk, or reputation values.

Match global and local manifests, preprocessing, feature order, initialization,
client availability, local training budget, and attack randomness within each condition.
Give methods comparable development tuning opportunities and report search cost
separately. Do not reuse later-round model deltas across diverged method trajectories.

## Evaluation protocols and allowed data use

Protocol A uses verified session/time-separated closed-set data. Protocol B holds
whole captures out and distinguishes benign-source generalization from binary
unseen-attack detection. Neither grouping feasibility nor 15-class support is assumed.

Approximate global allocation is 70% client pool, 7.5% trusted, 7.5% selection
validation, 15% final test. Local validation is approximately 20% within each client.
Group integrity outranks exact percentages. Only actual local-training rows fit
preprocessing. Binary and multiclass share applicable manifests, never target predictors.

Trusted data is an adaptive optimization resource. It supports client scoring and
candidate fitness. Selection validation supports tuning and checkpoint selection.
FLTrust may also use the permitted trusted resource for its server-reference update;
this is recorded separately from client training and candidate evaluation.
The locked test is accessible only to the final evaluation stage after protocol freeze.

Every real-data model experiment, including a resumed run, requires a fresh
source-verified Phase C PASS matching its dataset, task, fold, and scenario through
load_training_ready. Failed, stale, incompatible, or absent acceptance blocks execution.
Synthetic component tests do not authorize real-data experiments. Capture-recovery
approval and real B-to-C acceptance remain separate pending requirements.

The objectives minimize: one minus macro-F1, benign FPR, weighted current risk,
and weighted historical unreliability. Communication is measured separately.
Risk and reputation may be redundant; ablations must test their independent value.

## Clean-learning evidence and review

Establish centralized behavior before near-IID FedAvg, then non-IID FedAvg, before
poisoning and adaptive aggregation. For every condition advanced downstream, retain
an applicable report identifying its training-ready artifacts, feature variant, task,
fold, scenario, configuration, seed, initialization, and model checkpoints.

Record initial and subsequent training/evaluation measurements, class support, and
comparisons with the sanity baselines. Include finite-loss/update checks, known-answer
aggregation and single-client equivalence evidence where applicable. Investigate weak
learning, unexplained instability, and suspiciously perfect performance.

The review records evidence, investigation, conclusion, limitations, and one disposition:

- Acceptable learning behavior: the condition may advance.
- Unresolved behavior: advancement remains blocked.
- Explained negative finding: the condition may advance with documented limitations
  and restricted claims after investigation.

An explained negative finding cannot override broken software checks or a failed data
gate. No universal score threshold, monotonic-improvement requirement, or requirement
that the proposed method wins is imposed. Near-IID label/size balance does not imply
identical feature distributions. Empirical stability is not a convergence proof.
The full Phase G matrix remains in G; this review does not move it into D.

Use the [common diagnostic definitions](interfaces.md#future-round-diagnostics).
Report training, trusted-panel, and selection-validation measurements separately.
Selection validation supports review and checkpoint selection; trusted scores reflect
adaptive optimization. Local metrics lacking required support are explicitly unavailable.
Across matched seeds, retain individual results and the existing statistical summaries;
identify the reported checkpoint/round and never mix data roles or scenarios in one curve.

## Threat model and claim boundaries

The coordinator and trusted pool are assumed honest. Simulated compromised clients
can alter designated training labels or submitted deltas. They cannot alter trusted
or evaluation data. The defense receives no malicious-identity labels. Simulator
truth is available only to the attack harness and evaluation diagnostics.

Registered sample counts come from simulation manifests, not unverified Byzantine
client reports. Stable client identities are assumed. Secure aggregation, differential
privacy, adaptive adversary knowledge, and collusion require separately versioned studies.

Synthetic label-skew clients do not establish real organizational/device ownership.
Server-known preprocessing and visible individual updates do not establish a
privacy-preserving deployment. Whole-capture unsupported classes cannot be claimed
as ordinary closed-set results. CICIoT2023 replication trains new models; it is not
cross-dataset transfer of the Edge-IIoT model. VM deployment is a systems demonstration.

## Statistics and freeze policy

Use matched seeds 11, 22, 33, 44, 55, with a separate global split base seed of 42.
Report seed observations, mean, standard deviation, paired differences and 95%
intervals. Seed uncertainty is conditional on a fixed global split. For observational
uncertainty use session/capture clusters where independent-group support allows it;
packet counts must not be treated as independent replicates.

Freeze manifests, feature rules, numerical definitions, attack specifications,
selection rules, tuning budgets and eligible checkpoints before test evaluation.
Implementation details reserved for later phases must be resolved using development
evidence, recorded in configuration/protocol revisions, and never chosen from test scores.

Profile ordinary training and evaluation during D. Before scaling G, F must profile
one representative completed search using actual candidate aggregation and inference.
Record environment, workload, candidate/cache/evaluation/failure counts, construction,
inference and search overhead, total time and peak memory, with initialization/warm-up
separate. Publish projected matrix costs and assumptions alongside measured costs.
Synthetic fitness or sorting-only benchmarks do not establish actual search cost.

Population 24 and eight offspring generations remain development defaults. Any
measurement-based budget revision must be explicit before main comparisons, consistent
with matched comparisons, and recorded with its rationale. No automatic budget changes
between methods or seeds and no budget selection from final-test results are permitted.
The existing Phase H final freeze remains the boundary for final-test access.

## Known feasibility issues handed to Phase B

DNN full-file counts are prior audit expectations, not freshly verified results.
MITM source frame.time values and incomplete dates require semantic verification.
DNN membership must be preserved when recovering provenance; concatenating raw
source CSVs is a different benchmark. Exact duplicate observations and equal reduced
feature vectors are different concepts. Phase A makes no claim that preprocessing
or split feasibility has passed.
