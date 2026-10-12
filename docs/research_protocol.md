# Research protocol — edgeiiot-fl-v1

Status: Phase A research contract established. Numerical defaults remain development
settings until the Phase H experimental freeze. A changed hypothesis or endpoint
requires a new protocol revision and a recorded rationale before final-test access.

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
The locked test is accessible only to the final evaluation stage after protocol freeze.

The objectives minimize: one minus macro-F1, benign FPR, weighted current risk,
and weighted historical unreliability. Communication is measured separately.
Risk and reputation may be redundant; ablations must test their independent value.

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

## Known feasibility issues handed to Phase B

DNN full-file counts are prior audit expectations, not freshly verified results.
MITM source frame.time values and incomplete dates require semantic verification.
DNN membership must be preserved when recovering provenance; concatenating raw
source CSVs is a different benchmark. Exact duplicate observations and equal reduced
feature vectors are different concepts. Phase A makes no claim that preprocessing
or split feasibility has passed.
